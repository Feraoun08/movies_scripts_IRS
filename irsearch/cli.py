
"""Command-line interface for CineSearch.

Usage:
    python -m irsearch inspect <data_dir>
    python -m irsearch build <data_dir> --out <dir>
    python -m irsearch stats --out <dir>
    python -m irsearch term --out <dir> <words...>
"""

from __future__ import annotations

import argparse
import json

from .config import (
    ChunkConfig,
    Config,
    IndexConfig,
    ParserConfig,
    PreprocessConfig,
)
from .dataset import inspect_dataset
from .pipeline import build, load



# Helpers


def positive_int(value: str) -> int:
    """Accept only positive integers."""
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be a positive integer."
        )

    return number


def print_result(result) -> None:
    """Print dictionaries and lists as formatted JSON."""
    if isinstance(result, (dict, list)):
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    else:
        print(result)


def print_separator() -> None:
    print("-" * 65)



# INSPECT COMMAND

def run_inspect(args: argparse.Namespace) -> None:
    """Inspect the original dataset."""
    print("\nINSPECTING DATASET")
    print_separator()

    result = inspect_dataset(
        args.data_dir,
        sample=args.sample,
    )

    print_result(result)



# BUILD COMMAND


def run_build(args: argparse.Namespace) -> None:
    """Build the SQLite corpus and inverted index."""

    cfg = Config(
        parser=ParserConfig(
            source_preference=args.source,
        ),

        chunk=ChunkConfig(
            min_words=args.min_words,
            max_words=args.max_words,
            overlap_words=args.overlap,
        ),

        preprocess=PreprocessConfig(
            stem=not args.no_stem,
            drop_numbers=args.drop_numbers,
        ),

        speaker_mode=args.speaker_mode,

        index=IndexConfig(
            store_positions=args.positions,
        ),
    )

    print("\nSTARTING CINESEARCH BUILD")
    print_separator()

    report = build(
        data_dir=args.data_dir,
        out_dir=args.out,
        cfg=cfg,
        limit=args.limit,
    )

    print("\nBUILD REPORT")
    print_separator()
    print_result(report)



# STATS COMMAND


def run_stats(args: argparse.Namespace) -> None:
    """Display corpus and inverted-index statistics."""

    index, store, _, _ = load(args.out)

    try:
        statistics = {
            "index": index.stats(),
            "database": store.counts(),
        }

        print("\nCINESEARCH STATISTICS")
        print_separator()
        print_result(statistics)

    finally:
        store.close()



# TERM COMMAND


def run_term(args: argparse.Namespace) -> None:
    """
    Inspect a term in the inverted index.

    Displays:
        - Document frequency (DF)
        - Collection frequency (CF)
        - Term frequency (TF)
        - Matching screenplay chunks
    """

    index, store, preprocessor, _ = load(args.out)

    try:
        query = " ".join(args.words)

        # Apply the same preprocessing used during indexing.
        terms = list(dict.fromkeys(preprocessor.process(query)))

        if not terms:
            print("\nNo searchable terms after preprocessing.")
            return

        total_documents = index.stats()["N_documents"]

        print("\nTERM ANALYSIS")
        print_separator()

        for term in terms:

            postings = list(index.iter_postings(term))

            print(f"\nTerm: {term}")
            print(f"Document Frequency (DF): {index.df(term)}")
            print(f"Collection Frequency (CF): {index.cf(term)}")
            print(f"Total Documents: {total_documents}")

            if not postings:
                print("No documents contain this term.")
                continue

            # Select the documents with the highest TF.
            top_postings = sorted(
                postings,
                key=lambda item: item[1],
                reverse=True,
            )[:args.top]

            # Retrieve the original readable chunks.
            document_ids = [
                doc_id
                for doc_id, _ in top_postings
            ]

            chunks = store.get_chunks(document_ids)

            chunks_by_id = {
                chunk["chunk_id"]: chunk
                for chunk in chunks
            }

            print(f"\nTop {len(top_postings)} matching chunks:")
            print_separator()

            for doc_id, tf in top_postings:

                chunk = chunks_by_id.get(doc_id)

                if chunk is None:
                    continue

                snippet = (
                    chunk["text"]
                    .replace("\n", " ")
                    .strip()[:200]
                )

                print(
                    f"\nDocument ID: {doc_id} | TF: {tf}"
                )

                print(f"Movie: {chunk['title']}")

                print(
                    f"Scene: {chunk['scene_idx']} | "
                    f"Part: {chunk['part_idx'] + 1}/"
                    f"{chunk['n_parts']}"
                )

                print(f"Text: {snippet}...")

                print_separator()

    finally:
        store.close()



# ARGUMENT PARSER


def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="irsearch",
        description="CineSearch - Movie Screenplay Information Retrieval System",
        epilog=(
            "Example: python -m irsearch build "
            '"C:\\Users\\User\\Downloads\\movies_script" '
            "--out out --limit 30"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # INSPECT
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the original dataset.",
    )

    inspect_parser.add_argument(
        "data_dir",
        help="Path to the original dataset.",
    )

    inspect_parser.add_argument(
        "--sample",
        type=positive_int,
        default=40,
        help="Number of sample files to inspect.",
    )

    inspect_parser.set_defaults(func=run_inspect)

    # BUILD
    build_parser = subparsers.add_parser(
        "build",
        help="Build the corpus and inverted index.",
    )

    build_parser.add_argument(
        "data_dir",
        help="Path to the original dataset.",
    )

    build_parser.add_argument(
        "--out",
        required=True,
        help="Output directory.",
    )

    build_parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of movies to process.",
    )

    build_parser.add_argument(
        "--min-words",
        type=positive_int,
        default=250,
        help="Minimum target chunk size.",
    )

    build_parser.add_argument(
        "--max-words",
        type=positive_int,
        default=350,
        help="Maximum target chunk size.",
    )

    build_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words between chunks.",
    )

    build_parser.add_argument(
        "--speaker-mode",
        choices=["separate", "unit_once", "keep"],
        default="separate",
        help="How speaker names are handled during indexing.",
    )

    build_parser.add_argument(
        "--source",
        choices=["auto", "txt", "json"],
        default="auto",
        help="Preferred screenplay source.",
    )

    build_parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Disable Porter stemming.",
    )

    build_parser.add_argument(
        "--drop-numbers",
        action="store_true",
        help="Remove numerical tokens.",
    )

    build_parser.add_argument(
        "--positions",
        action="store_true",
        help="Store term positions for future phrase queries.",
    )

    build_parser.set_defaults(func=run_build)

    # STATS
    stats_parser = subparsers.add_parser(
        "stats",
        help="Display corpus and index statistics.",
    )

    stats_parser.add_argument(
        "--out",
        required=True,
        help="Directory containing the built corpus.",
    )

    stats_parser.set_defaults(func=run_stats)

    # TERM
    term_parser = subparsers.add_parser(
        "term",
        help="Inspect a term and retrieve matching chunks.",
    )

    term_parser.add_argument(
        "--out",
        required=True,
        help="Directory containing the built corpus.",
    )

    term_parser.add_argument(
 "--top",
        type=positive_int,
        default=3,
        help="Maximum number of matching chunks to display.",
    )

    term_parser.add_argument(
        "words",
        nargs="+",
        help="One or more words to investigate.",
    )

    term_parser.set_defaults(func=run_term)

    return parser



# MAIN


def main(argv: list[str] | None = None) -> None:

    parser = create_parser()
    args = parser.parse_args(argv)

    args.func(args)


if __name__ == "__main__":
    main()
