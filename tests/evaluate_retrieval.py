
"""Run manual relevance evaluation for CineSearch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from irsearch.pipeline import load
from irsearch.search import SearchEngine
from irsearch.evaluation import evaluate_rankings


QRELS_PATH = Path(__file__).with_name("relevance_judgments.json")


def load_judgments(path: Path) -> dict[str, set[int]]:
    """Load manually created relevance judgments."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The relevance file must contain a JSON object.")

    judgments = {}

    for query, ids in data.items():

        if not isinstance(query, str) or not isinstance(ids, list):
            raise ValueError(
                "Each entry must have a query string and a list of IDs."
            )

        if any(
            not isinstance(doc_id, int) or isinstance(doc_id, bool)
            for doc_id in ids
        ):
            raise ValueError(
                f"Invalid document ID in relevance judgments for: {query}"
            )

        judgments[query] = set(ids)

    return judgments


def display_candidates(query: str, results: list[dict]) -> None:
    """Display search results to help with manual relevance judgments."""

    print("\n" + "=" * 75)
    print(f"QUERY: {query}")
    print("=" * 75)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Movie: {result['title']}")
        print(f"Year: {result['year']}")
        print(f"Scene: {result['scene_idx']}")
        print(f"Similarity: {result['similarity']:.4f}")

        text = result["text"].replace("\n", " ")

        print(f"Text: {text[:350]}...")
        print("-" * 75)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate CineSearch retrieval quality."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to retrieve per query (default: 20).",
    )

    parser.add_argument(
        "--qrels",
        type=Path,
        default=QRELS_PATH,
        help="Path to the relevance judgments JSON file.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    judgments = load_judgments(args.qrels)

    if not judgments:
        print("No queries found in the relevance judgments file.")
        return

    # Load the existing corpus and index.
    index, store, preprocessor, cfg = load("out")

    try:
        engine = SearchEngine(
            index=index,
            store=store,
            preprocessor=preprocessor,
            tf_mode="raw",
        )

        rankings = {}

        print("\nCINESEARCH RETRIEVAL EVALUATION")
        print("=" * 75)
        print(f"Number of queries: {len(judgments)}")
        print(f"Results per query: {args.top_k}")

        # Retrieve and display candidates for every query.
        for query in judgments:

            results = engine.search(
                query,
                top_k=args.top_k,
            )

            rankings[query] = [
                result["chunk_id"]
                for result in results
            ]

            display_candidates(query, results)

        # Keep only queries with actual relevance judgments.
        active_judgments = {
            query: ids
            for query, ids in judgments.items()
            if ids
        }

        if not active_judgments:
            print("\nNO EVALUATION METRICS YET")
            print("=" * 75)
            print("No relevant chunk IDs have been entered yet.")
            print("Review the displayed results and edit:")
            print(args.qrels)
            print("\nAdd relevant chunk IDs to each query's array.")
            return

        # Evaluate the ranked results.
        report = evaluate_rankings(
            rankings=rankings,
            relevance_judgments=active_judgments,
            k=args.top_k,
        )

        print("\n\nEVALUATION RESULTS")
        print("=" * 75)

        for row in report["per_query"]:

            print(f"\nQuery: {row['query']}")
            print(f"Relevant chunks: {row['relevant_documents']}")
            print(f"Precision@{args.top_k}: {row[f'precision@{args.top_k}']:.4f}")
            print(f"Recall@{args.top_k}: {row[f'recall@{args.top_k}']:.4f}")
            print(f"F1@{args.top_k}: {row[f'f1@{args.top_k}']:.4f}")
            print(f"AP@{args.top_k}: {row[f'ap@{args.top_k}']:.4f}")

        print("\n" + "=" * 75)
        print("OVERALL RESULTS")
        print("=" * 75)

        print(f"Evaluated queries: {report['evaluated_queries']}")
        print(f"Precision@{args.top_k}: {report[f'precision@{args.top_k}']:.4f}")
        print(f"Recall@{args.top_k}: {report[f'recall@{args.top_k}']:.4f}")
        print(f"F1@{args.top_k}: {report[f'f1@{args.top_k}']:.4f}")
        print(f"MAP@{args.top_k}: {report[f'map@{args.top_k}']:.4f}")

        print("\nEvaluation completed.")

    finally:
        store.close()


if __name__ == "__main__":
    main()
