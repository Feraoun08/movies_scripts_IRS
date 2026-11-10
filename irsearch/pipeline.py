
"""Build and load the CineSearch corpus.

Build flow:
    Movie -> Scenes -> Chunks
                   |         |
                   |         +--> Readable text -> SQLite
                   |
                   +------------> Preprocessing -> Inverted index

SQLite stores original readable content.
The inverted index stores processed terms.
SQLite chunk_id and index doc_id always share the same ID.
"""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Callable

from .chunker import chunk_scene
from .config import Config
from .dataset import DatasetConfig, iter_movies
from .index import InvertedIndex
from .parser import parse_screenplay
from .preprocess import Preprocessor
from .store import CorpusStore


DB_NAME = "corpus.db"
INDEX_NAME = "index.pkl"
CONFIG_NAME = "config.json"
REPORT_NAME = "build_report.json"


def build(
    data_dir: str | Path,
    out_dir: str | Path,
    cfg: Config | None = None,
    ds: DatasetConfig | None = None,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Build the readable corpus and inverted index.

    Args:
        data_dir: Original dataset directory.
        out_dir: Directory where generated files are saved.
        cfg: CineSearch configuration.
        ds: Dataset configuration.
        limit: Optional maximum number of movies.
        log: Logging callback.

    Returns:
        A dictionary containing build statistics.
    """

    cfg = cfg or Config()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db_path = out / DB_NAME

    # Each build starts from scratch to preserve ID consistency.
    if db_path.exists():
        db_path.unlink()

    # Initialize components.
    preprocessor = Preprocessor(cfg.preprocess)

    # IMPORTANT:
    # The current InvertedIndex constructor accepts Config.
    index = InvertedIndex(cfg)

    t0 = time.time()

    # Counters.
    n_movies = 0
    n_scenes = 0
    n_no_boundary = 0
    n_skipped = 0
    n_no_meta = 0

    doc_id = 0
    chunk_words = []

    try:

        # The context manager guarantees database cleanup.
        with CorpusStore(db_path) as store:

            movies = iter_movies(
                data_dir,
                ds or DatasetConfig(),
                limit,
                source_preference=cfg.parser.source_preference,
                fallback_to_txt=cfg.parser.fallback_to_txt,
            )

            # Process every movie.
            for movie, screenplay_text in movies:

                log(f"\n[processing] {movie.title}")

                # 1. Parse screenplay into scenes.
                scenes = parse_screenplay(
                    screenplay_text,
                    cfg.parser,
                )

                if not scenes:
                    n_skipped += 1
                    log(f"[skip] No content parsed: {movie.title!r}")
                    continue

                # 2. Store movie metadata.
                store.add_movie(movie)
                n_movies += 1

                # Count movies with no useful metadata.
                has_metadata = any(
                    value is not None and value != []
                    for value in (
                        movie.year,
                        movie.genres,
                        movie.imdb_rating,
                        movie.metascore,
                        movie.imdb_votes,
                    )
                )

                if not has_metadata:
                    n_no_meta += 1

                # 3. Process each scene.
                for scene in scenes:

                    store.add_scene(movie.movie_id, scene)

                    n_scenes += 1

                    if not scene.has_boundary:
                        n_no_boundary += 1

                    # 4. Split the scene into searchable chunks.
                    chunks = chunk_scene(
                        scene,
                        movie.movie_id,
                        cfg,
                    )

                    for chunk in chunks:

                        # 5. Store original readable text.
                        store.add_chunk(
                            doc_id,
                            chunk,
                        )

                        # 6. Preprocess the index representation.
                        tokens = preprocessor.process(
                            chunk.index_text
                        )

                        # 7. Add the chunk to the inverted index.
                        index.add_document(
                            doc_id=doc_id,
                            tokens=tokens,
                            movie_id=movie.movie_id,
                            title=movie.title,
                            scene_idx=chunk.scene_idx,
                            part_idx=chunk.part_idx,
                        )

                        chunk_words.append(chunk.n_words)

                        # Both SQLite and the index use this ID.
                        doc_id += 1

                # Periodically commit to SQLite.
                if n_movies % 100 == 0:
                    store.commit()

                    elapsed = time.time() - t0

                    log(
                        f"[progress] {n_movies} movies | "
                        f"{doc_id} chunks | "
                        f"{elapsed:.1f}s"
                    )

            # Commit all remaining data.
            store.commit()

            # Get database counts before closing.
            store_counts = store.counts()

        # Database is now closed.

        # 8. Finalize and save the inverted index.
        index.finalize()

        index.save(out / INDEX_NAME)

        # 9. Save the configuration.
        config_data = cfg.to_dict()

        (out / CONFIG_NAME).write_text(
            json.dumps(config_data, indent=2),
            encoding="utf-8",
        )

        # 10. Validate database/index consistency.
        if store_counts["chunks"] != index.stats()["N_documents"]:
            raise RuntimeError(
                "Database/index mismatch: "
                f"{store_counts['chunks']} stored chunks, "
                f"{index.stats()['N_documents']} indexed documents."
            )

        if store_counts["movies"] != n_movies:
            raise RuntimeError(
                "Movie count mismatch between pipeline and database."
            )

        # 11. Calculate chunk size statistics.
        sorted_chunk_words = sorted(chunk_words)

        chunk_stats = {
            "min": (
                sorted_chunk_words[0]
                if sorted_chunk_words else 0
            ),
            "median": (
                sorted_chunk_words[len(sorted_chunk_words) // 2]
                if sorted_chunk_words else 0
            ),
            "max": (
                sorted_chunk_words[-1]
                if sorted_chunk_words else 0
            ),
        }

        # 12. Generate build report.
        report = {
            "movies": n_movies,
            "movies_skipped": n_skipped,
            "movies_without_metadata": n_no_meta,
            "scenes": n_scenes,
            "pseudo_scenes_without_boundaries": n_no_boundary,
            "chunks": doc_id,
            "processed_tokens": sum(
                index.doc_lengths
            ),
            "chunk_words": chunk_stats,
            "store_counts": store_counts,
            "index": index.stats(),
            "seconds": round(time.time() - t0, 1),
        }

        (out / REPORT_NAME).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        log("\n" + "=" * 60)
        log("CINESEARCH BUILD COMPLETED SUCCESSFULLY")
        log("=" * 60)

        log(f"Movies: {n_movies}")
        log(f"Scenes: {n_scenes}")
        log(f"Chunks: {doc_id}")
        log(f"Vocabulary: {index.stats()['vocabulary_size']}")
        log(f"Database: {db_path}")
        log(f"Index: {out / INDEX_NAME}")
        log(f"Time: {report['seconds']} seconds")

        return report

    except Exception:
        log("\n[error] Build failed. Check the traceback.")
        raise


def load(
    out_dir: str | Path,
) -> tuple[InvertedIndex, CorpusStore, Preprocessor, Config]:
    """
    Load an existing corpus.

    Returns:
        index, store, preprocessor, config
    """

    out = Path(out_dir)

    db_path = out / DB_NAME
    index_path = out / INDEX_NAME
    config_path = out / CONFIG_NAME

    # Check required files.
    required_files = (
        db_path,
        index_path,
        config_path,
    )

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Cannot load CineSearch corpus. Missing files:\n"
            + "\n".join(missing)
        )

    # Restore the exact configuration used during the build.
    config_data = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    cfg = Config.from_dict(config_data)

    # Load the index and readable corpus.
    index = InvertedIndex.load(index_path)

    store = CorpusStore(db_path)

    preprocessor = Preprocessor(cfg.preprocess)

    # Verify the loaded database and index.
    counts = store.counts()

    if counts["chunks"] != index.stats()["N_documents"]:
        store.close()

        raise RuntimeError(
            "Loaded database/index mismatch: "
            f"{counts['chunks']} chunks in SQLite, "
            f"{index.stats()['N_documents']} documents in index."
        )

    return index, store, preprocessor, cfg
