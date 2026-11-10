
from pathlib import Path

from irsearch.config import Config
from irsearch.dataset import DatasetConfig, iter_movies
from irsearch.parser import parse_screenplay
from irsearch.chunker import chunk_scenes
from irsearch.preprocess import Preprocessor
from irsearch.index import InvertedIndex


# Dataset location
DATASET_ROOT = Path(r"C:\Users\User\Downloads\movies_script")

# Start with 30 movies to keep the test manageable
NUMBER_OF_MOVIES = 30

# Save the resulting index separately from your future production index
SAVE_INDEX = True
OUTPUT_INDEX = Path("out") / "index_real_test.pkl"


def main():
    print("=" * 60)
    print("REAL DATASET - INVERTED INDEX TEST")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    # 1. Initialize the configuration and components
    config = Config()

    preprocessor = Preprocessor(config.preprocess)
    index = InvertedIndex(config)

    # Counters
    movies_processed = 0
    total_scenes = 0
    total_chunks = 0
    total_tokens = 0
    empty_chunks = 0

    # 2. Load real movies
    movies = iter_movies(
        data_dir=DATASET_ROOT,
        ds=DatasetConfig(),
        limit=NUMBER_OF_MOVIES
    )

    # 3. Process each movie
    for movie, screenplay_text in movies:

        movies_processed += 1

        print(
            f"\n[{movies_processed}/{NUMBER_OF_MOVIES}] "
            f"Processing: {movie.title}"
        )

        # Parse screenplay into scenes
        scenes = parse_screenplay(
            screenplay_text,
            config.parser
        )

        total_scenes += len(scenes)

        # Split scenes into chunks
        chunks = chunk_scenes(
            scenes,
            movie.movie_id,
            config
        )

        print(f"    Scenes: {len(scenes)}")
        print(f"    Chunks: {len(chunks)}")

        # 4. Preprocess and index every chunk
        for chunk in chunks:

            tokens = preprocessor.process(chunk.index_text)

            if not tokens:
                empty_chunks += 1

            total_tokens += len(tokens)

            # Each chunk becomes one document in the index
            doc_id = total_chunks

            index.add_document(
                doc_id=doc_id,
                tokens=tokens,
                movie_id=movie.movie_id,
                title=movie.title,
                scene_idx=chunk.scene_idx,
                part_idx=chunk.part_idx
            )

            total_chunks += 1

    # 5. Finalize the index
    index.finalize()

    # 6. Display dataset processing statistics
    print("\n")
    print("=" * 60)
    print("DATASET PROCESSING RESULTS")
    print("=" * 60)

    print(f"Movies processed:       {movies_processed}")
    print(f"Scenes detected:        {total_scenes}")
    print(f"Chunks created:         {total_chunks}")
    print(f"Empty chunks:           {empty_chunks}")
    print(f"Processed tokens:       {total_tokens}")

    # 7. Display inverted index statistics
    stats = index.stats()

    print("\n")
    print("=" * 60)
    print("INVERTED INDEX STATISTICS")
    print("=" * 60)

    for key, value in stats.items():
        print(f"{key:25}: {value}")

    # 8. Test some terms
    sample_terms = [
        "love",
        "kill",
        "murder",
        "police",
        "father",
        "mother"
    ]

    print("\n")
    print("=" * 60)
    print("SAMPLE TERM LOOKUPS")
    print("=" * 60)

    for term in sample_terms:

        document_frequency = index.df(term)
        collection_frequency = index.cf(term)

        postings = list(index.iter_postings(term))

        print(f"\nTerm: {term}")
        print(f"Document frequency (DF): {document_frequency}")
        print(f"Collection frequency (CF): {collection_frequency}")
        print(f"Number of postings: {len(postings)}")
        print(f"First postings: {postings[:5]}")

    # 9. Inspect one document
    if total_chunks > 0:

        print("\n")
        print("=" * 60)
        print("DOCUMENT METADATA TEST")
        print("=" * 60)

        print(index.doc_info(0))

    # 10. Save the index
    if SAVE_INDEX:

        index.save(OUTPUT_INDEX)

        print("\n")
        print("=" * 60)
        print("INDEX SAVED SUCCESSFULLY")
        print("=" * 60)

        print(f"Location: {OUTPUT_INDEX.resolve()}")

    # 11. Basic integrity checks
    assert movies_processed > 0, "No movies were processed!"
    assert total_chunks == stats["N_documents"]
    assert stats["vocabulary_size"] > 0
    assert stats["total_terms"] == total_tokens

    print("\n")
    print("=" * 60)
    print("ALL REAL-DATA CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
