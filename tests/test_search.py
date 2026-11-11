
from irsearch.pipeline import load
from irsearch.search import SearchEngine


# Load the previously built index and database.
index, store, preprocessor, cfg = load("out")

try:
    # Initialize the search engine.
    engine = SearchEngine(
        index=index,
        store=store,
        preprocessor=preprocessor,
        tf_mode="raw",
    )

    # Search for a screenplay concept.
    results = engine.search(
            "family relationships",
    top_k=5,
    title="20th Century Women",
 
    )

    print("\nSEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\nResult #{i}")
        print(f"Document ID: {result['chunk_id']}")
        print(f"Movie: {result['title']}")
        print(f"Year: {result['year']}")
        print(f"Scene: {result['scene_idx']}")
        print(f"Similarity: {result['similarity']:.4f}")

        print("\nText:")
        print(result["text"][:500])

    print("\nSEARCH STATISTICS")
    print(engine.stats())

finally:
    store.close()
