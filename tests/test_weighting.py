
from irsearch.pipeline import load
from irsearch.weighting import WeightingModel


def test_real_weighting():

    index, store, preprocessor, cfg = load("out")

    try:
        model = WeightingModel(index)

        # 1. Check model statistics.
        print("\nWEIGHTING STATISTICS")
        print(model.stats())

        assert model.n_documents == index.stats()["N_documents"]

        assert len(model.idf_values) == index.stats()["vocabulary_size"]

        # 2. Check a known term.
        print("\nTERM INFORMATION")
        print(model.explain_term("vendetta"))

        # 3. Transform a query.
        query = "vendetta revolution"

        tokens = preprocessor.process(query)

        query_vector = model.transform_query(tokens)

        print("\nQUERY TOKENS")
        print(tokens)

        print("\nQUERY VECTOR")
        print(query_vector)

        assert query_vector

        # 4. Rank the most relevant chunks.
        results = model.score(tokens, top_k=5)

        print("\nTOP 5 RESULTS")

        assert results

        chunks = store.get_chunks(
            [doc_id for doc_id, score in results]
        )

        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        for doc_id, similarity in results:

            chunk = chunks_by_id[doc_id]

            print("\n" + "=" * 60)

            print(f"Document ID: {doc_id}")
            print(f"Movie: {chunk['title']}")
            print(f"Scene: {chunk['scene_idx']}")
            print(f"Similarity: {similarity:.4f}")

            print(
                chunk["text"].replace("\n", " ")[:250]
            )

        # 5. Check similarity bounds.
        for doc_id, similarity in results:
            assert 0.0 <= similarity <= 1.0

        print("\nALL WEIGHTING CHECKS PASSED!")

    finally:
        store.close()


if __name__ == "__main__":
    test_real_weighting()
