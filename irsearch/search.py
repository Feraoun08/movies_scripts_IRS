


from __future__ import annotations

from typing import Any

from .weighting import WeightingModel


class SearchEngine:
    """Search screenplay chunks and return ranked readable results."""

    def __init__(
        self,
        index,
        store,
        preprocessor,
        tf_mode: str = "raw",
    ) -> None:
        self.index = index
        self.store = store
        self.preprocessor = preprocessor

        # Build the TF-IDF document vectors.
        self.weighting = WeightingModel(
            index=index,
            tf_mode=tf_mode,
        )

   
    # FILTERING
   

    @staticmethod
    def _matches_filters(
        result: dict[str, Any],
        *,
        year: int | None = None,
        genre: str | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        title: str | None = None,
    ) -> bool:
        """Check whether a result satisfies the requested filters."""

        # Movie year.
        if year is not None and result.get("year") != year:
            return False

        # Genre: case-insensitive partial match.
        if genre:
            genres = result.get("genres") or []

            if isinstance(genres, str):
                genres = [genres]

            genre_matches = any(
                genre.casefold() in str(item).casefold()
                for item in genres
            )

            if not genre_matches:
                return False

        # IMDb rating.
        rating = result.get("imdb_rating")

        if min_rating is not None:
            if rating is None or rating < min_rating:
                return False

        if max_rating is not None:
            if rating is None or rating > max_rating:
                return False

        # Movie title: case-insensitive partial match.
        if title:
            movie_title = result.get("title") or ""

            if title.casefold() not in movie_title.casefold():
                return False

        return True

  
    # SEARCH
    

    def search(
        self,
        query: str,
        top_k: int | None = 10,
        *,
        year: int | None = None,
        genre: str | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        # Validate parameters.
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None.")

        if min_rating is not None and max_rating is not None:
            if min_rating > max_rating:
                raise ValueError(
                    "min_rating cannot be greater than max_rating."
                )

        
        # STEP 1: PREPROCESS THE QUERY
        

        query_tokens = self.preprocessor.process(query)

        if not query_tokens:
            return []

        
        # STEP 2: FIND RELEVANT DOCUMENTS
       

        filters_active = any(
            value is not None
            for value in (
                year,
                genre,
                min_rating,
                max_rating,
                title,
            )
        )

        # When filters are active, retrieve all matching documents
        # first so that filtering does not accidentally remove
        # relevant results before the filters are applied.
        search_limit = None if filters_active else top_k

        ranked_documents = self.weighting.score(
            query_tokens=query_tokens,
            top_k=search_limit,
        )

        if not ranked_documents:
            return []

        
        # STEP 3: RETRIEVE ORIGINAL CHUNK TEXT FROM SQLITE
        

        document_ids = [
            doc_id
            for doc_id, similarity in ranked_documents
        ]

        stored_chunks = self.store.get_chunks(document_ids)

        # SQLite returns readable chunks in the requested order.
        # Build a lookup to reconnect each chunk to its score.
        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in stored_chunks
        }

        
        # STEP 4: COMBINE SCORES AND CHUNK INFORMATION
        

        results = []

        for doc_id, similarity in ranked_documents:

            chunk = chunks_by_id.get(doc_id)

            # Skip documents that have no corresponding SQLite chunk.
            if chunk is None:
                continue

            result = dict(chunk)

            result["similarity"] = similarity

            results.append(result)

        
        # STEP 5: APPLY METADATA FILTERS
        

        if filters_active:
            results = [
                result
                for result in results
                if self._matches_filters(
                    result,
                    year=year,
                    genre=genre,
                    min_rating=min_rating,
                    max_rating=max_rating,
                    title=title,
                )
            ]

        
        # STEP 6: RETURN THE TOP RESULTS
        

        if top_k is not None:
            results = results[:top_k]

        return results

    
    # QUERY INFORMATION
    

    def explain_query(self, query: str) -> dict[str, Any]:
        """Show how a query is processed by the search engine."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        tokens = self.preprocessor.process(query)
        query_vector = self.weighting.transform_query(tokens)

        return {
            "original_query": query,
            "processed_tokens": tokens,
            "recognized_terms": list(query_vector.keys()),
            "query_vector": query_vector,
        }

    
    # STATISTICS
   

    def stats(self) -> dict[str, Any]:
        """Return search engine statistics."""

        return {
            "index": self.index.stats(),
            "weighting": self.weighting.stats(),
            "database": self.store.counts(),
        }



"""Search engine for CineSearch.

Connects:
    - Preprocessor: processes user queries.
    - InvertedIndex: retrieves candidate documents.
    - WeightingModel: calculates TF-IDF and cosine similarity.
    - CorpusStore: retrieves original readable screenplay chunks.

The index document ID and SQLite chunk ID must refer to the same chunk.


 
        Search screenplay chunks using TF-IDF and cosine similarity.

        Args:
            query:
                User's original text query.

            top_k:
                Maximum number of results.
                None returns all matching results.

            year:
                Optional exact movie release year.

            genre:
                Optional case-insensitive partial genre match.

            min_rating:
                Optional minimum IMDb rating.

            max_rating:
                Optional maximum IMDb rating.

            title:
                Optional case-insensitive partial movie title.

        Returns:
            A list of dictionaries ordered by decreasing similarity.
            Each result contains the readable chunk and movie metadata.
        """
