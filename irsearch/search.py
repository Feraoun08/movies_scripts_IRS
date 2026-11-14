
"""Scene-level search engine for CineSearch.

Searches indexed screenplay chunks internally, groups matching
chunks by their original scene, and returns complete readable scenes.

Supports:
    - BM25 ranking
    - TF-IDF ranking
    - Movie metadata filters
    - Complete scene retrieval
"""

from __future__ import annotations

from typing import Any

from .weighting import WeightingModel, BM25Model


class SearchEngine:
    """Search screenplay scenes and return ranked complete scenes."""

    def __init__(
        self,
        index,
        store,
        preprocessor,
        ranking_method: str = "bm25",
        tf_mode: str = "raw",
    ) -> None:

        self.index = index
        self.store = store
        self.preprocessor = preprocessor
        self.ranking_method = ranking_method

        if ranking_method == "bm25":

            self.weighting = BM25Model(index)

        elif ranking_method == "tfidf":

            self.weighting = WeightingModel(
                index=index,
                tf_mode=tf_mode,
            )

        else:
            raise ValueError(
                "ranking_method must be 'bm25' or 'tfidf'."
            )

    # --------------------------------------------------
    # FILTERING
    # --------------------------------------------------

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
        """Check whether a scene's movie satisfies the filters."""

        if year is not None and result.get("year") != year:
            return False

        if genre:
            genres = result.get("genres") or []

            if isinstance(genres, str):
                genres = [genres]

            if not any(
                genre.casefold() in str(item).casefold()
                for item in genres
            ):
                return False

        rating = result.get("imdb_rating")

        if min_rating is not None:
            if rating is None or rating < min_rating:
                return False

        if max_rating is not None:
            if rating is None or rating > max_rating:
                return False

        if title:
            movie_title = result.get("title") or ""

            if title.casefold() not in movie_title.casefold():
                return False

        return True


        MAX_CHUNK_LIMIT = 8000

    def _rank_and_group(self, query_tokens, limit):
        ranked = self.weighting.score(query_tokens=query_tokens, top_k=limit)

        if not ranked:
            return {}, True

        chunks = self.store.get_chunks([doc_id for doc_id, _ in ranked])
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

        grouped = {}
        for doc_id, score in ranked:
            chunk = chunks_by_id.get(doc_id)
            if chunk is None:
                continue

            key = (chunk["movie_id"], chunk["scene_idx"])
            entry = grouped.setdefault(
                key, {"score": score, "matched_chunks": []}
            )
            entry["matched_chunks"].append({
                "chunk_id": doc_id,
                "part_idx": chunk["part_idx"],
                "score": score,
                "text": chunk["text"],
            })

        exhausted = limit is None or len(ranked) < limit
        return grouped, exhausted

    # --------------------------------------------------
    # SCENE SEARCH
    # --------------------------------------------------

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
        """
        Search screenplay content and return complete scenes.

        Results are grouped by movie and scene. If multiple chunks
        match the same scene, that scene appears only once.

        The scene score is the highest matching chunk score.
        """

        # --------------------------------------------------
        # STEP 1: VALIDATE INPUT
        # --------------------------------------------------

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer or None."
            )

        if min_rating is not None and max_rating is not None:
            if min_rating > max_rating:
                raise ValueError(
                    "min_rating cannot be greater than max_rating."
                )

        # --------------------------------------------------
        # STEP 2: PREPROCESS QUERY
        # --------------------------------------------------

        query_tokens = self.preprocessor.process(query)

        if not query_tokens:
            return []

        # --------------------------------------------------
        # STEP 3: RANK MATCHING CHUNKS
        # --------------------------------------------------

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

        if filters_active or top_k is None:
            limit = None
        else:
            limit = top_k * 20

        while True:
            grouped_scenes, exhausted = self._rank_and_group(
                query_tokens, limit
            )

            if (
                exhausted
                or len(grouped_scenes) >= top_k
                or limit >= self.MAX_CHUNK_LIMIT
            ):
                break

            limit = min(limit * 4, self.MAX_CHUNK_LIMIT)

        if not grouped_scenes:
            return []

        scene_keys = list(grouped_scenes.keys())
        # --------------------------------------------------
        # STEP 6: RETRIEVE COMPLETE SCENES FROM SQLITE
        # --------------------------------------------------

        complete_scenes = self.store.get_scenes(scene_keys)

        results = []

        for scene in complete_scenes:

            scene_key = (
                scene["movie_id"],
                scene["scene_idx"],
            )

            match_info = grouped_scenes[scene_key]

            result = dict(scene)

            result["score"] = match_info["score"]
            result["ranking_method"] = self.ranking_method

            result["matched_chunks"] = match_info[
                "matched_chunks"
            ]

            result["matched_chunk_count"] = len(
                match_info["matched_chunks"]
            )

            results.append(result)

        # --------------------------------------------------
        # STEP 7: APPLY MOVIE METADATA FILTERS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # STEP 8: SORT AND RETURN UNIQUE SCENES
        # --------------------------------------------------

        results.sort(
            key=lambda result: (
                -result["score"],
                result["movie_id"],
                result["scene_idx"],
            )
        )

        if top_k is not None:
            results = results[:top_k]

        return results

    # --------------------------------------------------
    # QUERY INFORMATION
    # --------------------------------------------------

    def explain_query(self, query: str) -> dict[str, Any]:
        """Show how a query is processed."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        tokens = self.preprocessor.process(query)

        recognized_terms = [
            term
            for term in dict.fromkeys(tokens)
            if self.index.df(term) > 0
        ]

        result = {
            "original_query": query,
            "processed_tokens": tokens,
            "recognized_terms": recognized_terms,
            "ranking_method": self.ranking_method,
        }

        if self.ranking_method == "tfidf":
            result["query_vector"] = (
                self.weighting.transform_query(tokens)
            )

        return result

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return search engine statistics."""

        return {
            "index": self.index.stats(),
            "weighting": self.weighting.stats(),
            "database": self.store.counts(),
        }
