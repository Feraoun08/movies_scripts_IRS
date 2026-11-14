"""Application service for CinemaSearch: movie search and scene search."""

from __future__ import annotations

from typing import Any

from .pipeline import load
from .search import SearchEngine

METHODS = ("bm25", "tfidf")


class CinemaSearchService:
    """Loads the corpus once and exposes the two independent search modes."""

    def __init__(
        self,
        corpus_dir: str = "out_full",
        default_method: str = "bm25",
    ) -> None:

        if default_method not in METHODS:
            raise ValueError(f"default_method must be one of {METHODS}.")

        self.default_method = default_method
        self.index, self.store, self.preprocessor, self.config = load(corpus_dir)
        self._engines: dict[str, SearchEngine] = {}

    
    # ENGINES (created lazily: TF-IDF builds heavy vectors)
    

    def _engine(self, method: str | None) -> SearchEngine:
        method = method or self.default_method

        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")

        if method not in self._engines:
            self._engines[method] = SearchEngine(
                self.index,
                self.store,
                self.preprocessor,
                ranking_method=method,
            )

        return self._engines[method]

    
    # MODE 1: MOVIE SEARCH (metadata only)
    

    def search_movies(
        self,
        title: str | None = None,
        year: int | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        return self.store.search_movies(
            title=title or None,
            year=year,
            min_rating=min_rating,
            max_rating=max_rating,
            limit=limit,
        )

    
    # MODE 2: SCENE SEARCH (screenplay content, complete scenes)
    

    def search_scenes(
        self,
        query: str,
        top_k: int = 10,
        method: str | None = None,
        year: int | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        engine = self._engine(method)

        found = engine.search(
            query.strip(),
            top_k=top_k,
            year=year,
            min_rating=min_rating,
            max_rating=max_rating,
            title=title or None,
        )

        return {
            "query": query.strip(),
            "method": engine.ranking_method,
            "count": len(found),
            "results": [self._format_scene(scene) for scene in found],
        }

    def get_scene(self, movie_id: str, scene_idx: int) -> dict[str, Any] | None:
        return self.store.get_scene(movie_id, scene_idx)

    @staticmethod
    def _format_scene(scene: dict[str, Any]) -> dict[str, Any]:
        formatted = dict(scene)

        chunks = formatted.pop("matched_chunks", [])
        formatted["matched_chunk_ids"] = [c["chunk_id"] for c in chunks]

        return formatted

    
    # LIFECYCLE
    

    def close(self) -> None:
        self.store.close()
