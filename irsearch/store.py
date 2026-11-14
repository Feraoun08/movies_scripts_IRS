
"""Readable corpus store (SQLite): Movie -> Scenes -> Chunks.

The original readable screenplay and complete scenes are stored here.

The inverted index stores preprocessed terms. Search results
are resolved to their readable chunks and complete scenes using
the shared document IDs and scene identifiers.
"""

from __future__ import annotations

import json
import sqlite3

from pathlib import Path
from typing import Iterable

from .models import Chunk, MovieRecord, Scene


_SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    genres TEXT,
    imdb_rating REAL,
    metascore REAL,
    imdb_votes INTEGER,
    source TEXT
);

CREATE TABLE IF NOT EXISTS scenes (
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    scene_idx INTEGER NOT NULL,
    heading TEXT,
    int_ext TEXT,
    location TEXT,
    time_of_day TEXT,
    n_words INTEGER,
    has_boundary INTEGER,
    text TEXT,
    PRIMARY KEY (movie_id, scene_idx)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    movie_id TEXT NOT NULL REFERENCES movies(movie_id),
    scene_idx INTEGER NOT NULL,
    part_idx INTEGER NOT NULL,
    n_parts INTEGER NOT NULL,
    n_words INTEGER NOT NULL,
    speakers TEXT,
    text TEXT NOT NULL,
    FOREIGN KEY (movie_id, scene_idx)
        REFERENCES scenes(movie_id, scene_idx)
);

CREATE INDEX IF NOT EXISTS idx_chunks_scene
ON chunks(movie_id, scene_idx);

CREATE INDEX IF NOT EXISTS idx_movies_year
ON movies(year);

CREATE INDEX IF NOT EXISTS idx_movies_rating
ON movies(imdb_rating);
"""


class CorpusStore:
    """Manage the readable movie, scene and chunk corpus in SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row

        # Enable foreign-key enforcement.
        self.db.execute("PRAGMA foreign_keys = ON")

        self.db.executescript(_SCHEMA)

    
    # JSON HELPER
    

    @staticmethod
    def _json_list(value: str | None) -> list:
        """Safely decode a JSON list stored in SQLite."""

        if not value:
            return []

        try:
            result = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []

        return result if isinstance(result, list) else []

    
    # WRITING
    

    def add_movie(self, movie: MovieRecord) -> None:
        """Insert a movie or update its metadata if it exists."""

        self.db.execute(
            """
            INSERT INTO movies (
                movie_id,
                title,
                year,
                genres,
                imdb_rating,
                metascore,
                imdb_votes,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(movie_id) DO UPDATE SET
                title = excluded.title,
                year = excluded.year,
                genres = excluded.genres,
                imdb_rating = excluded.imdb_rating,
                metascore = excluded.metascore,
                imdb_votes = excluded.imdb_votes,
                source = excluded.source
            """,
            (
                movie.movie_id,
                movie.title,
                movie.year,
                json.dumps(movie.genres or []),
                movie.imdb_rating,
                movie.metascore,
                movie.imdb_votes,
                movie.source,
            ),
        )

    def add_scene(self, movie_id: str, scene: Scene) -> None:
        """Insert a scene or update it if it already exists."""

        int_ext, location, tod = scene.location

        self.db.execute(
            """
            INSERT INTO scenes (
                movie_id,
                scene_idx,
                heading,
                int_ext,
                location,
                time_of_day,
                n_words,
                has_boundary,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(movie_id, scene_idx) DO UPDATE SET
                heading = excluded.heading,
                int_ext = excluded.int_ext,
                location = excluded.location,
                time_of_day = excluded.time_of_day,
                n_words = excluded.n_words,
                has_boundary = excluded.has_boundary,
                text = excluded.text
            """,
            (
                movie_id,
                scene.idx,
                scene.heading,
                int_ext,
                location,
                tod,
                scene.n_words,
                int(scene.has_boundary),
                scene.text,
            ),
        )

    def add_chunk(self, chunk_id: int, chunk: Chunk) -> None:
        """Insert a readable chunk using its shared index document ID."""

        self.db.execute(
            """
            INSERT INTO chunks (
                chunk_id,
                movie_id,
                scene_idx,
                part_idx,
                n_parts,
                n_words,
                speakers,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                chunk.movie_id,
                chunk.scene_idx,
                chunk.part_idx,
                chunk.n_parts,
                chunk.n_words,
                json.dumps(chunk.speakers or []),
                chunk.text,
            ),
        )

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    
    # CHUNK RETRIEVAL
    

    def get_chunks(self, chunk_ids: Iterable[int]) -> list[dict]:
        """Retrieve readable chunks while preserving requested ID order."""

        ids = list(chunk_ids)

        if not ids:
            return []

        by_id = {}
        batch_size = 900

        for start in range(0, len(ids), batch_size):

            batch = ids[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)

            rows = self.db.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.movie_id,
                    c.scene_idx,
                    c.part_idx,
                    c.n_parts,
                    c.n_words,
                    c.speakers,
                    c.text,

                    m.title,
                    m.year,
                    m.genres,
                    m.imdb_rating,
                    m.metascore,
                    m.imdb_votes,

                    s.heading,
                    s.location,
                    s.time_of_day

                FROM chunks c

                JOIN movies m
                    ON m.movie_id = c.movie_id

                LEFT JOIN scenes s
                    ON s.movie_id = c.movie_id
                    AND s.scene_idx = c.scene_idx

                WHERE c.chunk_id IN ({placeholders})
                """,
                batch,
            ).fetchall()

            for row in rows:
                data = dict(row)

                data["speakers"] = self._json_list(
                    data["speakers"]
                )

                data["genres"] = self._json_list(
                    data["genres"]
                )

                by_id[data["chunk_id"]] = data

        return [
            by_id[chunk_id]
            for chunk_id in ids
            if chunk_id in by_id
        ]

    
    # COMPLETE SCENE RETRIEVAL
    

    def get_scene(
        self,
        movie_id: str,
        scene_idx: int,
    ) -> dict | None:
        """Retrieve one complete scene with its movie metadata."""

        row = self.db.execute(
            """
            SELECT
                s.movie_id,
                s.scene_idx,
                s.heading,
                s.int_ext,
                s.location,
                s.time_of_day,
                s.n_words,
                s.has_boundary,
                s.text,

                m.title,
                m.year,
                m.genres,
                m.imdb_rating,
                m.metascore,
                m.imdb_votes

            FROM scenes s

            JOIN movies m
                ON m.movie_id = s.movie_id

            WHERE s.movie_id = ?
              AND s.scene_idx = ?
            """,
            (movie_id, scene_idx),
        ).fetchone()

        if row is None:
            return None

        data = dict(row)
        data["genres"] = self._json_list(data["genres"])

        return data

    def get_scenes(
        self,
        scene_keys: Iterable[tuple[str, int]],
    ) -> list[dict]:
        """
        Retrieve multiple complete scenes.

        Args:
            scene_keys:
                Iterable of (movie_id, scene_idx) pairs.

        Returns:
            Complete scenes with movie metadata, preserving
            the requested scene order.
        """

        keys = list(dict.fromkeys(scene_keys))

        if not keys:
            return []

        by_key = {}
        batch_size = 400

        for start in range(0, len(keys), batch_size):

            batch = keys[start:start + batch_size]

            conditions = " OR ".join(
                "(s.movie_id = ? AND s.scene_idx = ?)"
                for _ in batch
            )

            parameters = [
                value
                for movie_id, scene_idx in batch
                for value in (movie_id, scene_idx)
            ]

            rows = self.db.execute(
                f"""
                SELECT
                    s.movie_id,
                    s.scene_idx,
                    s.heading,
                    s.int_ext,
                    s.location,
                    s.time_of_day,
                    s.n_words,
                    s.has_boundary,
                    s.text,

                    m.title,
                    m.year,
                    m.genres,
                    m.imdb_rating,
                    m.metascore,
                    m.imdb_votes

                FROM scenes s

                JOIN movies m
                    ON m.movie_id = s.movie_id

                WHERE {conditions}
                """,
                parameters,
            ).fetchall()

            for row in rows:

                data = dict(row)

                data["genres"] = self._json_list(
                    data["genres"]
                )

                key = (
                    data["movie_id"],
                    data["scene_idx"],
                )

                by_key[key] = data

        return [
            by_key[key]
            for key in keys
            if key in by_key
        ]

    
    # MOVIE RETRIEVAL
    

    def search_movies(
        self,
        title: str | None = None,
        year: int | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search movies using their metadata, independently
        of screenplay text.
        """

        if limit <= 0:
            raise ValueError("limit must be positive.")

        conditions = []
        parameters = []

        if title:
            conditions.append("LOWER(title) LIKE LOWER(?)")
            parameters.append(f"%{title}%")

        if year is not None:
            conditions.append("year = ?")
            parameters.append(year)

        if min_rating is not None:
            conditions.append("imdb_rating >= ?")
            parameters.append(min_rating)

        if max_rating is not None:
            conditions.append("imdb_rating <= ?")
            parameters.append(max_rating)

        where_clause = (
            "WHERE " + " AND ".join(conditions)
            if conditions else ""
        )

        rows = self.db.execute(
            f"""
            SELECT *
            FROM movies
            {where_clause}
            ORDER BY
                CASE WHEN imdb_rating IS NULL THEN 1 ELSE 0 END,
                imdb_rating DESC,
                title COLLATE NOCASE
            LIMIT ?
            """,
            (*parameters, limit),
        ).fetchall()

        results = []

        for row in rows:

            data = dict(row)

            data["genres"] = self._json_list(
                data["genres"]
            )

            results.append(data)

        return results

    
    # STATISTICS
    

    def counts(self) -> dict[str, int]:
        """Return the number of movies, scenes and chunks."""

        return {
            table: self.db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in ("movies", "scenes", "chunks")
        }

    
    # CONTEXT MANAGER
    

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()

        self.db.close()

        return False
