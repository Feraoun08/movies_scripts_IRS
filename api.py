"""HTTP API for CinemaSearch.

Run from the project root:
    python -m uvicorn api:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from irsearch.service import CinemaSearchService

CORPUS_DIR = os.environ.get("CINEMASEARCH_CORPUS", "out_full")
DEFAULT_METHOD = os.environ.get("CINEMASEARCH_METHOD", "bm25")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = CinemaSearchService(CORPUS_DIR, DEFAULT_METHOD)
    yield
    app.state.service.close()


app = FastAPI(title="CinemaSearch", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def service_of(request: Request) -> CinemaSearchService:
    return request.app.state.service


# Endpoints are async on purpose: they run on the same thread that
# opened the SQLite connection, which avoids cross-thread errors.

@app.get("/movies")
async def search_movies(
    request: Request,
    title: str | None = None,
    year: int | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    return service_of(request).search_movies(
        title=title,
        year=year,
        min_rating=min_rating,
        max_rating=max_rating,
        limit=limit,
    )


@app.get("/scenes/search")
async def search_scenes(
    request: Request,
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50),
    method: Literal["bm25", "tfidf"] | None = None,
    year: int | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
    title: str | None = None,
):
    try:
        return service_of(request).search_scenes(
            q,
            top_k=top_k,
            method=method,
            year=year,
            min_rating=min_rating,
            max_rating=max_rating,
            title=title,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/scenes/{movie_id}/{scene_idx}")
async def get_scene(request: Request, movie_id: str, scene_idx: int):
    scene = service_of(request).get_scene(movie_id, scene_idx)

    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found.")

    return scene