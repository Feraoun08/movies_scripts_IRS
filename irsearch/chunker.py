"""Scene -> chunks (the searchable retrieval unit).

Rules (all configurable through ChunkConfig):
  * a scene of <= single_chunk_max_words words stays ONE chunk (short scenes are never split);
  * a longer scene is cut into n near-equal chunks of at most max_words words with an overlap of
    ~overlap_words words between consecutive chunks (default: 250-350 words, 50 overlap);
  * cut points are snapped to the nearest line boundary (end of a dialogue block or action
    paragraph) within snap_window_words, so a speech is not split when it can be avoided;
  * a chunk never crosses a scene boundary, and always carries (movie_id, scene_idx, part_idx).

Each chunk has two texts: `text` (readable, original casing/line breaks, speaker names included)
and `index_text` (what will be tokenized: no transitions, and speaker labels handled according to
Config.speaker_mode).
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from typing import NamedTuple

from .config import ChunkConfig, Config
from .models import SPEAKER, TRANSITION, Chunk, Scene


class _W(NamedTuple):
    text: str
    line: int
    kind: str
    speaker: str | None


def scene_words(scene: Scene) -> tuple[list[_W], list[int]]:
    """Flatten a scene into words + the list of line-end offsets (candidate cut points)."""
    words: list[_W] = []
    boundaries: list[int] = []
    for li, line in enumerate(scene.lines()):
        before = len(words)
        for w in line.text.split():
            words.append(_W(w, li, line.kind, line.speaker))
        if len(words) > before:
            boundaries.append(len(words))
    return words, boundaries


def _snap(ideal: int, boundaries: list[int], lo: int, hi: int) -> int:
    lo = min(lo, hi)
    a, b = bisect_left(boundaries, lo), bisect_right(boundaries, hi)
    cands = boundaries[a:b]
    if cands:
        return min(cands, key=lambda x: abs(x - ideal))
    return max(lo, min(ideal, hi))


def plan_spans(n_words: int, boundaries: list[int], cfg: ChunkConfig) -> list[tuple[int, int]]:
    """Return [(start, end), ...] word spans covering the scene."""
    if n_words == 0:
        return []
    if n_words <= cfg.single_chunk_max_words:
        return [(0, n_words)]
    mx, ov = cfg.max_words, cfg.overlap_words
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        remaining = n_words - start
        if remaining <= mx:
            spans.append((start, n_words))
            return spans
        # smallest number of chunks that covers `remaining` with overlap, then equal sizes
        n_rem = math.ceil((remaining - ov) / (mx - ov))
        size = (remaining + (n_rem - 1) * ov) / n_rem
        ideal = start + round(size)
        lo = start + max(ov + 1, int(size) - cfg.snap_window_words)
        hi = min(start + mx, ideal + cfg.snap_window_words)
        end = _snap(ideal, boundaries, lo, hi)
        spans.append((start, end))
        start = end - ov


def _readable(ws: list[_W]) -> str:
    out: list[str] = []
    cur = None
    for w in ws:
        if cur is None:
            out.append(w.text)
        elif w.line != cur:
            out.append("\n" + w.text)
        else:
            out.append(" " + w.text)
        cur = w.line
    return "".join(out)


def _index_view(ws: list[_W], speaker_mode: str) -> tuple[str, list[str]]:
    speakers: list[str] = []
    seen: set[str] = set()
    for w in ws:
        if w.speaker and w.speaker not in seen:
            seen.add(w.speaker)
            speakers.append(w.speaker)

    tokens: list[str] = []
    for w in ws:
        if w.kind == TRANSITION:
            continue
        if w.kind == SPEAKER and speaker_mode != "keep":
            continue
        tokens.append(w.text)
    if speaker_mode == "unit_once":
        tokens.extend(speakers)  # each distinct name once per chunk: searchable, no tf inflation
    return " ".join(tokens), speakers


def chunk_scene(scene: Scene, movie_id: str, cfg: Config) -> list[Chunk]:
    words, boundaries = scene_words(scene)
    spans = plan_spans(len(words), boundaries, cfg.chunk)
    chunks: list[Chunk] = []
    for part, (a, b) in enumerate(spans):
        ws = words[a:b]
        index_text, speakers = _index_view(ws, cfg.speaker_mode)
        chunks.append(
            Chunk(
                movie_id=movie_id,
                scene_idx=scene.idx,
                part_idx=part,
                n_parts=len(spans),
                text=_readable(ws),
                index_text=index_text,
                speakers=speakers,
                n_words=len(ws),
            )
        )
    return chunks
def chunk_scenes(scenes, movie_id, cfg=None):
    """
    Convert all scenes of one movie into searchable chunks.

    Each scene is processed independently.
    Chunks from different scenes are never mixed.
    """
    all_chunks = []

    for scene in scenes:
        chunks = chunk_scene(
            scene=scene,
            movie_id=movie_id,
            cfg=cfg
        )

        all_chunks.extend(chunks)

    return all_chunks
