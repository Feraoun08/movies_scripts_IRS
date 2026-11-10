

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple




HEADING = "heading"
ACTION = "action"
SPEAKER = "speaker"
PAREN = "paren"
DIALOGUE = "dialogue"
TRANSITION = "transition"


@dataclass
class MovieRecord:
    movie_id: str
    title: str

    year: int | None = None
    genres: list[str] = field(default_factory=list)

    imdb_rating: float | None = None
    metascore: float | None = None
    imdb_votes: int | None = None

    source: str = ""


@dataclass
class Element:
    kind: str
    text: str

    speaker: str | None = None

    
    annotation_type: str | None = None

    
    annotation_data: dict[str, Any] = field(default_factory=dict)


class Line(NamedTuple):
    kind: str
    text: str
    speaker: str | None


_HEADING_SPLIT = re.compile(
    r"\s*(INT\.?\s*/\s*EXT|EXT\.?\s*/\s*INT|I\s*/\s*E|INT|EXT)\.?\s*[-:]?\s*(.*)$"
)

_DASH = re.compile(r"\s+[-\u2013\u2014]+\s+")


def parse_heading(heading: str) -> tuple[str, str, str]:
    """Extract interior/exterior, location and time from a heading."""

    m = _HEADING_SPLIT.match(heading or "")

    if not m:
        return "", (heading or "").strip(), ""

    int_ext = re.sub(r"\s+", "", m.group(1)).replace(".", "")

    rest = m.group(2).strip()
    parts = _DASH.split(rest)

    if len(parts) > 1:
        return (
            int_ext,
            " - ".join(parts[:-1]).strip(),
            parts[-1].strip()
        )

    return int_ext, rest, ""


@dataclass
class Scene:
    idx: int
    heading: str
    elements: list[Element]

    # True for a scene with a reliable boundary.
    # False for a pseudo-scene created by the fallback parser.
    has_boundary: bool = True

    # Structured information from rule-based annotations.
    int_ext: str = ""
    locations: list[str] = field(default_factory=list)
    time_of_day: str = ""
    shot_type: str | None = None

    # Preserve any additional annotation information.
    annotation_data: dict[str, Any] = field(default_factory=dict)

    def lines(self) -> list[Line]:
        """Return readable lines in screenplay order."""

        out: list[Line] = []

        if self.heading:
            out.append(Line(HEADING, self.heading, None))

        out.extend(
            Line(e.kind, e.text, e.speaker)
            for e in self.elements
        )

        return out

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines())

    @property
    def location(self) -> tuple[str, str, str]:
        """Return structured location data, with heading fallback."""

        if self.int_ext or self.locations or self.time_of_day:
            return (
                self.int_ext,
                ", ".join(self.locations),
                self.time_of_day
            )

        return parse_heading(self.heading)

    @property
    def n_words(self) -> int:
        return sum(
            len(line.text.split())
            for line in self.lines()
        )


@dataclass
class Chunk:
    movie_id: str
    scene_idx: int
    part_idx: int
    n_parts: int

    # Original readable passage.
    text: str

    # Text selected for preprocessing and indexing.
    index_text: str

    # Distinct speakers associated with this chunk.
    speakers: list[str]

    n_words: int