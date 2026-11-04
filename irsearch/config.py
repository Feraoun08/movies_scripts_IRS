
"""Central configuration for CineSearch.

Configuration is divided into:
- Parser: screenplay source and parsing behavior.
- Chunking: size and overlap of searchable passages.
- Preprocessing: tokenization-related decisions.
- Indexing: inverted-index options.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal



SpeakerMode = Literal["separate", "unit_once", "keep"]




SourcePreference = Literal["auto", "json", "txt"]


@dataclass
class ParserConfig:
   
    source_preference: SourcePreference = "auto"

    
    fallback_to_txt: bool = True

   
    min_scene_headings: int = 3
    max_cue_words: int = 5
    max_cue_chars: int = 40
    drop_page_artifacts: bool = True

   
    preserve_annotation_data: bool = True


@dataclass
class ChunkConfig:
    
    min_words: int = 250
    max_words: int = 350

    
    overlap_words: int = 50

    
    single_chunk_max_words: int | None = None

    
    snap_window_words: int = 40

    def __post_init__(self) -> None:
        if self.min_words <= 0:
            raise ValueError("min_words must be greater than zero")

        if self.max_words < self.min_words:
            raise ValueError("max_words must be >= min_words")

        if not 0 <= self.overlap_words < self.min_words:
            raise ValueError(
                "overlap_words must be >= 0 and < min_words"
            )

        if self.snap_window_words < 0:
            raise ValueError("snap_window_words cannot be negative")

        if self.single_chunk_max_words is None:
            self.single_chunk_max_words = max(
                self.max_words,
                2 * self.min_words - self.overlap_words
            )

        if self.single_chunk_max_words < self.min_words:
            raise ValueError(
                "single_chunk_max_words must be >= min_words"
            )


@dataclass
class PreprocessConfig:
    # Ignore tokens shorter than this length.
    min_token_len: int = 2

    # Whether purely numeric tokens should be removed.
    drop_numbers: bool = False

    # Apply Porter stemming.
    stem: bool = True

    # Additional words to REMOVE from the vocabulary.
    extra_stopwords: tuple[str, ...] = ()

    # Default stopwords to KEEP.
    preserve_stopwords: tuple[str, ...] = ()


@dataclass
class IndexConfig:
  
    store_positions: bool = False


@dataclass
class Config:
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    speaker_mode: SpeakerMode = "separate"

    def to_dict(self) -> dict:
        """Convert configuration into a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct configuration from a dictionary."""

        parser = dict(d.get("parser", {}))
        chunk = dict(d.get("chunk", {}))
        preprocess = dict(d.get("preprocess", {}))
        index = dict(d.get("index", {}))

        # Convert lists back to tuples.
        if "extra_stopwords" in preprocess:
            preprocess["extra_stopwords"] = tuple(
                preprocess["extra_stopwords"]
            )

        # Support configurations saved with the previous field name.
        if "keep_stopwords" in preprocess:
            old_value = preprocess.pop("keep_stopwords")
            preprocess.setdefault(
                "preserve_stopwords", tuple(old_value)
            )

        if "preserve_stopwords" in preprocess:
            preprocess["preserve_stopwords"] = tuple(
                preprocess["preserve_stopwords"]
            )

        # Backward compatibility with the old top-level setting.
        if "store_positions" in d:
            index.setdefault("store_positions", d["store_positions"])

        return cls(
            parser=ParserConfig(**parser),
            chunk=ChunkConfig(**chunk),
            preprocess=PreprocessConfig(**preprocess),
            index=IndexConfig(**index),
            speaker_mode=d.get("speaker_mode", "separate"),
        )
