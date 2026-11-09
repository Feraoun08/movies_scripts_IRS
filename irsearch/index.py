
from __future__ import annotations

import pickle
from array import array
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Iterator, NamedTuple

from .config import Config


_FORMAT_VERSION = 1


class DocEntry(NamedTuple):
    movie_id: str
    title: str
    scene_idx: int
    part_idx: int


@dataclass
class Postings:
    doc_ids: array
    tfs: array
    positions: array | None = None
    offsets: array | None = None

    def __len__(self) -> int:
        return len(self.doc_ids)


class InvertedIndex:

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()

        self.store_positions = config.index.store_positions

        self.postings: dict[str, Postings] = {}
        self.docs: list[DocEntry] = []
        self.doc_lengths = array("I")

        self._finalized = False

   
    # 1. BUILDING THE INDEX
   

    def add_document(
        self,
        doc_id: int,
        tokens: list[str],
        *,
        movie_id: str,
        title: str,
        scene_idx: int,
        part_idx: int,
    ) -> None:

        if self._finalized:
            raise RuntimeError(
                "Cannot add documents after index finalization."
            )

        if doc_id != len(self.docs):
            raise ValueError(
                f"Expected doc_id {len(self.docs)}, got {doc_id}."
            )

        if not all(isinstance(token, str) and token for token in tokens):
            raise ValueError("Tokens must be non-empty strings.")

        self.docs.append(
            DocEntry(movie_id, title, scene_idx, part_idx)
        )

        self.doc_lengths.append(len(tokens))

        if self.store_positions:

            positions_by_term: dict[str, list[int]] = {}

            for position, term in enumerate(tokens):
                positions_by_term.setdefault(term, []).append(position)

            for term, positions in positions_by_term.items():

                posting = self._get_or_create(term)

                posting.doc_ids.append(doc_id)
                posting.tfs.append(len(positions))
                posting.positions.extend(positions)

        else:

            frequencies = Counter(tokens)

            for term, tf in frequencies.items():

                posting = self._get_or_create(term)

                posting.doc_ids.append(doc_id)
                posting.tfs.append(tf)

    def _get_or_create(self, term: str) -> Postings:

        posting = self.postings.get(term)

        if posting is None:

            posting = Postings(
                doc_ids=array("I"),
                tfs=array("I"),
                positions=array("I") if self.store_positions else None,
            )

            self.postings[term] = posting

        return posting

    def finalize(self) -> InvertedIndex:

        if self._finalized:
            return self

        if self.store_positions:

            for posting in self.postings.values():

                posting.offsets = array(
                    "I",
                    [0, *accumulate(posting.tfs)]
                )

        self._finalized = True

        return self

    
    # 2. COLLECTION STATISTICS
   

    @property
    def N(self) -> int:
        return len(self.docs)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def avg_doc_length(self) -> float:

        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths) / len(self.doc_lengths)

    def df(self, term: str) -> int:

        posting = self.postings.get(term)

        return len(posting) if posting else 0

    def cf(self, term: str) -> int:

        posting = self.postings.get(term)

        return sum(posting.tfs) if posting else 0

    def dictionary(self) -> list[str]:
        return sorted(self.postings)

    def n_postings(self) -> int:

        return sum(
            len(posting)
            for posting in self.postings.values()
        )

    def stats(self) -> dict:

        return {
            "N_documents": self.N,
            "vocabulary_size": self.vocabulary_size,
            "total_postings": self.n_postings(),
            "total_terms": sum(self.doc_lengths),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "has_positions": self.store_positions,
        }

    # -----------------------------------------------------
    # 3. SEARCHING THE INDEX
    # -----------------------------------------------------

    def iter_postings(
        self,
        term: str
    ) -> Iterator[tuple[int, int]]:
        """Return (doc_id, term_frequency) pairs."""

        posting = self.postings.get(term)

        if posting is not None:

            yield from zip(
                posting.doc_ids,
                posting.tfs
            )

    def get_positions(
        self,
        term: str,
        doc_id: int
    ) -> list[int]:
        """Return all positions of a term inside a document."""

        if not self.store_positions:
            raise RuntimeError(
                "Index was built without positional information."
            )

        posting = self.postings.get(term)

        if posting is None:
            return []

        if posting.offsets is None:
            raise RuntimeError(
                "Index must be finalized before retrieving positions."
            )

        i = bisect_left(posting.doc_ids, doc_id)

        if (
            i == len(posting.doc_ids)
            or posting.doc_ids[i] != doc_id
        ):
            return []

        start = posting.offsets[i]
        end = posting.offsets[i + 1]

        return list(posting.positions[start:end])

    def doc_info(self, doc_id: int) -> dict:

        if not 0 <= doc_id < self.N:
            raise IndexError(f"Invalid doc_id: {doc_id}")

        doc = self.docs[doc_id]

        return {
            "doc_id": doc_id,
            "movie_id": doc.movie_id,
            "title": doc.title,
            "scene_idx": doc.scene_idx,
            "part_idx": doc.part_idx,
            "length": self.doc_lengths[doc_id],
        }

    # -----------------------------------------------------
    # 4. SAVING AND LOADING
    # -----------------------------------------------------

    def save(self, path: str | Path) -> None:

        if not self._finalized:
            self.finalize()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with path.open("wb") as file:

            pickle.dump(
                {
                    "version": _FORMAT_VERSION,
                    "index": self,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(path: str | Path) -> InvertedIndex:

        with Path(path).open("rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict):
            raise ValueError("Invalid index file.")

        if data.get("version") != _FORMAT_VERSION:
            raise ValueError("Unsupported index format version.")

        index = data.get("index")

        if not isinstance(index, InvertedIndex):
            raise ValueError("Invalid index object.")

        return index
