

from __future__ import annotations

import math

from collections import Counter
from typing import Iterable, Mapping


class WeightingModel:
    """Calculate TF-IDF weights and cosine similarity."""

    def __init__(
        self,
        index,
        tf_mode: str = "raw",
    ) -> None:

        if tf_mode not in {"raw", "log"}:
            raise ValueError(
                "tf_mode must be either 'raw' or 'log'."
            )

        self.index = index
        self.tf_mode = tf_mode

        self.n_documents = index.stats()["N_documents"]

        # Term -> IDF weight.
        self.idf_values: dict[str, float] = {}

        # Document ID -> normalized sparse TF-IDF vector.
        self.document_vectors: list[dict[str, float]] = [
            {} for _ in range(self.n_documents)
        ]

        # Original, non-normalized document vector lengths.
        self.document_norms: list[float] = [
            0.0 for _ in range(self.n_documents)
        ]

        self._build()

   
    # TERM FREQUENCY
   

    def _tf_weight(self, tf: int) -> float:
        """Calculate the selected TF weighting."""

        if tf <= 0:
            return 0.0

        if self.tf_mode == "raw":
            return float(tf)

        # Logarithmic TF:
        
        return 1.0 + math.log(tf)

    
    # INVERSE DOCUMENT FREQUENCY
    

    def _calculate_idf(self, df: int) -> float:
        """Calculate smoothed IDF."""

        if self.n_documents == 0:
            return 0.0

        return math.log(
            (self.n_documents + 1) / (df + 1)
        ) + 1.0

   
    # BUILD DOCUMENT VECTORS
    

    def _build(self) -> None:
        """Build normalized sparse TF-IDF vectors."""

        squared_norms = [
            0.0 for _ in range(self.n_documents)
        ]

        # Iterate over every term in the vocabulary.
        for term in self.index.postings:

            df = self.index.df(term)

            idf = self._calculate_idf(df)

            self.idf_values[term] = idf

            # Retrieve all documents containing this term.
            for doc_id, tf in self.index.iter_postings(term):

                if not 0 <= doc_id < self.n_documents:
                    raise ValueError(
                        f"Invalid document ID: {doc_id}"
                    )

                tf_weight = self._tf_weight(tf)

                weight = tf_weight * idf

                self.document_vectors[doc_id][term] = weight

                squared_norms[doc_id] += weight ** 2

        # Calculate the original vector lengths.
        for doc_id, squared_norm in enumerate(squared_norms):

            norm = math.sqrt(squared_norm)

            self.document_norms[doc_id] = norm

            # Normalize every document vector.
            if norm > 0:

                self.document_vectors[doc_id] = {
                    term: weight / norm
                    for term, weight
                    in self.document_vectors[doc_id].items()
                }

   
    # QUERY VECTORIZATION
    

    def transform_query(
        self,
        tokens: Iterable[str],
    ) -> dict[str, float]:
        """
        Convert preprocessed query tokens into a
        normalized TF-IDF vector.

        Unknown terms are ignored.
        """

        frequencies = Counter(tokens)

        query_vector: dict[str, float] = {}

        for term, tf in frequencies.items():

            idf = self.idf_values.get(term)

            # Ignore terms not found in the collection.
            if idf is None:
                continue

            weight = self._tf_weight(tf) * idf

            if weight > 0:
                query_vector[term] = weight

        # Normalize the query vector.
        norm = math.sqrt(
            sum(weight ** 2 for weight in query_vector.values())
        )

        if norm == 0:
            return {}

        return {
            term: weight / norm
            for term, weight in query_vector.items()
        }

    
    # COSINE SIMILARITY
    

    def cosine_similarity(
        self,
        query_vector: Mapping[str, float],
        doc_id: int,
    ) -> float:
        """
        Calculate cosine similarity between a normalized
        query vector and a normalized document vector.
        """

        if not 0 <= doc_id < self.n_documents:
            raise IndexError(
                f"Document ID {doc_id} does not exist."
            )

        if not query_vector:
            return 0.0

        document_vector = self.document_vectors[doc_id]

        score = sum(
            query_weight * document_vector.get(term, 0.0)
            for term, query_weight in query_vector.items()
        )

        return max(0.0, min(1.0, score))

   
    # DOCUMENT SCORING
   

    def score(
        self,
        query_tokens: Iterable[str],
        top_k: int | None = 10,
        candidate_doc_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rank documents according to cosine similarity.

        Args:
            query_tokens:
                Preprocessed query tokens.

            top_k:
                Maximum number of results.
                None returns all matching documents.

            candidate_doc_ids:
                Optional collection of document IDs to score.
                If omitted, candidates are found through postings.

        Returns:
            List of (doc_id, similarity_score), sorted
            from highest to lowest similarity.
        """

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None.")

        query_vector = self.transform_query(query_tokens)

        if not query_vector:
            return []

        # Find candidate documents using the inverted index.
        if candidate_doc_ids is None:

            candidates: set[int] = set()

            for term in query_vector:

                for doc_id, _ in self.index.iter_postings(term):
                    candidates.add(doc_id)

        else:

            candidates = set(candidate_doc_ids)

            invalid_ids = [
                doc_id
                for doc_id in candidates
                if not isinstance(doc_id, int)
                or not 0 <= doc_id < self.n_documents
            ]

            if invalid_ids:
                raise ValueError(
                    f"Invalid candidate document IDs: {invalid_ids}"
                )

        # Calculate cosine similarity for candidates.
        results = []

        for doc_id in candidates:

            similarity = self.cosine_similarity(
                query_vector,
                doc_id,
            )

            if similarity > 0:
                results.append((doc_id, similarity))

        # Highest similarity first.
        results.sort(
            key=lambda item: (-item[1], item[0])
        )

        if top_k is not None:
            results = results[:top_k]

        return results

    
    # TERM INFORMATION
   

    def explain_term(self, term: str) -> dict:
        """Return the weighting information for one term."""

        return {
            "term": term,
            "df": self.index.df(term),
            "cf": self.index.cf(term),
            "idf": self.idf_values.get(term, 0.0),
        }

    
    # STATISTICS
    

    def stats(self) -> dict:
        """Return weighting model statistics."""

        total_nonzero_weights = sum(
            len(vector)
            for vector in self.document_vectors
        )

        return {
            "N_documents": self.n_documents,
            "vocabulary_size": len(self.idf_values),
            "nonzero_weights": total_nonzero_weights,
            "tf_mode": self.tf_mode,
            "normalized_documents": True,
        }



"""TF-IDF weighting and vector preparation for CineSearch.

This module:
    1. Calculates IDF for every term in the inverted index.
    2. Builds normalized sparse TF-IDF document vectors.
    3. Converts user queries into TF-IDF vectors.
    4. Calculates cosine similarity and ranks candidate documents.

Usage:
    from irsearch.pipeline import load
    from irsearch.weighting import WeightingModel

    index, store, preprocessor, cfg = load("out")

    weighting = WeightingModel(index)

    query_tokens = preprocessor.process("vendetta revolution")
    results = weighting.score(query_tokens, top_k=5)

    store.close()
"""