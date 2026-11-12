
"""Evaluation metrics for the CineSearch information retrieval system."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _unique_retrieved(
    retrieved_ids: Sequence[int],
    k: int | None = None,
) -> list[int]:
    """Remove duplicate document IDs while preserving ranking order."""

    if k is not None:
        _validate_k(k)
        retrieved_ids = retrieved_ids[:k]

    unique = []
    seen = set()

    for doc_id in retrieved_ids:
        if doc_id not in seen:
            unique.append(doc_id)
            seen.add(doc_id)

    return unique


def precision_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Collection[int],
    k: int,
) -> float:
    """Calculate Precision@K."""

    _validate_k(k)

    retrieved = _unique_retrieved(retrieved_ids, k)
    relevant = set(relevant_ids)

    relevant_retrieved = sum(
        doc_id in relevant for doc_id in retrieved
    )

    # The denominator is always k, even if fewer than k
    # documents were retrieved.
    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Collection[int],
    k: int,
) -> float:
    """Calculate Recall@K."""

    _validate_k(k)

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = _unique_retrieved(retrieved_ids, k)

    relevant_retrieved = sum(
        doc_id in relevant for doc_id in retrieved
    )

    return relevant_retrieved / len(relevant)


def f1_at_k(
    retrieved_ids: Sequence[int],
    relevant_ids: Collection[int],
    k: int,
) -> float:
    """Calculate F1@K from Precision@K and Recall@K."""

    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average_precision(
    retrieved_ids: Sequence[int],
    relevant_ids: Collection[int],
    k: int | None = None,
) -> float:
    """
    Calculate Average Precision (AP).

    Each relevant document contributes the precision at the
    rank where it was retrieved.

    AP = sum(Precision@rank for relevant retrieved documents)
         / total number of relevant documents
    """

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = _unique_retrieved(retrieved_ids, k)

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean_average_precision(
    rankings: Mapping[str, Sequence[int]],
    relevance_judgments: Mapping[str, Collection[int]],
    k: int | None = None,
) -> float:
    """Calculate MAP across queries with at least one relevant document."""

    average_precisions = []

    for query, relevant_ids in relevance_judgments.items():

        if not relevant_ids:
            continue

        retrieved_ids = rankings.get(query, [])

        average_precisions.append(
            average_precision(
                retrieved_ids,
                relevant_ids,
                k=k,
            )
        )

    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)


def evaluate_rankings(
    rankings: Mapping[str, Sequence[int]],
    relevance_judgments: Mapping[str, Collection[int]],
    k: int = 10,
) -> dict[str, Any]:
    """
    Evaluate ranked results for multiple queries.

    Args:
        rankings:
            Query -> ordered list of retrieved chunk IDs.

        relevance_judgments:
            Query -> collection of manually judged relevant chunk IDs.

        k:
            Cutoff used for Precision, Recall, F1 and AP.

    Returns:
        Per-query metrics and macro-averaged metrics.
    """

    _validate_k(k)

    per_query = []

    for query, relevant_ids in relevance_judgments.items():

        relevant = set(relevant_ids)

        # Queries without relevance judgments cannot be evaluated.
        if not relevant:
            continue

        retrieved = rankings.get(query, [])

        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        f1 = f1_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant, k=k)

        per_query.append({
            "query": query,
            "relevant_documents": len(relevant),
            "retrieved_documents": len(retrieved),
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"f1@{k}": f1,
            f"ap@{k}": ap,
        })

    if not per_query:
        return {
            "k": k,
            "evaluated_queries": 0,
            "per_query": [],
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"f1@{k}": 0.0,
            f"map@{k}": 0.0,
        }

    count = len(per_query)

    def mean(metric: str) -> float:
        return sum(row[metric] for row in per_query) / count

    return {
        "k": k,
        "evaluated_queries": count,
        "per_query": per_query,
        f"precision@{k}": mean(f"precision@{k}"),
        f"recall@{k}": mean(f"recall@{k}"),
        f"f1@{k}": mean(f"f1@{k}"),
        f"map@{k}": mean(f"ap@{k}"),
    }
