
import pytest

from irsearch.evaluation import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    mean_average_precision,
    evaluate_rankings,
)


def test_precision_at_k():
    retrieved = [1, 2, 3, 4]
    relevant = {2, 4}

    assert precision_at_k(retrieved, relevant, 4) == 0.5


def test_recall_at_k():
    retrieved = [1, 2, 3, 4]
    relevant = {2, 4, 5}

    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)


def test_precision_uses_k_as_denominator():
    retrieved = [2]
    relevant = {2, 3}

    assert precision_at_k(retrieved, relevant, 5) == 0.2


def test_f1_at_k():
    retrieved = [1, 2, 3]
    relevant = {2, 4}

    # Precision = 1/3, Recall = 1/2.
    assert f1_at_k(retrieved, relevant, 3) == pytest.approx(0.4)


def test_average_precision():
    retrieved = [1, 2, 3, 4]
    relevant = {2, 4}

    # Relevant results at ranks 2 and 4:
    # AP = ((1/2) + (2/4)) / 2 = 0.5.
    assert average_precision(retrieved, relevant) == 0.5


def test_average_precision_perfect_ranking():
    retrieved = [2, 4, 1, 3]
    relevant = {2, 4}

    assert average_precision(retrieved, relevant) == 1.0


def test_average_precision_wrong_ranking():
    retrieved = [1, 3, 2, 4]
    relevant = {2, 4}

    assert average_precision(retrieved, relevant) == pytest.approx(5 / 12)


def test_duplicate_results_are_not_counted_twice():
    retrieved = [2, 2, 3]
    relevant = {2, 3}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_empty_relevant_set():
    assert precision_at_k([1, 2], set(), 2) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert average_precision([1, 2], set()) == 0.0


def test_map():
    rankings = {
        "query one": [1, 2, 3],
        "query two": [4, 5, 6],
    }

    judgments = {
        "query one": {2},
        "query two": {5},
    }

    assert mean_average_precision(rankings, judgments) == 0.5


def test_evaluate_rankings():
    rankings = {
        "query one": [1, 2, 3],
        "query two": [4, 5, 6],
    }

    judgments = {
        "query one": {2},
        "query two": {5},
    }

    report = evaluate_rankings(rankings, judgments, k=3)

    assert report["evaluated_queries"] == 2
    assert report["precision@3"] == pytest.approx(1 / 3)
    assert report["recall@3"] == 1.0
    assert report["map@3"] == pytest.approx(0.5)


def test_empty_judgments_are_skipped():
    rankings = {
        "query one": [1, 2, 3],
        "query two": [4, 5, 6],
    }

    judgments = {
        "query one": {2},
        "query two": set(),
    }

    report = evaluate_rankings(rankings, judgments, k=3)

    assert report["evaluated_queries"] == 1
    assert report["map@3"] == pytest.approx(1 / 2)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, 0)

    with pytest.raises(ValueError):
        average_precision([1, 2], {1}, k=-1)
