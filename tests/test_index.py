
import pytest

from pathlib import Path

from irsearch.index import InvertedIndex
from irsearch.config import Config, IndexConfig


# FIXTURES


@pytest.fixture
def config():
    return Config(
        index=IndexConfig(store_positions=False)
    )


@pytest.fixture
def index(config):
    idx = InvertedIndex(config)

    idx.add_document(
        0,
        ["detective", "find", "body", "detective"],
        movie_id="movie_1",
        title="The Mystery",
        scene_idx=0,
        part_idx=0,
    )

    idx.add_document(
        1,
        ["killer", "hide", "weapon"],
        movie_id="movie_1",
        title="The Mystery",
        scene_idx=1,
        part_idx=0,
    )

    idx.add_document(
        2,
        ["detective", "investigate", "murder"],
        movie_id="movie_2",
        title="Dark Night",
        scene_idx=0,
        part_idx=0,
    )

    return idx


# 1. DOCUMENT INSERTION


def test_add_documents(index):
    assert index.N == 3
    assert len(index.docs) == 3


def test_document_ids_must_be_consecutive(config):
    idx = InvertedIndex(config)

    with pytest.raises(ValueError):
        idx.add_document(
            1,
            ["hello"],
            movie_id="movie_1",
            title="Test",
            scene_idx=0,
            part_idx=0,
        )


def test_empty_tokens_are_allowed(config):
    idx = InvertedIndex(config)

    idx.add_document(
        0,
        [],
        movie_id="movie_1",
        title="Test",
        scene_idx=0,
        part_idx=0,
    )

    assert idx.N == 1
    assert idx.doc_lengths[0] == 0
    assert idx.vocabulary_size == 0


def test_invalid_tokens_are_rejected(config):
    idx = InvertedIndex(config)

    with pytest.raises(ValueError):
        idx.add_document(
            0,
            ["hello", "", "world"],
            movie_id="movie_1",
            title="Test",
            scene_idx=0,
            part_idx=0,
        )


def test_cannot_add_after_finalize(index):
    index.finalize()

    with pytest.raises(RuntimeError):
        index.add_document(
            3,
            ["new"],
            movie_id="movie_3",
            title="New Movie",
            scene_idx=0,
            part_idx=0,
        )



# 2. TERM STATISTICS


def test_vocabulary_size(index):
    assert index.vocabulary_size == 8


def test_document_frequency(index):
    assert index.df("detective") == 2
    assert index.df("killer") == 1
    assert index.df("unknown") == 0


def test_collection_frequency(index):
    assert index.cf("detective") == 3
    assert index.cf("body") == 1
    assert index.cf("unknown") == 0


def test_average_document_length(index):
    # Document lengths: 4, 3, 3
    assert index.avg_doc_length == 10 / 3


def test_total_postings(index):
    assert index.n_postings() == 9


def test_total_terms(index):
    assert sum(index.doc_lengths) == 10


def test_dictionary(index):
    assert index.dictionary() == sorted(index.dictionary())
    assert "detective" in index.dictionary()



# 3. POSTINGS


def test_postings_are_sorted(index):
    results = list(index.iter_postings("detective"))

    assert results == [(0, 2), (2, 1)]


def test_single_term_posting(index):
    results = list(index.iter_postings("killer"))

    assert results == [(1, 1)]


def test_unknown_term_posting(index):
    results = list(index.iter_postings("unknown"))

    assert results == []



# 4. DOCUMENT INFORMATION


def test_doc_info(index):
    info = index.doc_info(1)

    assert info == {
        "doc_id": 1,
        "movie_id": "movie_1",
        "title": "The Mystery",
        "scene_idx": 1,
        "part_idx": 0,
        "length": 3,
    }


def test_invalid_doc_info(index):
    with pytest.raises(IndexError):
        index.doc_info(10)



# 5. COLLECTION STATISTICS


def test_stats(index):
    stats = index.stats()

    assert stats["N_documents"] == 3
    assert stats["vocabulary_size"] == 8
    assert stats["total_postings"] == 9
    assert stats["total_terms"] == 10
    assert stats["has_positions"] is False


def test_empty_index_stats(config):
    idx = InvertedIndex(config)

    stats = idx.stats()

    assert stats["N_documents"] == 0
    assert stats["vocabulary_size"] == 0
    assert stats["total_postings"] == 0
    assert stats["total_terms"] == 0
    assert stats["avg_doc_length"] == 0.0



# 6. POSITIONAL INDEX


@pytest.fixture
def positional_index():
    config = Config(
        index=IndexConfig(store_positions=True)
    )

    idx = InvertedIndex(config)

    idx.add_document(
        0,
        ["i", "will", "be", "back", "i", "will", "return"],
        movie_id="movie_1",
        title="Terminator",
        scene_idx=0,
        part_idx=0,
    )

    idx.add_document(
        1,
        ["he", "will", "return"],
        movie_id="movie_2",
        title="Another Movie",
        scene_idx=0,
        part_idx=0,
    )

    return idx


def test_positions_before_finalize(positional_index):
    with pytest.raises(RuntimeError):
        positional_index.get_positions("will", 0)


def test_positions_after_finalize(positional_index):
    positional_index.finalize()

    assert positional_index.get_positions("will", 0) == [1, 5]
    assert positional_index.get_positions("back", 0) == [3]
    assert positional_index.get_positions("return", 1) == [2]


def test_positions_for_missing_term(positional_index):
    positional_index.finalize()

    assert positional_index.get_positions("unknown", 0) == []


def test_positions_for_missing_document(positional_index):
    positional_index.finalize()

    assert positional_index.get_positions("will", 10) == []


def test_positions_disabled(config):
    idx = InvertedIndex(config)

    with pytest.raises(RuntimeError):
        idx.get_positions("hello", 0)


def test_finalize_is_safe_to_call_twice(positional_index):
    positional_index.finalize()
    first_offsets = {
        term: list(posting.offsets)
        for term, posting in positional_index.postings.items()
    }

    positional_index.finalize()

    second_offsets = {
        term: list(posting.offsets)
        for term, posting in positional_index.postings.items()
    }

    assert first_offsets == second_offsets



# 7. SAVE AND LOAD


def test_save_and_load(index, tmp_path):
    path = tmp_path / "out" / "index.pkl"

    index.save(path)

    assert path.exists()

    loaded = InvertedIndex.load(path)

    assert loaded.N == index.N
    assert loaded.vocabulary_size == index.vocabulary_size
    assert loaded.df("detective") == 2
    assert loaded.cf("detective") == 3
    assert list(loaded.iter_postings("detective")) == [
        (0, 2),
        (2, 1),
    ]


def test_save_and_load_with_positions(positional_index, tmp_path):
    path = tmp_path / "positional_index.pkl"

    positional_index.finalize()
    positional_index.save(path)

    loaded = InvertedIndex.load(path)

    assert loaded.store_positions is True
    assert loaded.get_positions("will", 0) == [1, 5]


def test_save_finalizes_index(index, tmp_path):
    path = tmp_path / "index.pkl"

    index.save(path)

    assert index._finalized is True

    loaded = InvertedIndex.load(path)

    assert loaded.N == 3
