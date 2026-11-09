import pytest

from irsearch.preprocess import Preprocessor
from irsearch.config import PreprocessConfig


@pytest.fixture
def prep():
    return Preprocessor(PreprocessConfig(stem=False))


def test_lowercase(prep):
    assert prep.process("HELLO WORLD") == ["hello", "world"]


def test_stopwords(prep):
    assert prep.process("the detective is here") == ["detective"]


def test_negations_preserved(prep):
    assert prep.process("no not nor") == ["no", "not", "nor"]


def test_contractions(prep):
    assert prep.process("don't kill John's friend") == [
        "not", "kill", "john", "friend"
    ]


def test_screenplay_noise(prep):
    assert prep.process("JOHN (V.O.) CONT'D INT. HOUSE") == [
        "john", "house"
    ]


def test_acronyms(prep):
    assert prep.process("He lives in the U.S.A.") == ["lives", "usa"]


def test_accents(prep):
    assert prep.process("café") == ["cafe"]


def test_numbers():
    prep = Preprocessor(
        PreprocessConfig(stem=False, drop_numbers=True)
    )
    assert prep.process("room 123") == ["room"]


def test_stemming():
    prep = Preprocessor(PreprocessConfig(stem=True))
    assert prep.process("killing killed kills") == ["kill", "kill", "kill"]


def test_empty_input(prep):
    assert prep.process("") == []
    assert prep.process(None) == []