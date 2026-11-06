"""Preprocessing chain, applied identically to documents and to queries:

    noise cleanup -> lowercase -> tokenization -> contraction handling
                  -> stop word removal -> Porter stemming

* Stoplist: deliberately conservative (function words only). Negations (no, not, nor) and
  direction/particle words (up, down, out, off) are KEPT.
* Script noise: INT, EXT, CONT'D, V.O., O.S., O.C., CONTINUED are removed.
* Contractions: "I'll" -> i (stop word), "John's" -> john, "don't" -> do + not.
* Stemming: Porter (original algorithm, as in the course); stems are cached.
"""
from __future__ import annotations

import re

from .config import PreprocessConfig

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself me
    more most my myself of on once only or other our ours ourselves own same she should so some
    such than that the their theirs them themselves then there these they this those through to
    too under until very was we were what when where which while who whom why will with would
    you your yours yourself yourselves
    """.split()
)

# Words that are formatting noise in screenplays, not content.
SCRIPT_NOISE: frozenset[str] = frozenset(
    {"int", "ext", "contd", "cont", "continued", "vo", "os", "oc"}
)

_NOISE_MARKERS = re.compile(
    r"(?<![a-z0-9])(?:v\.o\.|o\.s\.|o\.c\.|cont['\u2019]d\.?|contd)(?![a-z0-9])"
)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_SUFFIXES = ("'s", "'re", "'ve", "'ll", "'d", "'m")
_SPECIAL = {
    "can't": ["can", "not"],
    "won't": ["will", "not"],
    "shan't": ["shall", "not"],
    "ain't": ["not"],
}


def _expand(tok: str) -> list[str]:
    """Split a token with an apostrophe into its content pieces."""
    if "'" not in tok:
        return [tok]
    if tok in _SPECIAL:
        return _SPECIAL[tok]
    if tok.endswith("n't"):
        return [tok[:-3], "not"]
    for suf in _SUFFIXES:
        if tok.endswith(suf):
            return [tok[: -len(suf)]]
    return [tok.replace("'", "")]


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None) -> None:
        self.cfg = cfg or PreprocessConfig()
        self.stopwords = (STOPWORDS - set(self.cfg.keep_stopwords)) | set(self.cfg.extra_stopwords)
        self._stemmer = None
        self._cache: dict[str, str] = {}
        if self.cfg.stem:
            try:
                from nltk.stem import PorterStemmer
            except ImportError as exc:  # pragma: no cover
                raise ImportError("nltk is required for Porter stemming: pip install nltk") from exc
            self._stemmer = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)

    def stem(self, term: str) -> str:
        if self._stemmer is None:
            return term
        s = self._cache.get(term)
        if s is None:
            s = self._cache[term] = self._stemmer.stem(term)
        return s

    def process(self, text: str) -> list[str]:
        """Text -> list of index terms (same function for documents and queries)."""
        text = text.lower().replace("\u2019", "'")
        text = _NOISE_MARKERS.sub(" ", text)
        out: list[str] = []
        cfg = self.cfg
        for tok in _TOKEN.findall(text):
            for piece in _expand(tok):
                if not piece or len(piece) < cfg.min_token_len:
                    continue
                if cfg.drop_numbers and piece.isdigit():
                    continue
                if piece in self.stopwords or piece in SCRIPT_NOISE:
                    continue
                out.append(self.stem(piece))
        return out

    # a query goes through exactly the same chain as a document
    process_query = process
