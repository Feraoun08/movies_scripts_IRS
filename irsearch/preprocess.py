
"""

Processing stages:
    1. Cleaning:
        - Lowercase text.
        - Normalize accents.
        - Remove screenplay formatting noise.
        - Normalize acronyms such as U.S.A. -> usa.
        - Normalize thousands separators.

    2. Tokenization:
        - Extract words and numbers.
        - Expand contractions such as don't -> do not.
        - Normalize possessives such as John's -> john.

    3. Filtering:
        - Remove stopwords.
        - Remove screenplay noise words.
        - Remove tokens shorter than min_token_len.
        - Optionally remove numbers.

    4. Stemming:
        - Apply the original Porter stemming algorithm.
        - Cache previously stemmed terms.

The same process() method is used for both documents and queries.
"""

from __future__ import annotations

import re
import unicodedata

from .config import PreprocessConfig


#1-stop words 


STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself me
    more most my myself of on once only or other our ours ourselves own same she should so some
    such than that the their theirs them themselves then there these they this those through to
    too under until very was we were what when where which while who whom why will with would
    you your yours yourself yourselves ie eg etc pm
    """.split()
)


#2-screen specific noise 

SCRIPT_NOISE: frozenset[str] = frozenset(
    {
        "int",
        "ext",
        "contd",
        "cont",
        "continued",
        "vo",
        "os",
        "oc",
    }
)


#regular expressions

# Screenplay markers:
# V.O. -> removed
# O.S. -> removed
# O.C. -> removed
# CONT'D -> removed

_NOISE_MARKERS = re.compile(
    r"(?<![a-z0-9])"
    r"(?:v\.o\.|o\.s\.|o\.c\.|cont['\u2019]d\.?|contd)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)


# Acronyms:


_ACRONYM = re.compile(
    r"(?<![a-z0-9])(?:[a-z]\.){2,}",
    re.IGNORECASE,
)


# Thousands separators:
# 1,000 -> 1000

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


# Extract words and numbers, allowing internal apostrophes.

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


# Special character normalization.

_FOLD = str.maketrans(
    {
        "\u00df": "ss",  # ß -> ss
        "\u00e6": "ae",  # æ -> ae
        "\u0153": "oe",  # œ -> oe
        "\u00f8": "o",   # ø -> o
        "\u0142": "l",   # ł -> l
        "\u0111": "d",   # đ -> d
    }
)


#4 contractions

_SUFFIXES = ("'s", "'re", "'ve", "'ll", "'d", "'m")


_SPECIAL_CONTRACTIONS = {
    "can't": ["can", "not"],
    "won't": ["will", "not"],
    "shan't": ["shall", "not"],
    "ain't": ["not"],
}


def _expand(token: str) -> list[str]:
  

    if "'" not in token:
        return [token]

    if token in _SPECIAL_CONTRACTIONS:
        return _SPECIAL_CONTRACTIONS[token]

    if token.endswith("n't"):
        return [token[:-3], "not"]

    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            return [token[:-len(suffix)]]

    return [token.replace("'", "")]

#5-preproccessor

class Preprocessor:

    def __init__(self, cfg: PreprocessConfig | None = None) -> None:

        self.cfg = cfg or PreprocessConfig()

        # Normalize custom stopwords to lowercase.
        preserve = {
            word.lower()
            for word in self.cfg.preserve_stopwords
        }

        extra = {
            word.lower()
            for word in self.cfg.extra_stopwords
        }

        self.stopwords = (
            STOPWORDS - preserve
        ) | extra

        # Porter stemmer and cache.
        self._stemmer = None
        self._cache: dict[str, str] = {}

        if self.cfg.stem:
            try:
                from nltk.stem import PorterStemmer

            except ImportError as exc:
                raise ImportError(
                    "NLTK is required for Porter stemming. "
                    "Install it using: python -m pip install nltk"
                ) from exc

            self._stemmer = PorterStemmer(
                mode=PorterStemmer.ORIGINAL_ALGORITHM
            )

 #stage 1 : cleaning and tokenization

    def tokenize(self, text: str | bytes | None) -> list[str]:
        """
        Clean text and convert it into tokens.

        Returns:
            A list of lowercase normalized tokens.
        """

        if text is None:
            return []

        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode(
                "utf-8",
                errors="replace"
            )

        if not isinstance(text, str):
            raise TypeError(
                "Preprocessor expects a string, bytes, or None."
            )

        if not text:
            return []

        # Lowercase and normalize apostrophes.
        text = text.lower().replace("\u2019", "'")

        # Fold special characters.
        text = text.translate(_FOLD)

        # Normalize Unicode accents.
        text = unicodedata.normalize("NFKD", text)

        text = "".join(
            char
            for char in text
            if not unicodedata.combining(char)
        )

        # Remove screenplay formatting markers.
        text = _NOISE_MARKERS.sub(" ", text)

        # Normalize acronyms before token extraction.
        text = _ACRONYM.sub(
            lambda match: match.group(0).replace(".", ""),
            text,
        )

        # Remove thousands separators.
        text = _DIGIT_COMMA.sub("", text)

        # Extract tokens.
        raw_tokens = _TOKEN.findall(text)

        # Expand contractions.
        tokens = []

        for token in raw_tokens:
            tokens.extend(_expand(token))

        return tokens

   # stage 2 :filtring

    def filter(self, tokens: list[str]) -> list[str]:
  

        cfg = self.cfg

        filtered = []

        for token in tokens:

            if not token:
                continue

            # Preserve normalized acronyms such as USA and FBI,
            # even if they are shorter than min_token_len.
            is_acronym = token in {"usa", "uk", "fbi"}

            if (
                len(token) < cfg.min_token_len
                and not is_acronym
            ):
                continue

            if cfg.drop_numbers and token.isdigit():
                continue

            if token in self.stopwords:
                continue

            if token in SCRIPT_NOISE:
                continue

            filtered.append(token)

        return filtered

    # stage3: steeming

    def stem(self, term: str) -> str:
      

        if self._stemmer is None:
            return term

        # Preserve normalized acronyms and accent-folded
        # terms whose spelling should remain unchanged.
        if term in {"usa", "uk", "fbi", "cafe"}:
            return term

        cached = self._cache.get(term)

        if cached is not None:
            return cached

        result = self._stemmer.stem(term)

        self._cache[term] = result

        return result

    def stem_all(self, tokens: list[str]) -> list[str]:
       

        return [
            self.stem(token)
            for token in tokens
        ]

 #complete pipelind

    def process(self, text: str | bytes | None) -> list[str]:
      

        if text is None:
            return []

        tokens = self.tokenize(text)

        filtered = self.filter(tokens)

        return self.stem_all(filtered)

  #debugging

    def explain(
        self,
        text: str | bytes | None
    ) -> dict[str, list[str]]:
     

        tokens = self.tokenize(text)

        filtered = self.filter(tokens)

        stemmed = self.stem_all(filtered)

        return {
            "tokenize": tokens,
            "filter": filtered,
            "stem": stemmed,
        }

    # Queries and documents use exactly the same pipeline.
    process_query = process
