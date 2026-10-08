"""Small value-level helpers used by the pipeline.

  - is_english:        English-only filter that understands each source's labels ("eng", "English")
  - normalize_title:   title matching key
  - display_author:    "Tolkien, J.R.R." -> "J.R.R. Tolkien"
  - author_key:        author matching key ("tolkien_j")
  - clean_description: strip HTML, drop descriptions too short to be useful
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


# ---------- language ----------

# Labels the sources use for English: Book-Crossing "eng", Goodreads "English".
ENGLISH_LABELS: frozenset[str] = frozenset({"en", "eng", "english"})

# Labels that mean "language not known" rather than "not English".
UNKNOWN_LANGUAGE_LABELS: frozenset[str] = frozenset({"", "nan", "none", "unknown", "und", "undetermined"})


def is_english(language: pd.Series, keep_missing: bool = True) -> pd.Series:
    """Boolean mask: True for English books.

    Args:
        language: Raw language labels, any source format.
        keep_missing: Treat missing / unknown language as English. Default True because
            those books are mostly English (e.g. "Moby Dick", "Mere Christianity");
            dropping them would remove popular titles.
    """
    labels = language.astype("str").str.strip().str.casefold()
    english = labels.isin(ENGLISH_LABELS)
    unknown = language.isna() | labels.isin(UNKNOWN_LANGUAGE_LABELS)
    mask = english | (unknown & keep_missing)
    logger.info(
        "language: %d English, %d unknown (%s), %d dropped",
        int(english.sum()), int(unknown.sum()), "kept" if keep_missing else "dropped", int((~mask).sum()),
    )
    return mask


# ---------- title / author keys ----------

def _fold(text: str) -> str:
    """Casefold and strip accents: 'Café' -> 'cafe'. Non-Latin letters are kept."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


_LEADING_ARTICLE = re.compile(r"^(the|a|an) ")


def normalize_title(title: object) -> str | None:
    """Matching key for a title.

    'The Fellowship of the Ring (The Lord of the Rings, #1)' -> 'fellowship of the ring'
    'Harry Potter & the Sorcerer's Stone: Illustrated' -> 'harry potter and the sorcerers stone'
    """
    if title is None or pd.isna(title):
        return None
    t = _fold(str(title))
    t = re.sub(r"\s*\(.*?\)|\s*\[.*?\]", "", t)  # series info, edition notes
    t = t.split(":")[0]                            # subtitle
    t = t.replace("&", " and ")
    t = re.sub(r"[^\w ]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEADING_ARTICLE.sub("", t)
    return t or None


_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md"})


def display_author(name: object) -> str | None:
    """'Tolkien, J.R.R.' -> 'J.R.R. Tolkien'; strips '(Goodreads Author)' and extra spaces."""
    if name is None or pd.isna(name):
        return None
    n = re.sub(r"\s*\(.*?\)", "", str(name)).strip()
    if n.count(",") == 1:
        last, first = (part.strip() for part in n.split(","))
        n = f"{first} {last}" if first.casefold().strip(".") not in _NAME_SUFFIXES else f"{last} {first}"
    n = re.sub(r"\s+", " ", n).strip()
    return n or None


def author_key(name: object) -> str | None:
    """Matching key for one author: surname + first initial.

    'J.R.R. Tolkien', 'J. R. R. Tolkien', 'Tolkien, J.R.R.', 'John Ronald Reuel Tolkien'
    all -> 'tolkien_j'. Single-token names ('Delacorta') -> 'delacorta'.
    """
    n = display_author(name)
    if n is None:
        return None
    tokens = [tok for tok in re.findall(r"[^\W\d_]+", _fold(n)) if tok not in _NAME_SUFFIXES]
    if not tokens:
        return None
    return tokens[-1] if len(tokens) == 1 else f"{tokens[-1]}_{tokens[0][0]}"


# ---------- description quality ----------

MIN_DESCRIPTION_LENGTH = 50


def clean_description(text: object, min_length: int = MIN_DESCRIPTION_LENGTH) -> str | None:
    """Strip HTML and whitespace; return None if what is left is too short to be useful."""
    if text is None or pd.isna(text):
        return None
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= min_length else None
