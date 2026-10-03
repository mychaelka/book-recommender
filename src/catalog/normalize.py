"""Value-level normalisation shared by all source loaders.

  - placeholders ("No description available", "[NO_SUBJECT]", year 0 ...) -> real missing values
  - an English-only filter that understands each source's language labels ("eng", "English")
  - matching keys: ISBN-13, normalised title, author key
  - description quality

"Missing" means `pd.isna(value)` is True. In pandas 3 a `str` column stores missing values
as NaN, so we never try to force Python `None` into a Series.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


# ---------- placeholders ----------

# Values that mean "missing" in any text column. Compared after strip + casefold.
COMMON_PLACEHOLDERS: frozenset[str] = frozenset({"", "nan", "none", "null", "n/a", "[]"})

# Column-specific placeholders. Kept per column on purpose: "unknown" is a placeholder for
# language, but a global rule would also wipe a book actually titled "Unknown".
# One dict per table: a missing column then raises instead of being silently skipped.
BOOKCROSSING_BOOKS_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "description": frozenset({"no description available"}),
    "language": frozenset({"unknown"}),
}
BOOKCROSSING_SUBJECTS_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "subjects": frozenset({"[no_subject]"}),
}
GOODREADS_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "isbn": frozenset({"9999999999999"}),
}


def _canonical(series: pd.Series) -> pd.Series:
    """Strip + casefold, so placeholder comparison ignores case and padding."""
    return series.astype("str").str.strip().str.casefold()


def replace_placeholders(
    df: pd.DataFrame,
    placeholders: Mapping[str, Iterable[str]] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` where placeholder strings are replaced by missing values.

    Args:
        df: Source frame. Not modified.
        placeholders: Extra placeholders per column, on top of COMMON_PLACEHOLDERS.
        columns: Text columns to clean. Defaults to every column in `placeholders`
            plus all string/object columns of `df`.

    Raises:
        KeyError: A column in `placeholders` or `columns` does not exist in `df`.
    """
    placeholders = {col: frozenset(v.casefold() for v in vals) for col, vals in (placeholders or {}).items()}
    if columns is None:
        text_cols = df.select_dtypes(include=["object", "string", "str"]).columns
        columns = list(dict.fromkeys([*placeholders, *text_cols]))
    columns = list(columns)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not in frame: {missing}")

    out = df.copy()
    for col in columns:
        tokens = COMMON_PLACEHOLDERS | placeholders.get(col, frozenset())
        is_placeholder = _canonical(out[col]).isin(tokens) & out[col].notna()
        n = int(is_placeholder.sum())
        if n:
            logger.info("%s: %d placeholder values -> missing", col, n)
            out[col] = out[col].mask(is_placeholder)
    return out


def mask_imputed(df: pd.DataFrame, value_col: str, flag_col: str) -> pd.DataFrame:
    """Undo upstream imputation: set `value_col` to missing where `flag_col` is true.

    Book-Crossing (enriched) filled gaps with constants and flagged them, e.g.
    `filled_year=True` -> year 0, `filled_num_pages=True` -> 251 pages. A constant is not
    information, so we turn it back into a missing value. The flag may be bool or "True"/"False".
    """
    for col in (value_col, flag_col):
        if col not in df.columns:
            raise KeyError(f"Column not in frame: {col}")

    flag = df[flag_col]
    if flag.dtype != bool:
        flag = _canonical(flag).map({"true": True, "false": False})
        if flag.isna().any():
            raise ValueError(f"{flag_col} has values other than True/False")
        flag = flag.astype(bool)

    out = df.copy()
    out[value_col] = out[value_col].mask(flag)
    logger.info("%s: %d imputed values (per %s) -> missing", value_col, int(flag.sum()), flag_col)
    return out


# ---------- language ----------

# Labels the sources use for English: Book-Crossing "eng", Goodreads "English".
ENGLISH_LABELS: frozenset[str] = frozenset({"en", "eng", "english"})

# Labels that mean "language not known" rather than "not English".
UNKNOWN_LANGUAGE_LABELS: frozenset[str] = frozenset({"unknown", "und", "undetermined"})


def is_english(language: pd.Series, keep_missing: bool = True) -> pd.Series:
    """Boolean mask: True for English books.

    Args:
        language: Raw language labels, any source format.
        keep_missing: Treat missing / unknown language as English. Default True because
            those books are mostly English (e.g. "Moby Dick", "Mere Christianity");
            dropping them would remove popular titles.

    Usage:
        books = books[is_english(books["language"])]
    """
    labels = _canonical(language)
    english = labels.isin(ENGLISH_LABELS)
    unknown = language.isna() | labels.isin(UNKNOWN_LANGUAGE_LABELS | COMMON_PLACEHOLDERS)
    mask = english | (unknown & keep_missing)
    logger.info(
        "language: %d English, %d unknown (%s), %d dropped",
        int(english.sum()), int(unknown.sum()), "kept" if keep_missing else "dropped", int((~mask).sum()),
    )
    return mask


# ---------- ISBN ----------

def _isbn13_check_digit(first12: str) -> str:
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(first12))
    return str((10 - total % 10) % 10)


def _isbn10_is_valid(isbn10: str) -> bool:
    digits = [10 if c == "X" else int(c) for c in isbn10]
    return sum(d * (10 - i) for i, d in enumerate(digits)) % 11 == 0


def isbn_to_13(value: object) -> str | None:
    """Return a checksum-valid ISBN-13, or None.

    Accepts ISBN-10, ISBN-13, hyphens/spaces, and 9-digit ISBN-10s whose leading zero was
    lost by a numeric CSV parse (Book-Crossing has ~80 of these). Invalid checksums -> None,
    so typos never become matching keys.
    """
    if value is None or pd.isna(value):
        return None
    s = re.sub(r"[^0-9X]", "", str(value).upper())
    if len(s) == 9:
        s = "0" + s
    if len(s) == 10 and re.fullmatch(r"\d{9}[\dX]", s) and _isbn10_is_valid(s):
        core = "978" + s[:9]
        return core + _isbn13_check_digit(core)
    if len(s) == 13 and s.isdigit() and s[:3] in {"978", "979"} and _isbn13_check_digit(s[:12]) == s[12]:
        return s
    return None


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
    """Strip HTML and whitespace; return None if what is left is too short to be useful.

    Separate from placeholder handling on purpose: a 20-character blurb is real data of low
    quality, not a missing value, and the threshold is a tunable modelling choice.
    """
    if text is None or pd.isna(text):
        return None
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) >= min_length else None
