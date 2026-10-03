"""One loader per source. Each returns records in the common schema (RECORD_COLUMNS).

A record is one row of one source: a Goodreads book, a Google Books edition, or a
Book-Crossing (Open Library) work. Matching and merging happen later, in match.py / merge.py.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import pandas as pd

from catalog.normalize import (
    BOOKCROSSING_BOOKS_PLACEHOLDERS,
    BOOKCROSSING_SUBJECTS_PLACEHOLDERS,
    GOODREADS_PLACEHOLDERS,
    clean_description,
    display_author,
    is_english,
    isbn_to_13,
    mask_imputed,
    replace_placeholders,
)

logger = logging.getLogger(__name__)

GOODREADS = "goodreads"
GOOGLE7K = "google7k"
BOOKCROSSING = "bookcrossing"

RECORD_COLUMNS: list[str] = [
    "source",       # str: GOODREADS | GOOGLE7K | BOOKCROSSING
    "source_id",    # str: id inside that source, unique per source
    "isbn13s",      # list[str]: checksum-valid ISBN-13s
    "title",        # str
    "subtitle",     # str | NaN
    "authors",      # list[str]: display names, main author first
    "description",  # str | NaN: cleaned, >= MIN_DESCRIPTION_LENGTH
    "tags",         # list[str]: genres / subjects / categories
    "language",     # "en" | NaN (non-English records are dropped by the loaders)
    "year",         # Int64 | NA: earliest publication year this source knows
    "popularity",   # Int64: ratings / interactions count in this source
    "avg_rating",   # float | NaN: on this source's own scale
]

# Open Library subjects that describe the edition or the catalogue, not the book's content.
# Starter list - extend it from the top-100 subject review.
BOOKCROSSING_NOISE_SUBJECTS: frozenset[str] = frozenset({
    "large type books", "media tie-in", "open library staff picks",
})

# Goodreads author roles that are not the book's author.
_GOODREADS_ROLE = re.compile(r"\((?!Goodreads Author\)).*?\)")


# ---------- helpers ----------

def _year(value: object) -> int | None:
    """First 4-digit year in a string: 'November 2nd 2011' -> 2011."""
    if value is None or pd.isna(value):
        return None
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value))
    return int(match.group(1)) if match else None


def _two_digit_year(value: object, latest: int | None) -> int | None:
    """'10/30/44' -> 1944 or 2044. Picks the century that is not later than `latest`.

    Ancient works stay wrong (The Odyssey '10/28/00' -> 2000); 2-digit years lost that
    information and no rule can recover it.
    """
    if value is None or pd.isna(value):
        return None
    match = re.fullmatch(r"\d{1,2}/\d{1,2}/(\d{2})", str(value).strip())
    if not match:
        return None
    yy = int(match.group(1))
    latest = latest or 2020
    return 2000 + yy if 2000 + yy <= latest else 1900 + yy


def _isbn_list(*values: object) -> list[str]:
    return list(dict.fromkeys(i for i in (isbn_to_13(v) for v in values) if i))


def _finalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Enforce schema, types and invariants shared by every loader."""
    df = df.assign(source=source)
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{source}: loader did not produce columns {sorted(missing)}")
    df = df[RECORD_COLUMNS].copy()
    df["source_id"] = df["source_id"].astype("str")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype("Int64")
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").astype("float")

    df = df[df["title"].notna()]
    if not df["source_id"].is_unique:
        raise ValueError(f"{source}: source_id is not unique")
    logger.info("%s: %d records", source, len(df))
    return df.reset_index(drop=True)


# ---------- Goodreads Best Books Ever ----------

def _goodreads_authors(value: object) -> list[str]:
    """'J.R.R. Tolkien, Christopher Tolkien (Editor)' -> ['J.R.R. Tolkien'].

    Drops editors, illustrators, translators...; keeps '(Goodreads Author)' entries.
    """
    if value is None or pd.isna(value):
        return []
    parts = [p.strip() for p in str(value).split(",")]
    authors = [p for p in parts if not _GOODREADS_ROLE.search(p)] or parts[:1]
    return [a for a in (display_author(p) for p in authors) if a]


def _parse_list(value: object) -> list[str]:
    """"['Fantasy', 'Fiction']" -> ['Fantasy', 'Fiction']; malformed -> []."""
    if value is None or pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []


def load_goodreads(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    raw = raw.drop_duplicates("bookId")  # 54 exact duplicate rows in the scrape
    raw = replace_placeholders(raw, GOODREADS_PLACEHOLDERS)
    raw = raw[is_english(raw["language"])]

    edition_year = raw["publishDate"].map(_year)
    first_year = raw["firstPublishDate"].map(_year)
    first_year = first_year.fillna(
        pd.Series([_two_digit_year(v, e) for v, e in zip(raw["firstPublishDate"], edition_year)], index=raw.index)
    )

    df = pd.DataFrame({
        "source_id": raw["bookId"],
        "isbn13s": raw["isbn"].map(_isbn_list),
        "title": raw["title"],
        "subtitle": pd.NA,
        "authors": raw["author"].map(_goodreads_authors),
        "description": raw["description"].map(clean_description),
        "tags": raw["genres"].map(_parse_list),
        "language": raw["language"].where(raw["language"].isna(), "en"),
        "year": pd.concat([first_year, edition_year], axis=1).min(axis=1),
        "popularity": raw["numRatings"],
        "avg_rating": raw["rating"],
    })
    return _finalise(df, GOODREADS)


# ---------- 7k books (Google Books) ----------

def load_google7k(path: Path) -> pd.DataFrame:
    raw = replace_placeholders(pd.read_csv(path, dtype=str))
    df = pd.DataFrame({
        "source_id": raw["isbn13"],
        "isbn13s": [_isbn_list(a, b) for a, b in zip(raw["isbn13"], raw["isbn10"])],
        "title": raw["title"],
        "subtitle": raw["subtitle"],
        "authors": raw["authors"].map(
            lambda s: [a for a in (display_author(p) for p in str(s).split(";")) if a] if pd.notna(s) else []
        ),
        "description": raw["description"].map(clean_description),
        "tags": raw["categories"].map(lambda c: [c.strip()] if pd.notna(c) else []),
        "language": pd.NA,  # dataset has no language column
        "year": raw["published_year"],
        "popularity": raw["ratings_count"],
        "avg_rating": raw["average_rating"],
    })
    return _finalise(df, GOOGLE7K)


# ---------- Book-Crossing, enriched with Open Library ----------

def _bookcrossing_authors(name: object) -> list[str]:
    """'Zullo, Allan; Rodell, Chris' -> ['Allan Zullo', 'Chris Rodell']."""
    if name is None or pd.isna(name):
        return []
    return [a for a in (display_author(p) for p in str(name).split(";")) if a]


def load_bookcrossing(directory: Path) -> pd.DataFrame:
    books = pd.read_csv(directory / "books.csv", dtype=str)
    books = replace_placeholders(books, BOOKCROSSING_BOOKS_PLACEHOLDERS)
    books = mask_imputed(books, "year", "filled_year")
    books = books[is_english(books["language"])]

    authors = pd.read_csv(directory / "authors.csv", dtype=str).set_index("author_id")["name"]

    subjects = pd.read_csv(directory / "books_to_subjects.csv", dtype=str)
    subjects = replace_placeholders(subjects, BOOKCROSSING_SUBJECTS_PLACEHOLDERS).dropna(subset=["subjects"])
    subjects = subjects[~subjects["subjects"].str.casefold().isin(BOOKCROSSING_NOISE_SUBJECTS)]
    tags = subjects.groupby("work_id")["subjects"].agg(list)

    interactions = pd.read_csv(directory / "interactions.csv", dtype={"work_id": str, "rating": float})
    stats = interactions.groupby("work_id").agg(popularity=("work_id", "size"), avg_rating=("rating", "mean"))

    df = pd.DataFrame({
        "source_id": books["work_id"],
        "isbn13s": books["isbn"].map(_isbn_list),
        "title": books["title"],
        "subtitle": pd.NA,
        "authors": books["author_id"].map(authors).map(_bookcrossing_authors),
        "description": books["description"].map(clean_description),
        "tags": books["work_id"].map(tags).map(lambda t: t if isinstance(t, list) else []),
        "language": books["language"].where(books["language"].isna(), "en"),
        "year": pd.to_numeric(books["year"], errors="coerce").where(lambda y: y > 0),
        "popularity": books["work_id"].map(stats["popularity"]),
        "avg_rating": books["work_id"].map(stats["avg_rating"]),
    })
    return _finalise(df, BOOKCROSSING)
