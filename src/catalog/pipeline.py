"""
Creating one unified dataset from two sources

Data Sources:
    1. Goodreads Best Books Dataset (https://github.com/scostap/goodreads_bbe_dataset/tree/main)
        - Title, series, author, rating, description, language, isbn, genres, edition, pages, numratings, awards,
        numRatings, bbeScore, bbeVotes
    2. Book Crossings dataset (https://www.kaggle.com/datasets/simonbouchardk/book-recommendation-platform-data)
        - Title, year, description, language, num_pages
        - work_id but no ISBN
        - books_enriched contains LLM generated genre
        - books_to_subjects contain topics from OpenLibrary metadata
        - interactions.csv contain user ratings (many NA values)

Both datasets are joined into one table books.csv. Interactions from book crossing data is used for CF.
Content-based recommender uses a subset of data that has descriptions, CF uses all data with interactions.
"""

from __future__ import annotations

import ast
import re

from pathlib import Path
import pandas as pd

from catalog.normalize import is_english, clean_description, display_author

GOODREADS_PATH = "../../data/books.csv"
BOOKCROSSINGS_PATH = "/home/mysa/.cache/kagglehub/datasets/simonbouchardk/book-recommendation-platform-data/versions/1"

def select_first_author_goodreads(authors: object) -> str | None:
    """'J.R.R. Tolkien, Christopher Tolkien (Editor)' -> 'J.R.R. Tolkien'."""
    if pd.isna(authors):
        return None
    first = str(authors).split(",")[0]
    return re.sub(r"\s*\(.*?\)", "", first).strip() or None  # Removes (Goodreads Author) etc.


def select_first_author_bookcrossing(authors: object) -> str | None:
    """'Tolkien, J.R.R.' -> 'J.R.R. Tolkien'."""
    if pd.isna(authors):
        return None
    return display_author(str(authors).split(";")[0])


def parse_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def load_goodreads_dataset(path: Path) -> pd.DataFrame:
    """
    Return cleaned goodreads dataset; remove duplicates, keep only english books, only the first author
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df = df.drop_duplicates("bookId", keep="first")
    df = df[is_english(df["language"])]

    return pd.DataFrame({
        "bookId": df["bookId"],
        "title": df["title"],
        "author": df["author"].map(select_first_author_goodreads),
        "description": df["description"].map(clean_description),
        "tags": df["genres"].map(parse_list),
        "gr_num_ratings": pd.to_numeric(df["numRatings"], errors="coerce").fillna(0).astype(int),
    }).reset_index(drop=True)


def load_bookcrossing_dataset(directory: Path) -> pd.DataFrame:
    """
    Book-Crossing columns: id, title, author, description, tags, popularity.
    """
    df = pd.read_csv(directory / "books.csv", dtype=str, keep_default_na=False, na_values=[""])
    df = df[is_english(df["language"])]
    df["description"] = df["description"].replace("No description available", pd.NA)
    authors = pd.read_csv(directory / "authors.csv", dtype=str).set_index("author_id")["name"]

    subjects = pd.read_csv(directory / "books_to_subjects.csv", dtype=str)
    subjects = subjects[subjects["subjects"] != "[NO_SUBJECT]"]
    tags = subjects.groupby("work_id")["subjects"].agg(list)

    readers = (pd.read_csv(directory / "interactions.csv", dtype=str, usecols=["user_id", "work_id"])
               .drop_duplicates()
               .groupby("work_id").size())

    return pd.DataFrame({
        "bx_id": df["work_id"],
        "title": df["title"],
        "author": df["author_id"].map(authors).map(select_first_author_bookcrossing),
        "description": df["description"].map(clean_description),
        "tags": df["work_id"].map(tags).map(lambda x: x if isinstance(x, list) else []),
        "bx_readers": df["work_id"].map(readers).fillna(0).astype(int),  # alternative to num_ratings
    }).reset_index(drop=True)
