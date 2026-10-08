"""
Creating one unified dataset from two sources

Data Sources:
    1. Goodreads Best Books Dataset (https://github.com/scostap/goodreads_bbe_dataset/tree/main)
        - Title, series, author, rating, description, language, isbn, genres, edition, pages, numratings, awards,
        numRatings, bbeScore, bbeVotes
    2. Book Crossings dataset (https://www.kaggle.com/datasets/simonbouchardk/book-recommendation-platform-data)
        - Title, year, description, language, num_pages
        - books_enriched contains LLM generated genre
        - books_to_subjects contain topics from OpenLibrary metadata
        - interactions.csv contain user ratings (many NA values)

Both datasets are joined into one table books.csv. Interactions from book crossing data is used for CF.
Content-based recommender uses a subset of data that has descriptions, CF uses all data with interactions.
"""

from __future__ import annotations

import ast
import re
import kagglehub

from pathlib import Path
import pandas as pd

from catalog.normalize import is_english, clean_description, display_author, normalize_title, author_key

ROOT = Path(__file__).resolve().parents[2]
GOODREADS_PATH = ROOT / "data" / "books.csv"
BOOKCROSSING_HANDLE = "simonbouchardk/book-recommendation-platform-data/versions/1"
BOOKS_OUT = ROOT / "data" / "books.parquet"


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
        "gr_id": df["bookId"],
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


def add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    title = df["title"].map(normalize_title)
    author = df["author"].map(author_key)
    return df.assign(key=title + "|" + author)


def deduplicate_by_key(df: pd.DataFrame, id_col: str, popularity_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Keep one row per given key (the most popular). Rows without a key are unchanged.

    Returns (deduplicated frame, aliases): aliases maps EVERY id to the id that was kept, so
    data attached to a dropped duplicate (e.g. Book-Crossing interactions) can be moved over.
    """
    with_key = df[df["key"].notna()].sort_values(popularity_col, ascending=False)
    kept_id = with_key.groupby("key")[id_col].transform("first")
    aliases = pd.Series(kept_id.to_numpy(), index=with_key[id_col].to_numpy(), name="kept_id")
    deduplicated = pd.concat([with_key.drop_duplicates("key"), df[df["key"].isna()]])
    return deduplicated.sort_index(), aliases


def combine_tags(first: object, second: object) -> list[str]:
    seen_tags = set()
    result = []
    for tag in (first if isinstance(first, list) else []) + (second if isinstance(second, list) else []):
        norm = str(tag).strip().casefold()
        if norm and norm not in seen_tags:
            result.append(norm)
            seen_tags.add(norm)
    return result


def merge_sources(gr: pd.DataFrame, bx: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per book. Where match is found, books are combined into one, otherwise
    each is kept as is.
    """
    gr = gr.assign(key=gr["key"].fillna("gr-nokey:" + gr["gr_id"]))
    bx = bx.assign(key=bx["key"].fillna("bx-nokey:" + bx["bx_id"]))

    m = gr.merge(bx, on="key", how="outer", suffixes=("_gr", "_bx"))

    books = pd.DataFrame({
        "book_id": m["bx_id"].fillna("gr:" + m["gr_id"]),  # book-crossing when present to later join interactions
        "gr_id": m["gr_id"],
        "bx_id": m["bx_id"],
        "title": m["title_gr"].fillna(m["title_bx"]),
        "author": m["author_gr"].fillna(m["author_bx"]),
        "description": m["description_gr"].fillna(m["description_bx"]),
        "description_source": m["description_gr"].notna().map({True: "goodreads", False: None})\
            .fillna(m["description_bx"].notna().map({True: "bookcrossing"})),
        "tags": [combine_tags(a, b) for a, b in zip(m["tags_gr"], m["tags_bx"])],
        "gr_num_ratings": m["gr_num_ratings"].fillna(0).astype(int),
        "bx_readers": m["bx_readers"].fillna(0).astype(int),
    })
    books["has_description"] = books["description"].notna()

    if not books["book_id"].is_unique:
        raise ValueError("Book ID is not unique after merge")

    return books.set_index("book_id")


def build_books_dataset(goodreads_path: Path, bookcrossing_dir: Path, out_path: Path) -> pd.DataFrame:
    gr = add_match_key(load_goodreads_dataset(goodreads_path))
    bx = add_match_key(load_bookcrossing_dataset(bookcrossing_dir))

    gr, _ = deduplicate_by_key(gr, id_col="gr_id", popularity_col="gr_num_ratings")
    bx, bx_aliases = deduplicate_by_key(bx, id_col="bx_id", popularity_col="bx_readers")  # save also aliases here

    books = merge_sources(gr, bx)

    matched = books["gr_id"].notna() & books["bx_id"].notna()
    print(f"books: {len(books):,}  (goodreads {books['gr_id'].notna().sum():,}, "
          f"bookcrossing {books['bx_id'].notna().sum():,}, matched {matched.sum():,})")
    print(f"with description: {books['has_description'].sum():,}")
    print(books["description_source"].value_counts(dropna=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    books.to_parquet(out_path)
    bx_aliases.to_frame().to_parquet(out_path.with_name("bx_aliases.parquet"))
    return books


def main() -> None:
    bookcrossing_dir = Path(kagglehub.dataset_download(BOOKCROSSING_HANDLE))
    build_books_dataset(GOODREADS_PATH, bookcrossing_dir, BOOKS_OUT)


if __name__ == "__main__":
    main()