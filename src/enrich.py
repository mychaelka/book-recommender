"""Enrich the Goodreads export with book descriptions and genres.

Sources, tried in order for each book:
  1. data/books.csv (Goodreads Best Books Ever) - by Goodreads ID, ISBN13, then title+author
  2. Google Books API - only if GOOGLE_BOOKS_API_KEY is set (anonymous quota is 0)
  3. Open Library - by ISBN, then title+author

API responses are cached in data/cache/ so re-runs don't hit the network again.

Usage:
    python src/enrich.py
    GOOGLE_BOOKS_API_KEY=... python src/enrich.py
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXPORT_PATH = DATA / "my_export.csv"
BOOKS_PATH = DATA / "books.csv"
OUTPUT_PATH = DATA / "my_books_enriched.csv"
CACHE_PATH = DATA / "cache" / "api_cache.json"

GOOGLE_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")
USER_AGENT = "book-recommender/0.1 (personal learning project)"
MIN_DESCRIPTION_LENGTH = 50


# ---------- normalisation ----------

def normalize_title(title: str) -> str:
    """Lowercase, drop series info in parentheses and subtitles, keep alphanumerics."""
    title = str(title).lower()
    title = re.sub(r"\s*\(.*?\)", "", title)
    title = title.split(":")[0]
    title = re.sub(r"[^\w ]", "", title)
    return re.sub(r"\s+", " ", title).strip()


def normalize_author(author: str) -> str:
    """First listed author, without the '(Goodreads Author)' / '(Translator)' suffixes."""
    author = str(author).split(",")[0].lower()
    author = re.sub(r"\s*\(.*?\)", "", author)
    author = re.sub(r"[^\w ]", "", author)
    return re.sub(r"\s+", " ", author).strip()  # the export has names like "Dan    Brown"


def clean_isbn(value) -> str | None:
    """Goodreads wraps ISBNs as ="9781250234001"; extract the digits."""
    match = re.search(r"(\d{9}[\dX]|\d{13})", str(value))
    return match.group(1) if match else None


def book_isbns(row) -> list[str]:
    """ISBN13 then ISBN10, skipping missing ones (NaN is truthy, so check the type)."""
    return [isbn for isbn in (row["isbn13"], row["isbn10"]) if isinstance(isbn, str)]


def clean_description(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = re.sub(r"<[^>]+>", " ", text)  # Google Books returns HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= MIN_DESCRIPTION_LENGTH else None


# ---------- HTTP with cache ----------

class CachedClient:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def get_json(self, url: str, retries: int = 3) -> dict | None:
        if url in self.cache:
            return self.cache[url]
        for attempt in range(retries):
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    data = None
                    break
                if e.code == 429 or e.code >= 500:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"  HTTP {e.code} for {url}")
                return None  # don't cache unexpected errors
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  network error for {url}: {e}")
                time.sleep(2 ** (attempt + 1))
        else:
            return None  # retries exhausted, don't cache
        self.cache[url] = data
        time.sleep(0.3)  # be polite to free APIs
        return data

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))


# ---------- source 1: local dataset ----------

def match_local(my_books: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    books = books.dropna(subset=["description"]).copy()
    books["gid"] = pd.to_numeric(books["bookId"].str.extract(r"^(\d+)")[0])
    books["key"] = books["title"].map(normalize_title) + "|" + books["author"].map(normalize_author)

    by_id = books.drop_duplicates("gid").set_index("gid")
    by_isbn = books.drop_duplicates("isbn").set_index("isbn")
    by_key = books.drop_duplicates("key").set_index("key")

    results = []
    for _, row in my_books.iterrows():
        key = normalize_title(row["Title"]) + "|" + normalize_author(row["Author"])
        for index, value, method in [
            (by_id, row["Book Id"], "local_id"),
            (by_isbn, row["isbn13"], "local_isbn"),
            (by_key, key, "local_title_author"),
        ]:
            if pd.notna(value) and value in index.index:
                hit = index.loc[value]
                description = clean_description(hit["description"])
                if description:
                    results.append({"description": description,
                                    "genres": hit["genres"], "source": method})
                    break
        else:
            results.append({"description": None, "genres": None, "source": None})
    return pd.DataFrame(results, index=my_books.index)


# ---------- source 2: Google Books ----------

def fetch_google(client: CachedClient, row) -> dict | None:
    queries = [f"isbn:{isbn}" for isbn in book_isbns(row)[:1]]
    queries.append(f'intitle:{normalize_title(row["Title"])} inauthor:{normalize_author(row["Author"])}')

    for q in queries:
        url = "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(
            {"q": q, "maxResults": 3, "key": GOOGLE_API_KEY})
        data = client.get_json(url)
        for item in (data or {}).get("items", []):
            info = item.get("volumeInfo", {})
            description = clean_description(info.get("description"))
            if description:
                return {"description": description,
                        "genres": json.dumps(info.get("categories", [])),
                        "source": "google_books"}
    return None


# ---------- source 3: Open Library ----------

def _openlibrary_work(client: CachedClient, work_key: str) -> dict | None:
    work = client.get_json(f"https://openlibrary.org{work_key}.json")
    if not work:
        return None
    description = work.get("description")
    if isinstance(description, dict):
        description = description.get("value")
    description = clean_description(description)
    if not description:
        return None
    return {"description": description,
            "genres": json.dumps(work.get("subjects", [])[:10]),
            "source": "open_library"}


def fetch_openlibrary(client: CachedClient, row) -> dict | None:
    work_keys = []
    for isbn in book_isbns(row):
        edition = client.get_json(f"https://openlibrary.org/isbn/{isbn}.json")
        work_keys += [w["key"] for w in (edition or {}).get("works", [])]

    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode({
        "title": normalize_title(row["Title"]),
        "author": normalize_author(row["Author"]),
        "fields": "key",
        "limit": 3,
    })
    work_keys += [doc["key"] for doc in (client.get_json(url) or {}).get("docs", [])]

    for key in dict.fromkeys(work_keys):  # dedupe, keep order
        result = _openlibrary_work(client, key)
        if result:
            return result
    return None


# ---------- main ----------

def main():
    my_books = pd.read_csv(EXPORT_PATH)
    books = pd.read_csv(BOOKS_PATH)

    my_books["isbn13"] = my_books["ISBN13"].map(clean_isbn)
    my_books["isbn10"] = my_books["ISBN"].map(clean_isbn)
    my_books["Author"] = my_books["Author"].str.replace(r"\s+", " ", regex=True).str.strip()

    enriched = match_local(my_books, books)
    print(f"Matched in books.csv: {enriched['source'].notna().sum()} / {len(my_books)}")

    fetchers = [fetch_openlibrary]
    if GOOGLE_API_KEY:
        fetchers.insert(0, fetch_google)
    else:
        print("GOOGLE_BOOKS_API_KEY not set - using Open Library only")

    client = CachedClient(CACHE_PATH)
    missing = enriched.index[enriched["description"].isna()]
    try:
        for i, idx in enumerate(missing, 1):
            row = my_books.loc[idx]
            print(f"[{i}/{len(missing)}] {row['Title'][:70]}")
            for fetch in fetchers:
                result = fetch(client, row)
                if result:
                    enriched.loc[idx, list(result)] = list(result.values())
                    break
            if i % 20 == 0:
                client.save()
    finally:
        client.save()

    output = pd.DataFrame({
        "goodreads_id": my_books["Book Id"],
        "title": my_books["Title"],
        "author": my_books["Author"],
        "isbn13": my_books["isbn13"],
        "my_rating": my_books["My Rating"],
        "shelf": my_books["Exclusive Shelf"],
        "date_read": my_books["Date Read"],
        "year_published": my_books["Original Publication Year"],
        "description": enriched["description"],
        "genres": enriched["genres"],
        "description_source": enriched["source"],
    })
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {OUTPUT_PATH.relative_to(ROOT)}")
    print(output["description_source"].value_counts(dropna=False).to_string())
    still_missing = output[output["description"].isna()]
    if len(still_missing):
        print(f"\nStill missing descriptions ({len(still_missing)}):")
        print(still_missing[["title", "author", "shelf"]].to_string(index=False))


if __name__ == "__main__":
    main()
