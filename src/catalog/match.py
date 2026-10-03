"""Find records that describe the same book, as edges between record ids.

Records are identified by their positional index (rid) in the concatenated records frame.
Two exact rules, in order of confidence:
  1. "isbn"         - records share an ISBN-13
  2. "title_author" - records share normalised title + main author key

Guards against wrong merges (a wrong merge is worse than a missed one):
  - groups larger than a limit are skipped: an ISBN or title+author shared by dozens of
    records is bad data or a generic title ("Black Beauty" retellings, "Collected Poems")
  - ISBN edges are dropped when the two titles disagree: sources do mislabel ISBNs
    (7k lists 9780345339737 as "The Fellowship of the Ring", Book-Crossing as "The Return of the King")
"""

from __future__ import annotations

import logging

import pandas as pd

from catalog.normalize import author_key, normalize_title

logger = logging.getLogger(__name__)

EDGE_COLUMNS = ["rid_a", "rid_b", "method"]
MAX_GROUP_SIZE = 10
MIN_TITLE_OVERLAP = 0.5

# Ignored when comparing titles: they make unrelated titles look alike.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "in", "on", "to", "for", "with", "at", "by", "from",
    "book", "books", "volume", "vol", "part", "novel", "edition",
})


def titles_compatible(a: str | None, b: str | None, min_overlap: float = MIN_TITLE_OVERLAP) -> bool:
    """Do two normalised titles plausibly name the same book?

    Overlap of content words relative to the shorter title, so "hobbit" vs
    "hobbit or there and back again" passes and "fellowship of the ring" vs
    "return of the king" fails.
    """
    if not a or not b:
        return False
    ta, tb = set(a.split()) - _TITLE_STOPWORDS, set(b.split()) - _TITLE_STOPWORDS
    if not ta or not tb:          # title made only of stopwords, e.g. "It"
        return a == b
    return len(ta & tb) / min(len(ta), len(tb)) >= min_overlap


def match_keys(records: pd.DataFrame) -> pd.Series:
    """'fellowship of the ring|tolkien_j' per record; NA if title or author key is missing."""
    # astype("str"): an all-missing column would otherwise be object dtype and break the "+"
    title = records["title"].map(normalize_title).astype("str")
    author = records["authors"].map(lambda a: author_key(a[0]) if len(a) else None).astype("str")
    key = title + "|" + author
    return key.where(title.notna() & author.notna())


def _edges_from_groups(pairs: pd.DataFrame, key: str, method: str, max_group_size: int) -> pd.DataFrame:
    """Connect every rid in a key group to the group's smallest rid (a star is enough for union-find)."""
    pairs = pairs.dropna(subset=[key]).drop_duplicates()
    size = pairs.groupby(key)["rid"].transform("size")

    too_big = pairs[size > max_group_size]
    if len(too_big):
        examples = too_big[key].value_counts().head(5).to_dict()
        logger.warning("%s: skipped %d groups larger than %d, e.g. %s",
                       method, too_big[key].nunique(), max_group_size, examples)

    pairs = pairs[(size >= 2) & (size <= max_group_size)]
    root = pairs.groupby(key)["rid"].transform("min")
    edges = pd.DataFrame({"rid_a": root, "rid_b": pairs["rid"], "method": method})
    edges = edges[edges["rid_a"] != edges["rid_b"]]
    logger.info("%s: %d edges", method, len(edges))
    return edges


def find_edges(records: pd.DataFrame, max_group_size: int = MAX_GROUP_SIZE) -> pd.DataFrame:
    """All match edges between records. `records` must have a 0..n-1 RangeIndex."""
    if not records.index.equals(pd.RangeIndex(len(records))):
        raise ValueError("records must have a RangeIndex (rid = position)")

    isbn_pairs = records["isbn13s"].explode().rename("isbn").rename_axis("rid").reset_index()
    key_pairs = match_keys(records).rename("key").rename_axis("rid").reset_index()

    isbn_edges = _edges_from_groups(isbn_pairs, "isbn", "isbn", max_group_size)
    titles = records["title"].map(normalize_title).to_numpy()
    compatible = [titles_compatible(titles[a], titles[b]) for a, b in zip(isbn_edges["rid_a"], isbn_edges["rid_b"])]
    logger.info("isbn: dropped %d edges with conflicting titles", len(isbn_edges) - sum(compatible))
    isbn_edges = isbn_edges[compatible]

    # ISBN first: cluster.py applies edges in order, so stronger evidence wins conflicts.
    edges = pd.concat([
        isbn_edges,
        _edges_from_groups(key_pairs, "key", "title_author", max_group_size),
    ], ignore_index=True)
    return edges[EDGE_COLUMNS].astype({"rid_a": "int64", "rid_b": "int64"})
