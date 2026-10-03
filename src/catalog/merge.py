"""Collapse each cluster of records into one work, field by field.

Rules ("survivorship"):
  title, subtitle, authors, description, language  -> first non-missing value by SOURCE_PRIORITY
  isbn13s, tags                                    -> union over all records (tags keep priority order)
  year                                             -> earliest (each source knows some edition)
  popularity / avg_rating                          -> one column per source, never summed across
                                                      sources (different scales and user bases)
Every chosen text field records which source it came from (<field>_source).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalog.sources import BOOKCROSSING, GOODREADS, GOOGLE7K

# Goodreads first: longest English descriptions and properly cased titles.
# Book-Crossing last: often placeholder descriptions and edition-specific titles ("The red tent").
SOURCE_PRIORITY: tuple[str, ...] = (GOODREADS, GOOGLE7K, BOOKCROSSING)

# work_id comes from the first source in this order present in the cluster. Book-Crossing first,
# so works keep their Open Library id and interactions.csv joins without a lookup.
WORK_ID_PRIORITY: tuple[str, ...] = (BOOKCROSSING, GOODREADS, GOOGLE7K)
_WORK_ID_PREFIX = {BOOKCROSSING: "", GOODREADS: "gr:", GOOGLE7K: "g7k:"}

PRIORITY_FIELDS = ["title", "subtitle", "authors", "description", "language"]
_SOURCE_COLUMN_PREFIX = {GOODREADS: "gr", GOOGLE7K: "g7k", BOOKCROSSING: "bx"}


def _first_by_priority(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Per cluster: first non-missing value of `field` and the source it came from."""
    values = df[field]
    present = values.map(lambda v: len(v) > 0) if field == "authors" else values.notna()
    first = df[present].groupby("cluster", sort=False)[[field, "source"]].first()
    return first.rename(columns={"source": f"{field}_source"})


def _ordered_union(df: pd.DataFrame, field: str) -> pd.Series:
    """Per cluster: union of list values, de-duplicated case-insensitively, priority order kept."""
    exploded = df[["cluster", field]].explode(field).dropna(subset=[field])
    exploded = exploded[exploded[field].str.strip() != ""]
    exploded = exploded.assign(_key=exploded[field].str.casefold().str.strip())
    exploded = exploded.drop_duplicates(["cluster", "_key"])
    return exploded.groupby("cluster", sort=False)[field].agg(list)


def _work_ids(df: pd.DataFrame) -> pd.Series:
    rank = df["source"].map({s: i for i, s in enumerate(WORK_ID_PRIORITY)})
    best = df.assign(_rank=rank).sort_values(["cluster", "_rank", "source_id"]).groupby("cluster").first()
    return best["source"].map(_WORK_ID_PREFIX) + best["source_id"]


def _per_source_stats(df: pd.DataFrame) -> pd.DataFrame:
    """bx_popularity, gr_avg_rating, ... per cluster.

    Book-Crossing records in one cluster are different OL works with different readers, so
    their interaction counts are summed. Goodreads / Google duplicates describe the same
    ratings twice, so the max is taken.
    """
    out = []
    for source, prefix in _SOURCE_COLUMN_PREFIX.items():
        part = df[df["source"] == source].groupby("cluster")
        popularity = part["popularity"].sum() if source == BOOKCROSSING else part["popularity"].max()
        out += [popularity.rename(f"{prefix}_popularity"), part["avg_rating"].mean().rename(f"{prefix}_avg_rating")]
    stats = pd.concat(out, axis=1)
    pop_cols = [c for c in stats.columns if c.endswith("_popularity")]
    stats[pop_cols] = stats[pop_cols].fillna(0).astype("Int64")
    return stats


def merge_records(records: pd.DataFrame, clusters: pd.Series | np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge clustered records.

    Args:
        records: Common-schema records with a RangeIndex.
        clusters: Cluster label per record (same length as records).

    Returns:
        works:        one row per work, indexed by work_id
        work_sources: (work_id, source, source_id) - maps every source record to its work
        work_tags:    (work_id, tag, source) - long format, for tag analysis / per-source weighting
    """
    if len(clusters) != len(records):
        raise ValueError("clusters must have one label per record")

    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    df = records.assign(cluster=clusters, _rank=records["source"].map(rank))
    if df["_rank"].isna().any():
        raise ValueError(f"Unknown sources: {set(df.loc[df['_rank'].isna(), 'source'])}")
    df = df.sort_values(["cluster", "_rank", "source_id"], kind="stable")

    groups = df.groupby("cluster", sort=False)
    parts = [_first_by_priority(df, f) for f in PRIORITY_FIELDS]
    parts += [
        _ordered_union(df, "isbn13s").rename("isbn13s"),
        _ordered_union(df, "tags").rename("tags"),
        groups["year"].min().rename("year"),
        groups["source"].agg(lambda s: sorted(set(s))).rename("sources"),
        groups.size().rename("n_records"),
        _per_source_stats(df),
    ]
    works = pd.concat(parts, axis=1)
    for col in ("isbn13s", "tags", "authors"):
        works[col] = works[col].map(lambda v: v if isinstance(v, list) else [])

    work_id = _work_ids(df)
    works.index = works.index.map(work_id)
    works.index.name = "work_id"
    works = works.drop(columns=["subtitle_source", "language_source"])
    works["has_description"] = works["description"].notna()
    works["has_tags"] = works["tags"].map(len) > 0

    work_sources = pd.DataFrame({
        "work_id": df["cluster"].map(work_id), "source": df["source"], "source_id": df["source_id"],
    }).reset_index(drop=True)

    tags = df[["cluster", "source", "tags"]].explode("tags").dropna(subset=["tags"])
    work_tags = (
        pd.DataFrame({"work_id": tags["cluster"].map(work_id), "tag": tags["tags"], "source": tags["source"]})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return works.sort_index(), work_sources, work_tags
