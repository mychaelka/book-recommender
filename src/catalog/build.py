"""Build the unified catalogue: load -> match -> cluster -> merge -> validate -> write.

Usage:
    python -m catalog.build                    (from src/, or with src on PYTHONPATH)
    python -m catalog.build --out data/processed

Outputs (parquet) in --out:
    works.parquet         one row per book
    work_sources.parquet  source record -> work_id (use it to join Book-Crossing interactions)
    work_tags.parquet     (work_id, tag, source)
    match_edges.parquet   which records were matched, and how
    records.parquet       every source record before merging (rid = row position), for debugging
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import kagglehub
import pandas as pd

from catalog.cluster import assign_clusters
from catalog.match import find_edges
from catalog.merge import merge_records
from catalog.sources import GOODREADS, load_bookcrossing, load_goodreads, load_google7k

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Pinned versions: a new upload of a dataset must not silently change the catalogue.
BOOKCROSSING_HANDLE = "simonbouchardk/book-recommendation-platform-data/versions/1"
GOOGLE7K_HANDLE = "dylanjcastillo/7k-books-with-metadata/versions/3"
GOODREADS_PATH = ROOT / "data" / "books.csv"

# A "book" made of more records than this is almost certainly chained unrelated books.
MAX_CLUSTER_SIZE = 25


def load_records() -> pd.DataFrame:
    bookcrossing_dir = Path(kagglehub.dataset_download(BOOKCROSSING_HANDLE))
    google7k_dir = Path(kagglehub.dataset_download(GOOGLE7K_HANDLE))
    records = pd.concat([
        load_goodreads(GOODREADS_PATH),
        load_google7k(google7k_dir / "books.csv"),
        load_bookcrossing(bookcrossing_dir),
    ], ignore_index=True)
    return records


def validate(records: pd.DataFrame, works: pd.DataFrame, work_sources: pd.DataFrame) -> None:
    """Hard invariants. Raises on violation so a broken catalogue is never written."""
    errors = []
    if not works.index.is_unique:
        errors.append("work_id is not unique")
    if work_sources.duplicated(["source", "source_id"]).any():
        errors.append("a source record maps to more than one work")
    if len(work_sources) != len(records):
        errors.append(f"{len(records) - len(work_sources)} records lost during merge")
    if not set(work_sources["work_id"]) <= set(works.index):
        errors.append("work_sources references unknown work_ids")
    if (works["n_records"] > MAX_CLUSTER_SIZE).any():
        errors.append(f"clusters larger than {MAX_CLUSTER_SIZE}: chaining")
    if works["title"].isna().any():
        errors.append("works without a title")
    if errors:
        raise ValueError("Catalogue validation failed: " + "; ".join(errors))


def report(records: pd.DataFrame, edges: pd.DataFrame, works: pd.DataFrame) -> str:
    """Human-readable match report: the numbers you show when asked 'how good is the join?'."""
    lines = ["", "=== CATALOGUE BUILD REPORT ==="]

    lines.append("\nRecords per source:")
    lines += [f"  {s:<13} {n:>8,}" for s, n in records["source"].value_counts().items()]

    lines.append("\nMatch edges per method:")
    lines += [f"  {m:<13} {n:>8,}" for m, n in edges["method"].value_counts().items()]

    multi = works["n_records"] > 1
    lines.append(f"\nWorks: {len(works):,}  (merged from >1 record: {multi.sum():,})")
    combos = works["sources"].map(lambda s: " + ".join(s)).value_counts()
    lines += [f"  {c:<45} {n:>8,}" for c, n in combos.items()]

    lines.append("\nCluster sizes (records per work):")
    lines += [f"  {k:>3}: {v:>8,}" for k, v in works["n_records"].value_counts().sort_index().items()]
    lines.append("\nLargest clusters (check for chaining):")
    largest = works.nlargest(8, "n_records")
    lines += [f"  {n:>3}  {wid:<22} {t[:60]}" for wid, n, t in zip(largest.index, largest["n_records"], largest["title"])]

    lines.append("\nDescription coverage, source record alone -> merged work:")
    for source in records["source"].unique():
        before = records.loc[records["source"] == source, "description"].notna().mean()
        after = works.loc[works["sources"].map(lambda s, src=source: src in s), "has_description"].mean()
        lines.append(f"  {source:<13} {before:6.1%} -> {after:6.1%}")
    popular = works["bx_popularity"] >= 20
    lines.append(f"  works with >=20 Book-Crossing interactions: {works.loc[popular, 'has_description'].mean():.1%} "
                 f"of {popular.sum():,}")

    lines.append("\nDescription source (merged works):")
    lines += [f"  {s:<13} {n:>8,}" for s, n in works["description_source"].value_counts(dropna=False).items()]

    lines.append(f"\nWorks with neither description nor tags: {(~works['has_description'] & ~works['has_tags']).sum():,}")

    lotr = works[works["title"].str.contains("Lord of the Rings|Fellowship of the Ring", case=False, na=False)]
    lines.append("\nSanity check - Lord of the Rings works:")
    lines += [f"  {wid:<22} {'+'.join(s):<35} {t[:50]}" for wid, s, t in
              zip(lotr.index[:10], lotr["sources"][:10], lotr["title"][:10])]
    return "\n".join(lines)


def build(out_dir: Path) -> None:
    records = load_records()
    edges = find_edges(records)
    # Goodreads records are distinct books: never let a chain merge two of them.
    clusters = assign_clusters(len(records), edges, exclusive=(records["source"] == GOODREADS).to_numpy())
    works, work_sources, work_tags = merge_records(records, clusters)

    validate(records, works, work_sources)
    logger.info(report(records, edges, works))

    out_dir.mkdir(parents=True, exist_ok=True)
    works.to_parquet(out_dir / "works.parquet")
    work_sources.to_parquet(out_dir / "work_sources.parquet", index=False)
    work_tags.to_parquet(out_dir / "work_tags.parquet", index=False)
    records.rename_axis("rid").reset_index().to_parquet(out_dir / "records.parquet", index=False)
    edges.assign(
        source_a=records["source"].to_numpy()[edges["rid_a"]], id_a=records["source_id"].to_numpy()[edges["rid_a"]],
        source_b=records["source"].to_numpy()[edges["rid_b"]], id_b=records["source_id"].to_numpy()[edges["rid_b"]],
    ).to_parquet(out_dir / "match_edges.parquet", index=False)
    logger.info("Wrote %d works to %s", len(works), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build(args.out)


if __name__ == "__main__":
    main()
