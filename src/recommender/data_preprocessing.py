
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOOKS_PATH = ROOT / "data" / "books.parquet"

POPULARITY_SOURCES: tuple[str, ...] = ("gr_num_ratings", "bx_readers")


def load_books(path: Path = BOOKS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist, run `uv run python -m catalog.pipeline` first")
    return pd.read_parquet(path)


def source_percentile(counts: pd.Series) -> pd.Series:
    """0-1 percentile (counts are number of readers/ratings) among books present in this source (count > 0);
    Nan otherwise."""
    return counts.where(counts > 0).rank(pct=True)


def popularity_scores(books: pd.DataFrame, sources: tuple[str, ...] = POPULARITY_SOURCES,
                      how: str = "max") -> np.ndarray:
    """One 0-1 popularity score per book
    how: "max" = popular in any source; "mean" = average over the sources that know the book.
    Books that don't have popularity score in any book get 0.
    """
    missing = [c for c in sources if c not in books.columns]
    if missing:
        raise KeyError(f"Missing popularity columns: {missing}")
    if how not in ("max", "mean"):
        raise ValueError(f"How must be 'max' or 'mean', got {how}")
    percentiles = pd.concat({col: source_percentile(books[col]) for col in sources}, axis=1)
    combined = percentiles.max(axis=1) if how == "max" else percentiles.mean(axis=1)
    return combined.fillna(0.0).to_numpy(dtype=np.float32)