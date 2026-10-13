"""
Item-item collaborative filtering: a book is represented by the readers who interacted with it.
Only book-crossing dataset, Goodreads does not have users data
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from recommender.data_preprocessing import ROOT

BOOKCROSSING_HANDLE = "simonbouchardk/book-recommendation-platform-data/versions/1"


def load_interactions(book_ids: pd.Index) -> pd.DataFrame:
    """(user_id, book_id, rating) for every Book-Crossing interaction"""
    interactions = pd.read_csv(Path(kagglehub.dataset_download(BOOKCROSSING_HANDLE)) / "interactions.csv",
                               dtype={"user_id": str, "work_id": str})
    aliases = pd.read_parquet(ROOT / "data" / "bx_aliases.parquet")["kept_id"]
    interactions["book_id"] = interactions["work_id"].map(aliases).fillna(interactions["work_id"])
    interactions = interactions[interactions["book_id"].isin(book_ids)]
    return interactions.drop_duplicates(["user_id", "book_id"])[["user_id", "book_id", "rating"]]


def item_user_matrix(interactions: pd.DataFrame, book_ids: np.ndarray) -> sp.csr_matrix:
    """(n_books, n_users): 1 if a reader interacted with a book; rows L2-normalised.
    """
    position = pd.Series(np.arange(len(book_ids)), index=book_ids)
    users = interactions["user_id"].astype("category")
    rows = interactions["book_id"].map(position).to_numpy()
    cols = users.cat.codes.to_numpy()
    matrix = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                           shape=(len(book_ids), len(users.cat.categories)))
    return normalize(matrix)