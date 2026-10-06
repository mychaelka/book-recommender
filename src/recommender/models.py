"""
Fitting models (TF-IDF, embeddings,...)
    1. Popularity Recommender: The baseline other models should beat
    2. Content Recommender: TF-IDF over title + subtitle, book description and tags
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Recommender:
    """
    Base Recommender class. Subclasses must implement `score()`.
    Seeds is a matrix of shape (n_queries, n_books). For serving: n_queries = 1. For eval: n_queries is larger.
    Columns correspond to books that the user has specified (sth like LOTR), or user's history of multiple books.
    """
    def score(self, seeds: sp.csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, seeds: sp.csr_matrix, k: int = 10, without_seeds: bool = True) -> np.ndarray:
        """
        :param without_seeds: Never recommend books already mentioned
        :return: Row positions of the top k books.
        """
        scores = self.score(seeds)
        if without_seeds:
            rows, cols = seeds.nonzero()
            scores[rows, cols] = -np.inf
        return top_k(scores, k)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]  # first k for every row (query, unordered)
    order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]  # order those first k
    return np.take_along_axis(idx, order, axis=1)  # apply indices from order onto idx and return


@dataclass(frozen=True, eq=False)
class PopularityRecommender(Recommender):
    """
    Baseline recommender based on popularity
    """
    popularity: np.ndarray

    def score(self, seeds: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.popularity, (seeds.shape[0], 1)).astype(np.float32)


@dataclass(frozen=True)
class ContentRecommender(Recommender):
    pass


@dataclass(frozen=True)
class HybridRecommender(Recommender):
    pass