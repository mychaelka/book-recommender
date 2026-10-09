"""
Fitting models (TF-IDF, embeddings,...)
    1. Popularity Recommender: The baseline other models should beat
    2. Content Recommender: TF-IDF over title + subtitle, book description and tags
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass


@dataclass(frozen=True)
class Recommender:
    """
    Base Recommender class. Subclasses must implement `score()`.
    Seeds is a matrix of shape (n_queries, n_books). For serving: n_queries = 1. For eval: n_queries is larger.
    Columns correspond to books that the user has specified (sth like LOTR), or user's history of multiple books.
    """
    def score(self, seeds: sp.csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, seeds: sp.csr_matrix, k: int = 10, without_seeds: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        :param seeds: Matrix of shape (n_queries, n_books); books the user has already read
        :param k: How many books to recommend
        :param without_seeds: Never recommend books already mentioned
        :return: Row positions of the top k books and their corresponding scores.
        """
        scores = self.score(seeds)
        if without_seeds:
            rows, cols = seeds.nonzero()
            scores[rows, cols] = -np.inf
        positions = top_k(scores, k)
        return positions, np.take_along_axis(scores, positions, axis=1)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]  # first k for every row (query, unordered)
    order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]  # order those first k
    return np.take_along_axis(idx, order, axis=1)  # apply indices from order onto idx and return


def empty_seeds(n_items: int, n_queries: int = 1) -> sp.csr_matrix:
    """Queries without any seed book (e.g. 'just show popular books')."""
    return sp.csr_matrix((n_queries, n_items), dtype=np.float32)


def seeds_from_positions(positions: list[list[int]], n_items: int) -> sp.csr_matrix:
    """[[3], [7, 12]] -> 2 x n_items matrix with 1s at the seed positions."""
    rows = [r for r, items in enumerate(positions) for _ in items]
    cols = [c for items in positions for c in items]
    return sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(positions), n_items))


@dataclass(frozen=True, eq=False)
class PopularityRecommender(Recommender):
    """
    Baseline recommender based on popularity
    """
    popularity: np.ndarray

    def score(self, seeds: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.popularity, (seeds.shape[0], 1)).astype(np.float32)


def fit_tfidf(texts: pd.Series) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit TF-IDF on all book texts (title, description, tags). Rows of the matrix
    are L2-normalized."""
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.5,
                                 sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    return vectorizer, vectorizer.fit_transform(texts)


@dataclass(frozen=True, eq=False)
class ContentRecommender(Recommender):
    """
    Content-based recommender using cosine similarity.
    Works with TF-IDF (sparse matrix sp.csr_matrix)
    and embeddings (dense matrix np.ndarray)
    """
    vectors: np.ndarray | sp.csr_matrix  # (n_books, dim), rows L2-normalized; "database"
    candidates: np.ndarray | None = None  # bool (n_books,): books allowed as recommendations

    def score(self, seeds: sp.csr_matrix) -> np.ndarray:
        profile = seeds @ self.vectors  # (n_queries, dim), adds up seed books' vectors; builds user 'taste profile'; "a bit of wizard, a lot of hobbit and a bit of magic"
        scores = profile @ self.vectors.T  # (n_queries, n_books)
        scores = scores.toarray() if sp.issparse(scores) else np.asarray(scores)
        scores = scores.astype(np.float32)

        if self.candidates is not None:
            scores[:, ~self.candidates] = -np.inf

        return scores



@dataclass(frozen=True, eq=False)
class HybridRecommender(Recommender):
    pass