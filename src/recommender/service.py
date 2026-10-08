from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

from pathlib import Path
from recommender.data_preprocessing import popularity_scores, BOOKS_PATH, load_books
from recommender.models import Recommender, PopularityRecommender, seeds_from_positions, empty_seeds

MAX_SEEDS = 20

class RecommenderService:
    def __init__(self, books: pd.DataFrame) -> None:
        self.books = books
        self.ids = books.index.to_numpy()
        self.positions = {book_id: i for i, book_id in enumerate(self.ids)}
        self.models: dict[str, Recommender] = {
            "popularity": PopularityRecommender(popularity_scores(books)),
        }

    def _book(self, pos: int) -> dict:
        row = self.books.iloc[pos]
        return {
            "book_id": str(self.ids[pos]),
            "title": str(row["title"]),
            "author": None if pd.isna(row["author"]) else str(row["author"]),
        }

    def get_book(self, book_id: str) -> dict:
        if book_id not in self.positions:
            raise KeyError(f"Book with id {book_id} does not exist in the database")
        return self._book(self.positions[book_id])

    def _results(self, model: str, seeds: sp.csr_matrix, k: int) -> list[dict]:
        if model not in self.models:
            raise ValueError(f"Unknown model {model!r}; available models: {sorted(self.models)}")
        positions, scores = self.models[model].recommend(seeds, k)
        return [{**self._book(pos), "score": round(float(score), 4)} for pos, score in zip(positions[0], scores[0])]

    def popular(self, k: int = 10) -> list[dict]:
        """Most popular books overall: no query book."""
        return self._results("popularity", empty_seeds(len(self.ids)), k)

    def recommend(self, book_ids: list[str], model: str = "popularity", k: int = 10) -> list[dict]:
        """Recommendations for specified books (the books themselves are excluded)."""
        book_ids = list(dict.fromkeys(book_ids))  # drop repeats, keep order
        if not 1 <= len(book_ids) <= MAX_SEEDS:
            raise ValueError(f"Give between 1 and {MAX_SEEDS} books")
        unknown = [b for b in book_ids if b not in self.positions]
        if unknown:
            raise KeyError(unknown)
        seeds = seeds_from_positions([[self.positions[b] for b in book_ids]], len(self.ids))
        return self._results(model, seeds, k)


def build_recommender_service(data_path: Path = BOOKS_PATH) -> RecommenderService:
    return RecommenderService(load_books(data_path))
