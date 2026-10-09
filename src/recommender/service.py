from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

from pathlib import Path
from recommender.data_preprocessing import popularity_scores, BOOKS_PATH, load_books, build_text
from recommender.models import (
    Recommender,
    PopularityRecommender,
    ContentRecommender,
    fit_tfidf,
    seeds_from_positions,
    empty_seeds,
)

MAX_SEEDS = 20
SNIPPET_CHARS = 300

class RecommenderService:
    def __init__(self, books: pd.DataFrame) -> None:
        self.books = books
        self.ids = books.index.to_numpy()
        self.positions = {book_id: i for i, book_id in enumerate(self.ids)}
        self.tfidf_vectorizer, tfidf_matrix = fit_tfidf(build_text(self.books))
        candidates = books["has_description"].to_numpy()
        self.models: dict[str, Recommender] = {
            "popularity": PopularityRecommender(popularity_scores(books)),
            "tfidf": ContentRecommender(tfidf_matrix, candidates),
        }

    def _book(self, pos: int, snippet_chars: int | None = SNIPPET_CHARS) -> dict:
        row = self.books.iloc[pos]
        description = None if pd.isna(row["description"]) else str(row["description"])
        if description and snippet_chars and len(description) > snippet_chars:
            description = description[:snippet_chars].rsplit(" ", 1)[0] + "…"
        return {
            "book_id": str(self.ids[pos]),
            "title": str(row["title"]),
            "author": None if pd.isna(row["author"]) else str(row["author"]),
            "description": description,
        }

    def get_book(self, book_id: str, full: bool = False) -> dict:
        if book_id not in self.positions:
            raise KeyError(book_id)
        pos = self.positions[book_id]
        book = self._book(pos, snippet_chars=None if full else SNIPPET_CHARS)
        if full:
            book["tags"] = [str(t) for t in self.books.iloc[pos]["tags"]]
        return book

    def _results(self, model: str, seeds: sp.csr_matrix, k: int, require_signal: bool = False) -> list[dict]:
        """Top k books from `model`. require_signal: fail instead of returning arbitrary books when
        no book scores above 0 (e.g. a content query whose books have no usable text)."""
        if model not in self.models:
            raise ValueError(f"Unknown model {model!r}; available models: {sorted(self.models)}")
        positions, scores = self.models[model].recommend(seeds, k)
        if require_signal and not (scores[0] > 0).any():
            raise ValueError("Not enough information about these books to find similar ones")
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
        is_content_model = isinstance(self.models.get(model), ContentRecommender)
        return self._results(model, seeds, k, require_signal=is_content_model)


def build_recommender_service(data_path: Path = BOOKS_PATH) -> RecommenderService:
    return RecommenderService(load_books(data_path))
