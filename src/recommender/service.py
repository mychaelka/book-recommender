from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path

from recommender.title_search import TitleIndex
from recommender.data_preprocessing import (
    popularity_scores,
    BOOKS_PATH,
    load_books,
    build_text
)
from recommender.models import (
    PopularityRecommender,
    ContentRecommender,
    fit_tfidf,
    seeds_from_positions,
    empty_seeds,
    PopularityReranker,
    HybridRecommender
)
from recommender.embeddings import load_embeddings

CONTENT_MODELS = {"tfidf", "embeddings", "hybrid", "hybrid_popular"}

MAX_SEEDS = 20
SNIPPET_CHARS = 300
TITLE_ACCEPT = 0.9
TITLE_SUGGEST = 0.6
TITLE_MARGIN = 0.2
MATCH, DID_YOU_MEAN, NOT_FOUND = "match", "did_you_mean", "not_found"
HYBRID_ALPHA, HYBRID_BETA = 0.5, 0.2


class RecommenderService:
    def __init__(self, books: pd.DataFrame, embeddings: np.ndarray | None = None) -> None:
        self.books = books
        popularity = popularity_scores(books)
        self.titles = TitleIndex.build(books["title"], popularity)
        self.ids = books.index.to_numpy()
        self.positions = {book_id: i for i, book_id in enumerate(self.ids)}

        self.tfidf_vectorizer, tfidf_matrix = fit_tfidf(build_text(self.books))
        candidates = books["has_description"].to_numpy()
        self.has_text = np.diff(tfidf_matrix.indptr) > 0  # books with at least one TF-IDF word

        tfidf = ContentRecommender(tfidf_matrix, candidates=candidates)
        self.models = {"popularity": PopularityRecommender(popularity), "tfidf": tfidf}
        if embeddings is not None:
            semantic = ContentRecommender(embeddings, candidates=candidates)
            fused = HybridRecommender(tfidf, semantic, alpha=HYBRID_ALPHA)
            self.models["embeddings"] = semantic
            self.models["hybrid"] = fused
            self.models["hybrid_popular"] = PopularityReranker(fused, popularity, beta=HYBRID_BETA)

    def search_titles(self, query: str, n: int = 5) -> tuple[str, list[dict]]:
        """returns ("match" | "did you mean" | "not found", books with 0-1 match score)"""
        match = self.titles.match_scores(query)
        quality = self.books["has_description"].to_numpy() * 0.05 + self.books["author"].notna().to_numpy() * 0.05
        ranked = match + quality + 0.1 * self.titles.popularity  # 0.1 * because popularity breaks near-ties

        picked = []
        for pos in np.argsort(-ranked):
            if len(picked) == n or match[pos] < TITLE_SUGGEST:
                break
            picked.append(pos)

        books = [{**self._book(p), "match": round(float(match[p]), 3)} for p in picked]
        if not books:
            return NOT_FOUND, []
        best, runner_up = books[0]["match"], (books[1]["match"] if len(books) > 1 else 0.0)
        if best >= TITLE_ACCEPT and best - runner_up >= TITLE_MARGIN:
            return MATCH, books[:1]
        return DID_YOU_MEAN, books


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

    def _results(self, model: str, seeds: sp.csr_matrix, k: int,) -> list[dict]:
        """Top k books from `model`"""
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

        seed_positions = [self.positions[b] for b in book_ids]
        if model in CONTENT_MODELS and not self.has_text[seed_positions].any():
            raise ValueError("Not enough information about these books to find similar ones")

        seeds = seeds_from_positions([seed_positions], len(self.ids))
        return self._results(model, seeds, k)

    def recommend_by_title(self, title: str, model: str = "popularity", k: int = 10) -> dict:
        """Resolve a typed title, then recommend, but only if the match is unambiguous.

        Returns {"status", "query", "suggestions", "recommendations"}:
          match         -> query = the book, recommendations filled
          did_you_mean  -> suggestions filled, no recommendations: the user must choose
          not_found     -> everything empty
        """
        status, matches = self.search_titles(title)
        if status != MATCH:
            return {"status": status, "query": None, "suggestions": matches, "recommendations": []}
        book = matches[0]
        return {
            "status": MATCH,
            "query": book,
            "suggestions": [],
            "recommendations": self.recommend([book["book_id"]], model=model, k=k),
        }


def build_recommender_service(data_path: Path = BOOKS_PATH) -> RecommenderService:
    books = load_books(data_path)
    return RecommenderService(books, load_embeddings(books.index.to_numpy()))
