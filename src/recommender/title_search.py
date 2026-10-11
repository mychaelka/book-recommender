"""
Fuzzy title lookup (e.g. when user types 'fellowship' -> 'Did you mean Fellowship of the ring?')
Uses character 3-gram TF-IDF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from catalog.normalize import normalize_title

@dataclass(frozen=True, eq=False)
class TitleIndex:
    vectorizer: TfidfVectorizer
    matrix: sp.csr_matrix  # fitted TF-IDF, titles vs. 3grams
    popularity: np.ndarray  # 0-1, used to break near-ties

    @classmethod
    def build(cls, titles: pd.Series, popularity: np.ndarray) -> TitleIndex:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), dtype=np.float32)
        matrix = vectorizer.fit_transform(titles.map(normalize_title).fillna(""))
        return cls(vectorizer, matrix, popularity)

    def match_scores(self, query: str) -> np.ndarray:
        """0-1 similarity of every book to the query, 0 for all if query is empty"""
        q = normalize_title(query)
        if not q:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return (self.matrix @ self.vectorizer.transform([q]).T).toarray().ravel()