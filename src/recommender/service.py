from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from recommender.data_preprocessing import popularity_scores, BOOKS_PATH
from recommender.models import PopularityRecommender

@dataclass(frozen=True)
class Book:
    pass


class RecommenderService:
    def __init__(self, books: pd.DataFrame) -> None:
        self.books = books
        #self.popularity = po
        self.popularity_recommender = PopularityRecommender()


    def get_book(self):
        pass

    def recommend(self, work_id: int, model: Book, k: int) -> list[Book]:
        pass


def build_recommender_service(data_path: Path = BOOKS_PATH) -> RecommenderService:
    pass