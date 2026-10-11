from pydantic import BaseModel
from typing import Literal


class Book(BaseModel):
    book_id: str
    title: str
    author: str | None = None
    description: str | None = None


class BookDetails(Book):
    tags: list[str] = []


class ScoredBook(Book):
    score: float


class RecommendationResponse(BaseModel):
    query: list[Book]
    model: str
    recommendations: list[ScoredBook]


class TitleMatch(Book):
    match: float


class TitleSearchResponse(BaseModel):
    query: str
    status: Literal["match", "did_you_mean", "not_found"]
    books: list[TitleMatch]


class TitleRecommendationResponse(BaseModel):
    title: str
    status: Literal["match", "did_you_mean", "not_found"]
    model: str
    query: Book | None = None
    suggestions: list[TitleMatch] = []
    recommendations: list[ScoredBook] = []