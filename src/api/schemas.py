from pydantic import BaseModel


class Book(BaseModel):
    book_id: str
    title: str
    author: str | None = None


class ScoredBook(Book):
    score: float


class RecommendationResponse(BaseModel):
    query: list[Book]
    model: str
    recommendations: list[ScoredBook]
