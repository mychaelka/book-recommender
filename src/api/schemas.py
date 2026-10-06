from pydantic import BaseModel


class Book(BaseModel):
    work_id: str
    title: str
    authors: list[str]
    year: int | None = None


class RecommendationResponse(BaseModel):
    query: Book
    model: str
    recommendations: list[Book]