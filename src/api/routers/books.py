from typing import Annotated

from fastapi import APIRouter, Query, HTTPException

from api.dependencies import ServiceDep
from api.schemas import ScoredBook, BookDetails, TitleSearchResponse

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/popular")
def popular_books(service: ServiceDep, k: Annotated[int, Query(ge=1, le=100)] = 10) -> list[ScoredBook]:
    return service.popular(k=k)


@router.get("/search")
def search_titles(
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Book title; typos are fine")],
    n: Annotated[int, Query(ge=1, le=20)] = 5,
) -> TitleSearchResponse:
    status, books = service.search_titles(q, n=n)
    return TitleSearchResponse(query=q, status=status, books=books)


@router.get("/{book_id}")
def book_details(book_id: str, service: ServiceDep) -> BookDetails:
    try:
        return service.get_book(book_id, full=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown book_id {book_id!r}")

