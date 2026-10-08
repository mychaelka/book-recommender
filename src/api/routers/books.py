from typing import Annotated

from fastapi import APIRouter, Query

from api.dependencies import ServiceDep
from api.schemas import ScoredBook

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/popular")
def popular_books(service: ServiceDep, k: Annotated[int, Query(ge=1, le=100)] = 10) -> list[ScoredBook]:
    return service.popular(k=k)