from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import ServiceDep
from api.schemas import RecommendationResponse

router = APIRouter(prefix="/books", tags=["recommendations"])

ModelName = Literal["popularity"]          # extend as you add models: Literal["popularity", "tfidf", ...]


@router.get("/{work_id}/recommendations")
def recommendations(
    work_id: str,
    service: ServiceDep,
    model: Annotated[ModelName, Query(description="Which model to use")] = "popularity",
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecommendationResponse:
    try:
        query = service.get_book(work_id)
        recs = service.recommend(work_id, model=model, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown work_id {work_id!r}")
    return RecommendationResponse(query=query, model=model, recommendations=recs)