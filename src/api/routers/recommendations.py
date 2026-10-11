from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import ServiceDep
from api.schemas import RecommendationResponse, TitleRecommendationResponse

from recommender.service import NOT_FOUND

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

ModelName = Literal["popularity", "tfidf", "embeddings", "hybrid", "hybrid_popular"]


@router.get("")
def recommendations(
    service: ServiceDep,
    book_id: Annotated[list[str], Query(min_length=1, max_length=20, description="Repeat for several books")],
    model: Annotated[ModelName, Query(description="Which model to use")] = "popularity",
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecommendationResponse:
    try:
        query = [service.get_book(b) for b in dict.fromkeys(book_id)]
        recs = service.recommend(book_id, model=model, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown book_id(s): {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RecommendationResponse(query=query, model=model, recommendations=recs)


@router.get("/by-title")
def recommendations_by_title(
    service: ServiceDep,
    title: Annotated[str, Query(min_length=1, max_length=200, description="Book title; typos are fine")],
    model: Annotated[ModelName, Query(description="Which model to use")] = "popularity",
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> TitleRecommendationResponse:
    try:
        result = service.recommend_by_title(title, model=model, k=k)
    except ValueError as e:                                  # e.g. the matched book has no usable text
        raise HTTPException(status_code=422, detail=str(e))
    if result["status"] == NOT_FOUND:
        raise HTTPException(status_code=404, detail=f"No book with a title like {title!r}")
    return TitleRecommendationResponse(title=title, model=model, **result)