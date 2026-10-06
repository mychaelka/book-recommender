from typing import Annotated

from fastapi import Depends, Request

from recommender.service import RecommenderService


def get_service(request: Request) -> RecommenderService:
    return request.app.state.service


ServiceDep = Annotated[RecommenderService, Depends(get_service)]