from contextlib import asynccontextmanager

from fastapi import FastAPI

import os
from pathlib import Path

from api.dependencies import ServiceDep
from api.routers import books, recommendations
from recommender.service import build_recommender_service
from recommender.data_preprocessing import BOOKS_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_recommender_service(Path(os.environ.get("BOOKS_PATH", BOOKS_PATH)))
    yield


app = FastAPI(title="Book recommender", version="0.1.0", lifespan=lifespan)
app.include_router(books.router)
app.include_router(recommendations.router)


@app.get("/health")
def health(service: ServiceDep) -> dict:
    return {"status": "ok", "books": len(service.ids), "models": sorted(service.models)}