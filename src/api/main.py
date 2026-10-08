from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import books, recommendations
from recommender.service import build_recommender_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = build_recommender_service()
    yield


app = FastAPI(title="Book recommender", version="0.1.0", lifespan=lifespan)
app.include_router(books.router)
app.include_router(recommendations.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}