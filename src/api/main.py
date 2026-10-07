from fastapi import FastAPI
from api.routers import recommendations
app = FastAPI(title="Book recommender", version="0.1.0")
app.include_router(recommendations.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}