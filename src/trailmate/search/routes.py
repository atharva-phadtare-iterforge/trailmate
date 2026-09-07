from fastapi import APIRouter, FastAPI

from .models import SearchRequest, SearchResponse
from .service import search_trails


app = FastAPI(
    title="TrailMate Search",
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = search_trails(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "results": results
    }


app.include_router(router)