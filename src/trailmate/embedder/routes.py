from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

from .service import generate_embedding


app = FastAPI(
    title="TrailMate Embedder",
)

router = APIRouter()


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    embedding = generate_embedding(request.text)

    return {
        "embedding": embedding
    }


app.include_router(router)