import requests

from ..config import EMBEDDER_URL

JAVA_SEARCH_URL = "http://127.0.0.1:8081/search"

def search_trails(
    query: str,
    top_k: int = 3,
) -> list[dict]:

    response = requests.post(
        JAVA_SEARCH_URL,
        json={
            "query": query,
            "topK": top_k,
        },
        timeout=30,
    )

    response.raise_for_status()

    results = response.json()

    return results

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{EMBEDDER_URL}/embed",
        json={"text": text},
    )

    response.raise_for_status()

    return response.json()["embedding"]