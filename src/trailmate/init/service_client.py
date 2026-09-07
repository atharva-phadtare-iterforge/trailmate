import requests

from ..config import EMBEDDER_URL


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{EMBEDDER_URL}/embed",
        json={"text": text},
    )

    response.raise_for_status()

    return response.json()["embedding"]