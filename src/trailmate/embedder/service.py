from .model import model


def generate_embedding(text: str) -> list[float]:
    embedding = model.encode(text)

    return embedding.tolist()