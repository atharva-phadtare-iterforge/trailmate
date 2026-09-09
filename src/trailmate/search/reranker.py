from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    query: str,
    candidates: list[dict],
) -> list[dict]:
    if not candidates:
        return []

    pairs = [
        (
            query,
            candidate["chunk_text"],
        )
        for candidate in candidates
    ]

    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )

    return candidates