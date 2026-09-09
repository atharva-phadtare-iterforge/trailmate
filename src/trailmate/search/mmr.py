import numpy as np

def cosine_similarity(vector_a: list[float], vector_b: list[float],) -> float:
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)

def normalize_scores(candidates: list[dict]) -> None:
    scores = [
        candidate["rerank_score"]
        for candidate in candidates
    ]

    if not scores:
        return

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        for candidate in candidates:
            candidate["relevance_score"] = 1.0

        return

    for candidate in candidates:
        candidate["relevance_score"] = (
            (candidate["rerank_score"] - minimum)
            / (maximum - minimum)
        )


def mmr_select(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int,
    lambda_value: float = 0.7,
) -> list[dict]:

    if not candidates:
        return []

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(
            "lambda_value must be between 0 and 1"
        )

    normalize_scores(candidates)

    remaining = candidates.copy()
    selected = []

    while remaining and len(selected) < top_k:
        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:
            relevance = candidate["relevance_score"]

            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    cosine_similarity(
                        candidate["embedding"],
                        selected_candidate["embedding"],
                    )
                    for selected_candidate in selected
                )

            mmr_score = (
                lambda_value * relevance
                - (1 - lambda_value) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        best_candidate["mmr_score"] = best_score

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected