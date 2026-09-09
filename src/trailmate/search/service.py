from pgvector import Vector

from ..database.connection import get_connection
from ..init.service_client import get_embedding
from .mmr import mmr_select
from .reranker import rerank


CANDIDATE_K = 20
MMR_LAMBDA = 0.7


def search_trails(
    query: str,
    top_k: int,
) -> list[dict]:

    query_embedding = get_embedding(query)

    embedding = Vector(query_embedding)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    trail_id,
                    trail_name,
                    section_name,
                    chunk_text,
                    embedding,
                    embedding <=> %s AS distance
                FROM trail_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    embedding,
                    embedding,
                    CANDIDATE_K,
                ),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    candidates = []

    for row in rows:
        candidates.append(
        {
            "id": str(row[0]),
            "trail_id": row[1],
            "trail_name": row[2],
            "section_name": row[3],
            "chunk_text": row[4],
            "embedding": row[5].to_list(),
            "distance": float(row[6]),
        }
    )

    # Stage 2: Cross-Encoder reranking
    candidates = rerank(
        query=query,
        candidates=candidates,
    )

    # Keep only the strongest chunk for each trail.
    best_by_trail = {}

    for candidate in candidates:
        trail_id = candidate["trail_id"]

        if trail_id not in best_by_trail:
            best_by_trail[trail_id] = candidate

    trail_candidates = list(
        best_by_trail.values()
    )

    # Stage 3: MMR
    results = mmr_select(
        query_embedding=query_embedding,
        candidates=trail_candidates,
        top_k=top_k,
        lambda_value=MMR_LAMBDA,
    )

    # Do not expose internal ranking data.
    for result in results:
        result.pop("embedding", None)
        result.pop("rerank_score", None)
        result.pop("relevance_score", None)
        result.pop("mmr_score", None)

    return results