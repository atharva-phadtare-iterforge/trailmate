from pgvector import Vector

from ..database.connection import get_connection
from ..init.service_client import get_embedding


def search_trails(query: str, top_k: int) -> list[dict]:
    # Embed the user's query using the same
    # embedding model used during ingestion.
    embedding = Vector(
        get_embedding(query)
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            # Retrieve the initial candidate chunks.
            candidate_k = max(20, top_k * 5)

            cursor.execute(
                """
                SELECT
                    id,
                    trail_id,
                    trail_name,
                    section_name,
                    chunk_text,
                    embedding <=> %s AS distance
                FROM trail_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    embedding,
                    embedding,
                    candidate_k,
                ),
            )

            rows = cursor.fetchall()

        # Keep only the best matching chunk
        # for each trail.
        best_by_trail = {}

        for row in rows:
            trail_id = row[1]
            distance = float(row[5])

            result = {
                "id": str(row[0]),
                "trail_id": row[1],
                "trail_name": row[2],
                "section_name": row[3],
                "chunk_text": row[4],
                "distance": distance,
            }

            if (
                trail_id not in best_by_trail
                or distance < best_by_trail[trail_id]["distance"]
            ):
                best_by_trail[trail_id] = result

        # Rank the distinct trails by their
        # best matching chunk.
        results = sorted(
            best_by_trail.values(),
            key=lambda result: result["distance"],
        )

        # Return the requested number of trails.
        return results[:top_k]

    finally:
        connection.close()