from pgvector import Vector

from ..database.connection import get_connection
from ..init.service_client import get_embedding


def search_trails(query: str, top_k: int) -> list[dict]:
    # Get embedding from embedding service
    embedding = Vector(get_embedding(query))

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
                    embedding <=> %s AS distance
                FROM trail_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    embedding,
                    embedding,
                    top_k,
                ),
            )

            rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append(
                {
                    "id": str(row[0]),
                    "trail_id": row[1],
                    "trail_name": row[2],
                    "section_name": row[3],
                    "chunk_text": row[4],
                    "distance": float(row[5]),
                }
            )

        return results

    finally:
        connection.close()