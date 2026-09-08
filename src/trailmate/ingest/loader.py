import uuid
from pathlib import Path

from ..database.connection import get_connection
from ..init.service_client import get_embedding
from .chunker import chunk_markdown


DATA_DIR = Path("data/trails")


def load_trails():
    files = sorted(DATA_DIR.glob("*.md"))

    if not files:
        raise RuntimeError(
            f"No Markdown files found in {DATA_DIR}"
        )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            # Clear existing data so we don't create duplicates
            cursor.execute("DELETE FROM trail_chunks")

            total_chunks = 0

            for file_path in files:
                chunks = chunk_markdown(file_path)

                print(
                    f"{file_path.name:35} -> "
                    f"{len(chunks)} chunks"
                )

                for chunk in chunks:
                    embedding = get_embedding(
                        chunk["chunk_text"]
                    )

                    cursor.execute(
                        """
                        INSERT INTO trail_chunks (
                            id,
                            trail_id,
                            trail_name,
                            section_name,
                            chunk_text,
                            embedding
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        """,
                        (
                            uuid.uuid4(),
                            chunk["trail_id"],
                            chunk["trail_name"],
                            chunk["section_name"],
                            chunk["chunk_text"],
                            embedding,
                        ),
                    )

                    total_chunks += 1

            connection.commit()

            print()
            print(f"Loaded {total_chunks} chunks")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    load_trails()