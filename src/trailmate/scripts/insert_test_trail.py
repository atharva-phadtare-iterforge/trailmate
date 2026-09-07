import uuid
import requests
import psycopg

text = "An easy trail around a beautiful lake with several waterfalls."

# 1. Get embedding from our embedder service
response = requests.post(
    "http://127.0.0.1:8001/embed",
    json={"text": text},
)

response.raise_for_status()

embedding = response.json()["embedding"]

# 2. Connect to TrailMate database
connection = psycopg.connect(
    "host=localhost port=5433 dbname=trailmate user=litellm password=litellm"
)

# 3. Insert trail chunk
with connection.cursor() as cursor:
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
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            uuid.uuid4(),
            "test-trail-1",
            "Cascade Falls",
            "Overview",
            text,
            embedding,
        ),
    )

connection.commit()
connection.close()

print("Trail inserted successfully!")