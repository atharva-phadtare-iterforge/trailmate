import requests
import uuid

text = "An easy trail around a beautiful lake with several waterfalls."

response = requests.post(
    "http://127.0.0.1:8001/embed",
    json={"text": text},
)

response.raise_for_status()

embedding = response.json()["embedding"]

print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])