import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from config import QDRANT_API_KEY, QDRANT_URL
client = None
COLLECTION_NAME = "journal_entries"


def init_db():
    global client

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1536,  # match embedding model
                distance=Distance.COSINE
            )
        )

    print("Qdrant initialized.")


def get_db():
    return client