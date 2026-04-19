import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
client = None

def init_db():
    global client

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if os.getenv("COLLECTION_NAME") not in names:
        client.create_collection(
            collection_name=os.getenv("COLLECTION_NAME"),
            vectors_config=VectorParams(
                size=1536,  # match embedding model
                distance=Distance.COSINE
            )
        )

    print("Qdrant initialized.")


def get_db():
    return client