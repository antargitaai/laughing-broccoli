import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
client = None

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
import os

client = None

def init_db():
    global client

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # ✅ Auto-detect vector size
    if EMBEDDING_MODEL == "text-embedding-3-large":
        vector_size = 3072
    elif EMBEDDING_MODEL == "text-embedding-3-small":
        vector_size = 1536
    else:
        raise ValueError("Unsupported embedding model")

    # ---------------- CHECK COLLECTION ----------------
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print("✅ Collection created")

    else:
        print("✅ Collection already exists")

    # ---------------- CREATE INDEX (CRITICAL) ----------------
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("✅ user_id index created")
    except Exception:
        print("ℹ️ user_id index already exists")

    print("🚀 Qdrant initialized successfully")


def get_db():
    global client
    if client is None:
        init_db()
    return client


def get_db():
    return client
