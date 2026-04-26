import os
import uuid
import asyncio
from datetime import datetime

from db import get_db

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import models

# ================== INIT ==================

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.3
)

embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL")
)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# ================== DB INIT ==================

def init_qdrant():
    client = get_db()

    # ✅ Ensure user_id index exists
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # already exists

    return client


# ================== PROCESS ENTRY ==================

async def process_entry(entry):
    text = entry.content

    summary_prompt = f"""
Summarize this journal entry in 2 concise lines.
Focus ONLY on:
- emotional state
- key internal conflict

Entry:
{text}
"""

    signal_prompt = f"""
Extract max 5 emotional/behavioral signals.

Entry:
{text}
"""

    summary_task = asyncio.to_thread(llm.invoke, summary_prompt)
    signal_task = asyncio.to_thread(llm.invoke, signal_prompt)

    summary, signals = await asyncio.gather(summary_task, signal_task)

    summary_text = summary.content.strip()
    signals_text = signals.content.strip()

    # ✅ embed summary only (efficient)
    embedding = await asyncio.to_thread(
        embedding_model.embed_query,
        summary_text
    )

    return summary_text, signals_text, embedding


# ================== STORE ENTRY ==================

def store_entry(entry, summary, signals, embedding):
    client = init_qdrant()

    # ✅ VALID UUID (deterministic)
    point_id = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{entry.user_id}_{entry.entry_id}"
    ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "user_id": entry.user_id,
                    "entry_id": entry.entry_id,
                    "date_time": entry.date_time.isoformat(),
                    "summary": summary,
                    "signals": signals,
                    "text": entry.content
                }
            )
        ]
    )


# ================== FETCH ENTRIES ==================

def get_entries_by_date(user_id: str, date: str):
    client = get_db()

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]
        ),
        limit=100,
        with_payload=True,
    )

    entries = []

    for point in results:
        payload = point.payload or {}
        dt = payload.get("date_time", "")

        if dt.startswith(date):
            entries.append(payload.get("text", ""))

    return entries


# ================== SUMMARIZE DAY ==================

async def summarize_day(entries: list[str]):
    if not entries:
        return "No entries found for this day."

    combined = "\n\n".join(entries)

    prompt = f"""
You are a thoughtful journaling assistant.

1. Summarize the day emotionally
2. Highlight stress / burnout / exams if present
3. Ask 1-2 reflective questions
4. End with a short supportive message

Journal:
{combined}
"""

    response = await asyncio.to_thread(llm.invoke, prompt)

    return response.content.strip()
