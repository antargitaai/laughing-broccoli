import os
import uuid
import time
import asyncio
from db import get_db
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import models
import os

llm = None

def init_llm():
    global llm

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME")
    )

embedding_model = OpenAIEmbeddings( api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("EMBEDDING_MODEL") )

async def process_entry(entry):

    text = entry.content

    summary_prompt = f"""
    Summarize this journal entry in 2 concise lines.
    Focus ONLY on:
    - emotional state
    - key internal conflict

    Avoid storytelling. Be precise.

    Entry:
    {text}
    """

    signal_prompt = f"""
    Extract key emotional and behavioral signals (max 5).

    Entry:
    {text}
    """

    summary_task = asyncio.to_thread(llm.invoke, summary_prompt)
    signal_task = asyncio.to_thread(llm.invoke, signal_prompt)

    summary, signals = await asyncio.gather(summary_task, signal_task)

    summary_text = summary.content.strip()
    signals_text = signals.content.strip()

    # ✅ Embed summary instead of full text
    embedding = await asyncio.to_thread(
        embedding_model.embed_query, summary_text
    )

    return (
        summary_text,
        signals_text,
        embedding
    )

def store_entry(entry, summary, signals, embedding):
    client = get_db()

    point_id = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{entry.user_id}_{entry.entry_id}"
    ))

    client.upsert(
        collection_name=os.getenv("COLLECTION_NAME"),
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

def get_entries_by_date(user_id: str, date: str):
    client = get_db()

    results = client.scroll(
        collection_name=os.getenv("COLLECTION_NAME"),
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
        with_vectors=False,
    )

    points = results[0]

    entries = []

    for p in points:
        payload = p.payload or {}

        dt = payload.get("date_time", "")
        if dt.startswith(date):  # ✅ filter here
            entries.append(payload.get("text", ""))

    return entries

async def summarize_day(entries: list[str]):

    if not entries:
        return "No entries found for this day."

    combined_text = "\n\n".join(entries)

    prompt = f"""
You are a calm, wise, and compassionate journaling companion, speaking like Krishna—gentle, insightful, and deeply caring.

Based on the user’s journal entries:

- Respond in ONLY 1–2 lines (strictly do not exceed this limit)
- Your response must be in Hinglish ONLY (no pure English sentences)
- Include a few relevant emojis (nature + expressions 🌿✨🙂😔)
- Sound warm, supportive, and personal—not robotic
- If the journal reflects stress, negativity, confusion, or self-doubt, gently ask a caring question about it
- Otherwise, offer a short, uplifting reflection or encouragement
- Maintain a serene, wise, and friendly tone, like quiet Krishna-like guidance

Special case:
- If NO journal entries are provided or text is empty, gently express concern and ask why they haven’t written today

Journal Entries:
{combined_text}
"""

    response = await asyncio.to_thread(llm.invoke, prompt)

    return response.content.strip()
