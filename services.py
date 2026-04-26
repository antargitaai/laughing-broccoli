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

    point_id = f"{entry.user_id}_{entry.entry_id}"  # ✅ deterministic ID

    client.upsert(
        collection_name=os.getenv("COLLECTION_NAME"),
        points=[
            {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "user_id": entry.user_id,
                    "entry_id": entry.entry_id,
                    "date_time": entry.date_time.isoformat(),
                    "summary": summary,
                    "signals": signals,
                    "text": entry.content
                }
            }
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
                ),
                models.FieldCondition(
                    key="date_time",
                    match=models.MatchText(value=date),  # ✅ FIXED
                ),
            ]
        ),
        limit=50,
        with_payload=True,
        with_vectors=False,
    )

    points = results[0]

    entries = [
        p.payload.get("text", "")
        for p in points
        if p.payload and p.payload.get("text")
    ]

    return entries

async def summarize_day(entries: list[str]):

    if not entries:
        return "No entries found for this day."

    combined_text = "\n\n".join(entries)

    prompt = f"""
You are a thoughtful journaling assistant.

Based on the user's journal entries for the day:

1. Summarize how their day went emotionally and behaviorally
2. Highlight any stress, exam pressure, or concerns (if present)
3. Ask 1–2 gentle reflective questions if needed
4. End with a short motivational / supportive message

Keep it:
- Warm
- Personal
- Not robotic

Pay special attention to:
- stress
- burnout
- exams
- self-doubt

Journal Entries:
{combined_text}
"""

    response = await asyncio.to_thread(llm.invoke, prompt)

    return response.content.strip()
