import asyncio
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import OPENAI_API_KEY, EMBEDDING_MODEL, MODEL_NAME
import uuid
from db import get_db, COLLECTION_NAME

llm = None

def init_llm():
    global llm

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME
    )

embedding_model = OpenAIEmbeddings( api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL )

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
        collection_name=COLLECTION_NAME,
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