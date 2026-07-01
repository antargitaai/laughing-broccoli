import os
import uuid
import time
import asyncio
from db import get_db
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import models
import os
from fastembed import SparseTextEmbedding

from dotenv import load_dotenv
load_dotenv()

llm = None

def init_llm():
    global llm

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("MODEL_NAME")
    )

sparse_model = SparseTextEmbedding(model_name=os.getenv("SPARSE_EMBEDDING_MODEL"))
dense_embedding_model = OpenAIEmbeddings( api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("DENSE_EMBEDDING_MODEL") )

async def process_entry(entry):

    text = entry.content

    summary_prompt = f"""
You are an information extraction system.

Your task is to produce a factual summary of the journal entry.

Rules:
- Extract only information explicitly stated in the text.
- Do not infer emotions, intentions, motivations, personality traits, relationships, or mental state unless they are directly expressed.
- If evidence is insufficient, state that no clear emotional information is present.
- Preserve uncertainty. Never fill gaps with assumptions.
- Keep the summary under 25 words.
- Write in neutral, objective language suitable for semantic search.

    Entry:
    {text}
    """

    signal_prompt = f"""
You are an information extraction system.

Extract only observable emotional or behavioral signals that are explicitly supported by the text.

Rules:
- Do not infer hidden feelings or motivations.
- Do not perform psychological analysis.
- Do not exaggerate ordinary conversation.
- If no meaningful signals are present, return an empty list.
- Each signal should be 2-6 words.
    Entry:
    {text}
    """

    summary_task = asyncio.to_thread(llm.invoke, summary_prompt)
    signal_task = asyncio.to_thread(llm.invoke, signal_prompt)

    summary, signals = await asyncio.gather(summary_task, signal_task)

    summary_text = summary.content.strip()
    signals_text = signals.content.strip()

    # ✅ Embed summary instead of full text
    dense = await asyncio.to_thread(
        dense_embedding_model.embed_query, 
        summary_text,
    )
    sparse = next(sparse_model.embed([summary_text]))

    return (
        summary_text,
        signals_text,
        dense,
        sparse,
    )

def store_entry(entry, summary, signals, dense, sparse):
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
                vector={
                    "antargita-dense-vectors": dense,
                    "antargita-sparse-vectors": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
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

    combined_text = "\n\n".join(entries)

    prompt = f"""
You are a calm, wise, and compassionate journaling companion, speaking like Krishna—gentle, insightful, and deeply caring.

Based on the user’s journal entries:

- Respond in ONLY 1–2 lines (strictly do not exceed this limit)
- Your response must be in Hinglish ONLY (no pure English sentences)
- Include a few relevant emojis (nature + expressions, dont COMBINE 2-3 emojis consecutively, you can use 2 3 emojis in different places, but not clump up.)
- Sound warm, supportive, and personal—not robotic
- If the journal reflects stress, negativity, confusion, or self-doubt, encourage and motivate the user on the basis of bhagwad geeta updesh, but  DO NOT INCLUDE SHLOK OR ANY BHAGWAD GEETA REFERENCE IN THE RESPONSE.
- Otherwise, offer a short, uplifting reflection or encouragement
- Maintain a serene, wise, and friendly tone, like quiet Krishna-like guidance

Special case:
- If NO journal entries are provided or text is empty, gently express concern and ask why they haven’t written today

Journal Entries:
{combined_text}
"""

    response = await asyncio.to_thread(llm.invoke, prompt)

    return response.content.strip()