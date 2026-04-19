import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File

from db import init_db
from models import JournalEntry
from services import init_llm, process_entry, store_entry


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_llm()
    init_db()   # ✅ only once

    print("Server started. Models and DB loaded.")
    yield

    print("Server shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "running"}


# 🔹 Single entry ingestion
@app.post("/ingest")
async def ingest(entry: JournalEntry):

    summary, signals, embedding = await process_entry(entry)

    store_entry(entry, summary, signals, embedding)  # ✅ fixed

    return {
        "message": "Entry processed",
        "summary": summary,
        "signals": signals
    }


# 🔹 Bulk upload JSON
@app.post("/uploadjson")
async def upload_json(file: UploadFile = File(...)):

    content = await file.read()
    content_str = content.decode("utf-8")

    data = json.loads(content_str)

    # Normalize to list
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise Exception(f"Invalid JSON format. Got: {type(data)}")

    results = []

    for entry_dict in data:

        if isinstance(entry_dict, str):
            entry_dict = json.loads(entry_dict)

        if not isinstance(entry_dict, dict):
            raise Exception(f"Still not a dict: {entry_dict}")

        entry = JournalEntry(**entry_dict)

        summary, signals, embedding = await process_entry(entry)

        store_entry(entry, summary, signals, embedding)  # ✅ fixed

        results.append({
            "entry_id": entry.entry_id,
            "summary": summary,
            "signals": signals
        })

    return {"processed": results}