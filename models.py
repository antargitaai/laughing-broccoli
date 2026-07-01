from pydantic import BaseModel
from datetime import datetime
import uuid

class JournalEntry(BaseModel):
    entry_id: uuid.UUID
    user_id: uuid.UUID
    content: str
    date_time: datetime  