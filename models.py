from pydantic import BaseModel
from datetime import datetime


class JournalEntry(BaseModel):
    entry_id: str
    user_id: str
    content: str
    date_time: datetime  