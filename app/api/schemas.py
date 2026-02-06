# app/api/schemas.py

from pydantic import BaseModel
from typing import List


# ---------------- Query Request ----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# ---------------- Source Model ----------------
class Source(BaseModel):
    document_name: str
    source: str
    page_number: int
    chunk_id: str


# ---------------- Query Response ----------------
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
