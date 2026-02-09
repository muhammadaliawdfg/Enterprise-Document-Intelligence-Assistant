

from pydantic import BaseModel
from typing import List, Optional


#  Query Request
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# Source Model 
class Source(BaseModel):
    documentName: str
    pageNumber: Optional[int] = None
    text: str


# Query Response 
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
