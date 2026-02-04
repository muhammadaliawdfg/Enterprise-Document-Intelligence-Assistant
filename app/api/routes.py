from fastapi import APIRouter, UploadFile, File, HTTPException
import os, shutil
from pydantic import BaseModel

from app.services.ingestion import ingest_pdf
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.rag import rag_pipeline  # Module 5

router = APIRouter()
UPLOAD_DIR = "storage/documents"

# ---------------- Query Models ----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str

# ---------------- Health Check ----------------
@router.get("/health")
def health_check():
    return {"status": "EDIA backend is running"}

# ---------------- Upload Endpoint ----------------
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Module 1–2: Ingestion
    chunks = ingest_pdf(file_path, file.filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # Module 3: Embeddings
    embedding_service = EmbeddingService()
    embedded_chunks = embedding_service.embed_chunks(chunks)

    # Module 4: Vector DB
    vector_store = VectorStoreService()
    documents_for_db = [
        {
            "id": chunk["metadata"]["chunk_id"],
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            "metadata": chunk["metadata"]
        }
        for chunk in embedded_chunks
    ]
    vector_store.add_documents(documents_for_db)

    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "embedding_dim": len(embedded_chunks[0]["embedding"])
    }

# ---------------- Query Endpoint (Module 5 RAG) ----------------
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    RAG endpoint: Retrieve top-k chunks and generate answer using LLM.
    """
    try:
        answer = rag_pipeline(request.query, top_k=request.top_k)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
