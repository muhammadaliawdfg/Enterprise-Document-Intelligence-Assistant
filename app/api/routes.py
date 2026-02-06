# router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import logging

from app.services.ingestion import ingest_pdf
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.rag import rag_pipeline
from app.api.schemas import QueryRequest, QueryResponse

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Router ----------------
router = APIRouter(tags=["EDIA API"])
UPLOAD_DIR = "storage/documents"

# ---------------- Health Check ----------------
@router.get("/health")
def health_check():
    return {
        "status": "running",
        "service": "EDIA RAG Backend",
        "version": "1.0"
    }

# ---------------- Upload Endpoint ----------------
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    logger.info(f"Upload request received: {file.filename}")

    # Validate PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Ensure storage directory
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    try:
        if os.path.exists(file_path):
            logger.info(f"File already exists and will be overwritten: {file.filename}")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"File saving failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )

    # ---------------- Module 1–2: Ingestion ----------------
    chunks = ingest_pdf(file_path, file.filename)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text found in PDF"
        )

    # ---------------- Module 3: Embeddings ----------------
    embedding_service = EmbeddingService()
    embedded_chunks = embedding_service.embed_chunks(chunks)

    # ---------------- Module 4: Vector Store ----------------
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

    logger.info(f"File indexed successfully: {file.filename}")

    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "embedding_dim": len(embedded_chunks[0]["embedding"])
    }

# ---------------- Query Endpoint (Module 5 RAG) ----------------
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):

    logger.info(f"Query received: {request.query}")

    try:
        result = rag_pipeline(
            request.query,
            top_k=request.top_k
        )
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )
