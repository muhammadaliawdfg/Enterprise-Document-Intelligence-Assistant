from datetime import date
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_cleaner import clean_text

from datetime import datetime

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(text: str):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_pdf(pdf_path: str, document_name: str):
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page in pages:
        cleaned = clean_text(page["text"])
        chunks = chunk_text(cleaned)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "document_name": document_name,
                    "source": document_name,
                    "page_number": page["page_number"],
                    "chunk_id": f"{document_name}_p{page['page_number']}_{idx}",
                    "chunk_index": idx,
                    "token_length": len(chunk.split()),
                    "ingestion_time": datetime.utcnow().isoformat()

                }
            })

    return all_chunks
