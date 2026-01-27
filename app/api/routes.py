import os
from fastapi import APIRouter, UploadFile,File,HTTPException
import shutil
router = APIRouter()
UPLOAD_DIR = "storage/documents"
@router.get("/health")
def health_check():
    return {"status": "EDIA backend is running"}
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # check file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    # create storage directory if not exists
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # save file to storage directory
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {
        "message": "File uploaded successfully",
        "filename": file.filename
        }      