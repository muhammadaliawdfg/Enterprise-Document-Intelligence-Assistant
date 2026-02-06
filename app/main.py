# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router  # your router file

app = FastAPI(title="EDIA Assistant")

# ---------------- CORS Setup ----------------
origins = [
    "http://localhost:3000",  # your frontend URL
    "http://127.0.0.1:3000",  # sometimes used
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allow specific frontend
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],         # allow all headers
)

# ---------------- Include Router ----------------
app.include_router(router)
