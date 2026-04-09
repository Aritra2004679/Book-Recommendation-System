# api.py
# Run: uvicorn api:app --reload --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ── Import YOUR modules ─────────────────────────────────────
from rag_pipeline import get_recommendations
from llama_classifier import safe_classify
from app import get_books   # reuse logic

app = FastAPI(title="Book Recommender API")

# ── CORS (React connection) ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev (later restrict)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str
    min_rating: float = 3.5


class BookItem(BaseModel):
    isbn: str
    title: str
    author: Optional[str] = None
    cover_url: str


class RecommendResponse(BaseModel):
    category: str
    llm_response: str
    books: List[BookItem]


# ── Fallback classifier ─────────────────────────────────────
def simple_fallback(query):
    return "unknown"


# ── Remove duplicate books (IMPORTANT FIX) ───────────────────
def remove_duplicates(books):
    seen = set()
    unique_books = []

    for b in books:
        key = b.get("isbn")
        if key and key not in seen:
            seen.add(key)
            unique_books.append(b)

    return unique_books


# ── MAIN ENDPOINT ───────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    query = req.query.strip()

    if not query:
        return RecommendResponse(
            category="unknown",
            llm_response="Please enter a valid query.",
            books=[]
        )

    # 1️⃣ Classification
    category = safe_classify(query, simple_fallback)

    # 2️⃣ RAG + LLM
    llm_response = get_recommendations(query)

    # 3️⃣ Get books
    raw_books = get_books(query)

    # 🔥 FIX: remove duplicates
    raw_books = remove_duplicates(raw_books)

    books = []
    for b in raw_books:
        isbn = b.get("isbn")

        # Skip invalid ISBN
        if not isbn:
            continue

        books.append(
            BookItem(
                isbn=isbn,
                title=b.get("title") or "Unknown Title",
                author=b.get("author") or None,
                cover_url=f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
            )
        )

    return RecommendResponse(
        category=category,
        llm_response=llm_response,
        books=books
    )


# ── Health Check ────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}