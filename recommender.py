# recommender.py

import pandas as pd
from rag_pipeline import get_recommendations
from llama_classifier import classify_llama

# ==============================
# Load Dataset
# ==============================
try:
    books = pd.read_csv("books_cleaned.csv")
    print("✅ Dataset loaded successfully")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# Normalize column names (important fix)
books.columns = books.columns.str.lower()

# ==============================
# Hybrid Recommendation Function
# ==============================
def hybrid_recommend(query: str, min_rating: float = 3.5):
    """
    Hybrid recommendation system:
    1. RAG-based LLM recommendations
    2. LLM-based category classification
    3. Smart filtering using rating + category + keywords
    """

    if not query or len(query.strip()) == 0:
        return {
            "llm_response": "Please enter a valid query.",
            "category": "unknown",
            "filtered_books": pd.DataFrame()
        }

    print("\n🔍 Step 1: RAG Recommendations...")
    llm_response = get_recommendations(query)

    print("🏷️ Step 2: Detecting category...")
    category = classify_llama(query)

    print("📊 Step 3: Filtering structured data...")

    # Detect columns safely
    title_col = "title" if "title" in books.columns else None
    author_col = "authors" if "authors" in books.columns else None
    genre_col = "genre" if "genre" in books.columns else None
    rating_col = "average_rating" if "average_rating" in books.columns else None

    # -------------------------------
    # Base filtering (rating)
    # -------------------------------
    if rating_col:
        filtered_books = books[books[rating_col] >= min_rating]
    else:
        filtered_books = books.copy()

    # -------------------------------
    # Genre filtering
    # -------------------------------
    try:
        if genre_col and category != "unknown":
            filtered_books = filtered_books[
                filtered_books[genre_col].astype(str).str.contains(category, case=False, na=False)
            ]
    except Exception as e:
        print("⚠️ Genre filtering issue:", e)

    # -------------------------------
    # Keyword filtering
    # -------------------------------
    try:
        keyword_filter = None

        if title_col:
            keyword_filter = filtered_books[title_col].astype(str).str.contains(query, case=False, na=False)

        if author_col:
            author_filter = filtered_books[author_col].astype(str).str.contains(query, case=False, na=False)
            keyword_filter = keyword_filter | author_filter if keyword_filter is not None else author_filter

        if keyword_filter is not None:
            filtered_books = filtered_books[keyword_filter]

    except Exception as e:
        print("⚠️ Keyword filtering issue:", e)

    # -------------------------------
    # Fallback (Top Rated)
    # -------------------------------
    if filtered_books.empty:
        print("⚠️ No exact matches found. Showing top-rated books instead...")
        if rating_col:
            filtered_books = books.sort_values(by=rating_col, ascending=False)
        else:
            filtered_books = books.head(10)

    # -------------------------------
    # Final cleanup
    # -------------------------------
    filtered_books = filtered_books.drop_duplicates(subset=title_col) if title_col else filtered_books

    return {
        "llm_response": llm_response,
        "category": category,
        "filtered_books": filtered_books.head(10)
    }

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("\n📚 Book Recommendation System Ready!")

    while True:
        query = input("\n🔍 Enter your query (or type 'exit'): ")

        if query.lower() == "exit":
            print("👋 Exiting system...")
            break

        result = hybrid_recommend(query)

        print("\n" + "=" * 50)
        print("🤖 LLM Recommendations:\n")
        print(result["llm_response"])

        print("\n" + "=" * 50)
        print(f"🏷️ Detected Category: {result['category']}")

        print("\n" + "=" * 50)
        print("📖 Top Filtered Books:\n")

        for _, row in result["filtered_books"].iterrows():
            title = row.get("title", "N/A")
            author = row.get("authors", "N/A")
            rating = row.get("average_rating", "N/A")

            print(f"➡ {title}")
            print(f"   👤 Author: {author}")
            print(f"   ⭐ Rating: {rating}\n")