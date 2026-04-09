# app.py
import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

from llama_classifier import safe_classify

from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import requests

# -------------------------------
# Load API KEY (Groq)
# -------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Simple fallback classifier
# -------------------------------
def simple_fallback(query):
    return "unknown"

# -------------------------------
# Load Embedding + DB
# -------------------------------
class LocalEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding = LocalEmbedding()

db_books = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# -------------------------------
# Cover Handling (MEDIUM SIZE)
# -------------------------------
def get_cover_url(isbn):
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"

def display_cover(isbn):
    url = get_cover_url(isbn)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.image(url, use_container_width=True)
        else:
            st.image("cover-not-found.jpg", use_container_width=True)
    except:
        st.image("cover-not-found.jpg", use_container_width=True)

# -------------------------------
# 🔥 Fetch REAL book data (TITLE + AUTHOR)
# -------------------------------
def fetch_book_details(isbn):
    try:
        url = f"https://openlibrary.org/isbn/{isbn}.json"
        res = requests.get(url)

        if res.status_code != 200:
            return None, None

        data = res.json()

        title = data.get("title", None)

        # Fetch author name
        author_name = None
        if "authors" in data:
            author_key = data["authors"][0]["key"]
            author_res = requests.get(f"https://openlibrary.org{author_key}.json")

            if author_res.status_code == 200:
                author_data = author_res.json()
                author_name = author_data.get("name", None)

        return title, author_name

    except:
        return None, None

# -------------------------------
# Retrieve Books (UNIQUE + API)
# -------------------------------
def get_books(query):
    docs = db_books.similarity_search(query, k=20)
    books = []
    seen_isbns = set()

    for doc in docs:
        try:
            parts = doc.page_content.split()
            isbn = parts[0]

            # Remove duplicates
            if isbn in seen_isbns:
                continue
            seen_isbns.add(isbn)

            # 🔥 Fetch real data
            title, author = fetch_book_details(isbn)

            if not title:
                title = "Unknown Title"

        except:
            continue

        books.append({
            "isbn": isbn,
            "title": title,
            "author": author
        })

    return books

# -------------------------------
# GROQ LLM FUNCTION
# -------------------------------
def groq_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(
    page_title="📚 Book Recommender",
    layout="wide"
)

st.title("📚 LLM-Based Book Recommendation System")
st.write("AI-powered recommendations using RAG + LLM 🚀")

# -------------------------------
# USER INPUT
# -------------------------------
query = st.text_input("🔍 Enter your interest (e.g., magic, romance, science):")

if st.button("Recommend Books"):

    if not query.strip():
        st.warning("⚠️ Please enter a valid query")

    else:
        with st.spinner("🤖 Thinking..."):

            # -------------------------------
            # Step 1: Classification
            # -------------------------------
            category = safe_classify(query, simple_fallback)
            st.subheader("📂 Predicted Category")
            st.success(category)

            # -------------------------------
            # Step 2: LLM Recommendations
            # -------------------------------
            rag_prompt = f"""
            User interest: {query}
            Category: {category}

            Recommend books with short explanations.
            """

            result = groq_llm(rag_prompt)

            st.subheader("🧠 AI Recommendations")
            st.write(result)

            # -------------------------------
            # Step 3: Book Grid UI
            # -------------------------------
            st.subheader("📚 Books You May Like")

            books = get_books(query)

            cols = st.columns(5)

            for i, book in enumerate(books):
                with cols[i % 5]:

                    display_cover(book["isbn"])

                    st.markdown(f"**📖 {book['title']}**")

                    # Show author only if available
                    if book["author"]:
                        st.caption(f"👤 {book['author']}")