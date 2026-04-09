# rag_pipeline.py

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load API KEY (ROBUST)
# -------------------------------
load_dotenv()
load_dotenv("API.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Check your API.env file.")

# -------------------------------
# Custom Local Embedding Class
# -------------------------------
class LocalEmbedding:
    def __init__(self):
        print("📦 Loading local embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


embedding = LocalEmbedding()

# -------------------------------
# Load Chroma Vector DB
# -------------------------------
print("📂 Loading Chroma DB...")

db_books = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

print("✅ Chroma DB loaded")

# -------------------------------
# Initialize LLM
# -------------------------------
print("🤖 Initializing LLM...")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

print("✅ LLM ready")

# -------------------------------
# Main RAG Function
# -------------------------------
def get_recommendations(query: str) -> str:

    if not query or len(query.strip()) == 0:
        return "Please enter a valid query."

    try:
        print("\n🔍 Step 1: Searching vector DB...")
        docs = db_books.similarity_search(query, k=8)  # 🔥 increased results

        if not docs:
            return "No relevant books found."

        print(f"📄 Step 2: Retrieved {len(docs)} documents")

        # 🔥 CLEAN CONTEXT
        context = "\n".join([
            f"- {doc.page_content[:200]}" for doc in docs
        ])

        print("🤖 Step 3: Sending to LLM...")

        prompt = f"""
You are a smart book recommendation assistant.

User Interest:
{query}

Relevant Books:
{context}

Your Task:
- Recommend 5 books
- Mention book names clearly
- Give 1 line reason
- Keep output neat

Format:
1. Book Name - Reason
"""

        response = llm.invoke(prompt)

        print("✅ Step 4: LLM Response received")

        return response.content.strip()

    except Exception as e:
        print("❌ [ERROR] RAG failed:", e)
        return "Sorry, something went wrong while generating recommendations."


# -------------------------------
# Test block
# -------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting RAG Pipeline Test...")

    test_query = "Books about adventure and magic"
    print("📌 Query:", test_query)

    result = get_recommendations(test_query)

    print("\n📚 FINAL RECOMMENDATIONS:\n")
    print(result)