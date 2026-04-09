# llama_classifier.py

import os
from groq import Groq
from dotenv import load_dotenv

# ✅ Load environment variables (supports both names)
load_dotenv()
load_dotenv("API.env")

# 🔹 Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# 🔹 Global labels
LABELS = [
    "fantasy",
    "romance",
    "thriller",
    "science",
    "history",
    "biography",
    "children",
    "mystery"
]


def classify_llama(text: str) -> str:
    """
    Classify book description using Groq Llama model
    """

    # 🔹 Safety check
    if not text or len(text.strip()) == 0:
        return "unknown"

    # 🔹 Limit input size
    text = text[:1000]

    prompt = f"""
You are an expert book classifier.

Classify the following text into EXACTLY ONE category from:
{LABELS}

Rules:
- Only return ONE word from the list
- No explanation
- No punctuation

Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ stable Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )

        result = response.choices[0].message.content

        if not result:
            return "unknown"

        # 🔥 Clean output strongly
        result = result.strip().lower()
        result = result.replace(".", "").replace(",", "").split()[0]

        # 🔹 Validate label
        if result not in LABELS:
            return "unknown"

        return result

    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return "unknown"


def safe_classify(text: str, fallback_function) -> str:
    """
    Safe classification with fallback
    """

    result = classify_llama(text)

    if result == "unknown":
        print("[INFO] Using fallback classifier...")
        return fallback_function(text)

    return result


# 🔹 Test block
if __name__ == "__main__":
    test_text = "A magical story about dragons and adventure"

    print("Testing Classification...")
    print("Input:", test_text)
    print("API KEY LOADED:", os.getenv("GROQ_API_KEY") is not None)
    print("Output:", classify_llama(test_text))