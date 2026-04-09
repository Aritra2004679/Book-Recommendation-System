from llama_classifier import generate_text

LABELS = ["fantasy", "romance", "mystery", "sci-fi", "horror", "self-help"]

def classify_genre(text):
    prompt = f"""
    You are a strict classifier.

    Classify the given text into ONLY ONE of these genres:
    {", ".join(LABELS)}

    Rules:
    - Output ONLY the genre name
    - No explanation
    - Lowercase only

    Text: {text}
    """

    result = generate_text(prompt).lower().strip()

    # Safety fallback
    for label in LABELS:
        if label in result:
            return label

    return "unknown"