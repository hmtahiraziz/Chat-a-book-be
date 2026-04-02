from app.services.provider_service import Provider, get_chat_model

_CATEGORIES = [
    "book_summary",
    "chapter_summary",
    "factual_qa",
    "character_qa",
    "comparison",
    "other",
]


def classify_query(question: str, chat_provider: Provider = "ollama") -> str:
    llm = get_chat_model(chat_provider, temperature=0)
    prompt = (
        "Classify the user question into exactly one label from this list:\n"
        + ", ".join(_CATEGORIES)
        + "\n\n"
        "Hints:\n"
        "- book_summary: summarize the entire book / whole story / full plot overview.\n"
        "- chapter_summary: summarize one chapter (user names or numbers a chapter).\n"
        "- character_qa: about a specific character, who someone is, motivations, dialogue.\n"
        "- factual_qa: what happened when, definitions, explicit facts from the text.\n"
        "- comparison: compare or contrast two or more people, events, themes (vs, versus).\n"
        "- other: general questions not fitting above.\n\n"
        f"Question: {question}\n"
        "Return only the label, lowercase, no punctuation."
    )
    label = llm.invoke(prompt).content.strip().lower()
    return label if label in _CATEGORIES else "other"
