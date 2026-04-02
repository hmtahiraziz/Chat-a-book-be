import re

from langchain_core.vectorstores import VectorStore

from app.services.provider_service import Provider, get_chat_model
from app.services.rag_chat_service import dedupe_docs, format_context_blocks
from app.services.vector_service import retrieve_from_store


def _llm(chat_provider: Provider):
    return get_chat_model(chat_provider, temperature=0.2)


def summarize_book(store: VectorStore, max_docs: int = 30, chat_provider: Provider = "ollama") -> str:
    half = max(12, max_docs // 2)
    docs_a = retrieve_from_store(
        store,
        "main plot protagonist conflict central problem rising action climax themes",
        k=half,
    )
    docs_b = retrieve_from_store(
        store,
        "ending resolution denouement key relationships secondary characters setting",
        k=half,
    )
    docs = dedupe_docs(docs_a + docs_b)
    if len(docs) < max_docs:
        docs_c = retrieve_from_store(
            store,
            "important scenes dialogue tone narrative style world building",
            k=max_docs - len(docs) + 4,
        )
        docs = dedupe_docs(docs + docs_c)
    docs = docs[:max_docs]
    context = format_context_blocks(docs)
    prompt = (
        "Summarize this book using only the excerpts below. If something is missing, omit it rather than inventing.\n"
        "Structure your answer with:\n"
        "1) Main plot\n2) Key characters\n3) Major themes\n4) Ending / resolution overview\n\n"
        f"Excerpts:\n{context}"
    )
    return _llm(chat_provider).invoke(prompt).content.strip()


def summarize_chapter(
    store: VectorStore, chapter: str, max_docs: int = 18, chat_provider: Provider = "ollama"
) -> str:
    chapter_norm = chapter.strip().lower()
    chapter_docs = []
    docs = retrieve_from_store(
        store,
        f"events scenes dialogue and plot details from {chapter} section",
        k=80,
        fetch_k=96,
    )
    for doc in docs:
        c = str(doc.metadata.get("chapter", "")).strip().lower()
        if c == chapter_norm:
            chapter_docs.append(doc)
        if len(chapter_docs) >= max_docs:
            break

    if not chapter_docs:
        # Fallback: try matching by chapter number only.
        num = re.findall(r"[0-9ivxlcdm]+", chapter_norm)
        if num:
            target = num[0]
            for doc in docs:
                c = str(doc.metadata.get("chapter", "")).strip().lower()
                if target in c:
                    chapter_docs.append(doc)
                if len(chapter_docs) >= max_docs:
                    break

    if not chapter_docs:
        return "Chapter summary not found in indexed content."

    context = format_context_blocks(chapter_docs)
    prompt = (
        f"Summarize {chapter} of this book using only the excerpts below. "
        "Stay accurate; do not invent scenes or characters not supported by the text.\n\n"
        f"Excerpts:\n{context}"
    )
    return _llm(chat_provider).invoke(prompt).content.strip()
