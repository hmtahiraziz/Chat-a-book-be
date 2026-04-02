"""RAG prompt construction, intent-aware retrieval queries, and context formatting."""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.services.vector_service import retrieve_from_store

MAX_HISTORY_MESSAGES = 12

_COMPARISON_SPLIT = re.compile(
    r"\s+(?:vs\.?|versus|compared to|compared with)\s+",
    re.IGNORECASE,
)


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []
    for d in docs:
        key = d.page_content.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def format_context_blocks(docs: List[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        chapter = doc.metadata.get("chapter")
        loc_parts: list[str] = []
        if page is not None:
            loc_parts.append(f"Page {page}")
        if chapter:
            loc_parts.append(str(chapter))
        loc = " · ".join(loc_parts) if loc_parts else "Location unknown"
        blocks.append(f"[Excerpt {i} | {loc}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def format_history_for_prompt(history: list[tuple[str, str]]) -> str | None:
    if not history:
        return None
    lines: list[str] = []
    for role, content in history[-MAX_HISTORY_MESSAGES:]:
        label = "User" if role == "user" else "Assistant"
        c = content.strip()
        if len(c) > 2000:
            c = c[:2000] + "…"
        lines.append(f"{label}: {c}")
    return "\n".join(lines)


def retrieval_query_for_intent(intent: str, question: str) -> str:
    if intent == "character_qa":
        return (
            f"{question}\n\n"
            "Focus on named characters, their relationships, dialogue, motivations, and actions."
        )
    if intent == "factual_qa":
        return (
            f"{question}\n\n"
            "Focus on explicit facts, definitions, dates, and stated details in the text."
        )
    if intent == "comparison":
        return (
            f"{question}\n\n"
            "Focus on similarities, differences, and contrasts between the subjects mentioned."
        )
    return question


def try_split_comparison(question: str) -> tuple[str | None, str | None]:
    m = _COMPARISON_SPLIT.search(question)
    if not m:
        return None, None
    left = question[: m.start()].strip(' \t,;:[]()"\'')
    right = question[m.end() :].strip(' \t,;:[]()"\'')
    if len(left) < 2 or len(right) < 2:
        return None, None
    return left, right


def gather_documents_for_rag(
    store: VectorStore,
    intent: str,
    question: str,
    k: int,
) -> List[Document]:
    base_q = retrieval_query_for_intent(intent, question)

    if intent == "comparison":
        left, right = try_split_comparison(question)
        if left and right:
            half = max(3, (k + 1) // 2)
            d1 = retrieve_from_store(store, left, k=half)
            d2 = retrieve_from_store(store, right, k=half)
            merged = dedupe_docs(d1 + d2)
            if len(merged) < k:
                extra = retrieve_from_store(store, base_q, k=k - len(merged) + 2)
                merged = dedupe_docs(merged + extra)
            return merged[:k]

    return retrieve_from_store(store, base_q, k=k)


RAG_SYSTEM_RULES = """You are a precise assistant answering ONLY from the book excerpts below.

Rules:
- Use only information from the excerpts. If they do not contain enough to answer, say exactly: Not found in book.
- If excerpts conflict, mention the conflict briefly and prefer the clearer passage.
- After important claims, add brief citations using excerpt numbers, e.g. [1] or [2][3].
- Be concise unless the user asks for detail."""


def build_full_rag_prompt(
    question: str,
    context: str,
    history_block: str | None,
) -> str:
    parts: list[str] = [RAG_SYSTEM_RULES, ""]
    if history_block:
        parts.append("Prior conversation (for context only; excerpts are the source of truth):")
        parts.append(history_block)
        parts.append("")
    parts.append("Excerpts:")
    parts.append(context)
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append(
        "Answer using citations [n] referring to excerpt numbers above. "
        "Do not invent quotes or locations that are not supported by the excerpts."
    )
    return "\n".join(parts)
