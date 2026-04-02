import re
from typing import List, Tuple

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

_CHAPTER_PATTERN = re.compile(r"\bchapter\s+([0-9]+|[ivxlcdm]+)\b", re.IGNORECASE)


def extract_pages(file_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
    return pages


def chunk_pages(
    book_id: str,
    pages: List[Tuple[int, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    cs = CHUNK_SIZE if chunk_size is None else chunk_size
    co = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: List[Document] = []
    current_chapter = "Unknown"
    for page_num, text in pages:
        match = _CHAPTER_PATTERN.search(text)
        if match:
            raw = match.group(1)
            current_chapter = f"Chapter {raw.upper() if raw.isalpha() else raw}"

        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "book_id": book_id,
                        "page": page_num,
                        "chapter": current_chapter,
                    },
                )
            )
    return docs
