import asyncio
import hashlib
import random
import re
import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import (
    ADMIN_API_TOKEN,
    PINECONE_API_KEY,
    PINECONE_SERVERLESS_CLOUD,
    PINECONE_SERVERLESS_REGION,
    public_vector_store_info,
    require_mongodb_config,
    require_pinecone_config,
)
from app.models import ChatRequest, ClassifyRequest, CreatePineconeIndexRequest, TtsRequest
from app.services.book_service import (
    count_books_with_pdf_file_id,
    delete_book as mongo_delete_book,
    delete_pdf_file,
    get_book,
    list_books,
    read_pdf_bytes,
    store_pdf,
    upsert_book,
)
from app.services.classifier_service import classify_query
from app.services.document_service import chunk_pages, extract_pages
from app.services.progress_service import delete_progress, load_progress, save_progress
from app.services.provider_service import Provider, get_chat_model, get_embedding_model
from app.services.rag_chat_service import (
    build_full_rag_prompt,
    format_context_blocks,
    format_history_for_prompt,
    gather_documents_for_rag,
)
from app.services.summary_service import summarize_book, summarize_chapter
from app.services.tts_service import synthesize_openai_tts_wav
from app.services.vector_service import (
    clear_book_index_vectors,
    index_exists,
    list_book_documents_page,
    load_book_store,
)

app = FastAPI(title="BookChat API (LangChain)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]")
_RETRY_SECONDS = re.compile(r"retry in ([0-9]*\.?[0-9]+)s", re.IGNORECASE)
_TERMINAL_INGEST_STATUSES = frozenset({"completed", "failed", "stopped"})

ingest_status: dict[str, dict[str, Any]] = {}
ingest_control: dict[str, dict[str, bool]] = {}


@app.on_event("startup")
def _validate_required_services() -> None:
    try:
        require_mongodb_config()
        require_pinecone_config()
    except RuntimeError as exc:
        raise RuntimeError(f"Startup configuration error: {exc}") from exc


def verify_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    if not ADMIN_API_TOKEN:
        return
    if x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Admin routes require header X-Admin-Token (set ADMIN_API_TOKEN on server).",
        )


def _sanitize_filename(name: str) -> str:
    from pathlib import Path

    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base) or "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _book_id_from_filename(filename: str) -> str:
    from pathlib import Path

    stem = Path(filename).stem.lower()
    safe = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return safe or f"book-{int(time.time())}"


def _display_filename_and_safe_name(display_name: str | None, upload_filename: str) -> tuple[str, str]:
    from pathlib import Path

    if display_name and display_name.strip():
        label = Path(display_name.strip()).name.strip()
        if label:
            if len(label) > 240:
                raise HTTPException(
                    status_code=400,
                    detail="display_name is too long (max 240 characters).",
                )
            library_name = label if label.lower().endswith(".pdf") else f"{label}.pdf"
            return library_name, _sanitize_filename(library_name)
    return upload_filename, _sanitize_filename(upload_filename)


def _build_doc_signature(
    content: bytes,
    max_pages: int | None,
    chunk_size: int,
    chunk_overlap: int,
    embedding_provider: str,
    book_id: str,
) -> str:
    payload = hashlib.sha256(content).hexdigest()
    return f"{payload}:{max_pages}:{chunk_size}:{chunk_overlap}:{embedding_provider}:{book_id}"


def _parse_retry_delay(exc: Exception) -> float | None:
    message = str(exc)
    match = _RETRY_SECONDS.search(message)
    if match:
        return float(match.group(1))
    return None


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "BookChat API is running"}


@app.get("/server/info")
def server_info() -> dict[str, Any]:
    """Public server metadata for the UI (Pinecone, MongoDB). No secrets."""
    return public_vector_store_info()


@app.post("/admin/pinecone/index")
def admin_create_pinecone_index(
    body: CreatePineconeIndexRequest,
    _admin: None = Depends(verify_admin),
) -> dict[str, Any]:
    if not PINECONE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="PINECONE_API_KEY is not set; cannot create Pinecone indexes.",
        )
    cloud = (body.cloud or PINECONE_SERVERLESS_CLOUD).strip()
    region = (body.region or PINECONE_SERVERLESS_REGION).strip()
    dim = body.effective_dimension()
    try:
        from pinecone.exceptions import PineconeApiException

        from app.services.pinecone_store import create_serverless_pinecone_index

        return create_serverless_pinecone_index(
            body.name,
            dim,
            metric=body.metric,
            cloud=cloud,
            region=region,
        )
    except PineconeApiException as exc:
        status = getattr(exc, "status", None)
        msg = str(exc)
        if status == 409 or "already exists" in msg.lower():
            raise HTTPException(
                status_code=409,
                detail=f"A Pinecone index with this name already exists: {body.name!r}",
            ) from exc
        raise HTTPException(status_code=400, detail=msg) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/ingest/status")
def get_ingest_status(filename: str | None = None) -> dict[str, Any]:
    if filename:
        safe_name = _sanitize_filename(filename)
        status = ingest_status.get(safe_name)
        if status is None:
            return {"status": "idle", "filename": safe_name}
        payload = {**status}
        current = payload.get("status")
        if current not in _TERMINAL_INGEST_STATUSES:
            started_at = payload.get("started_at")
            if isinstance(started_at, (int, float)):
                payload["elapsed_seconds"] = round(max(0.0, time.time() - started_at), 1)
        return payload
    return {"active_jobs": list(ingest_status.values())}


@app.post("/ingest/control")
def control_ingest(filename: str, action: str) -> dict[str, Any]:
    safe_name = _sanitize_filename(filename)
    control = ingest_control.setdefault(safe_name, {"paused": False, "stop": False})

    if action == "pause":
        control["paused"] = True
    elif action == "resume":
        control["paused"] = False
    elif action == "stop":
        control["stop"] = True
        control["paused"] = False
    else:
        raise HTTPException(status_code=400, detail="action must be pause, resume, or stop")

    return {"filename": safe_name, "action": action, "control": control}


def _upsert_book_record(
    *,
    book_id: str,
    library_filename: str,
    base_book_id: str,
    pdf_file_id: str,
    pages: int,
    chunks: int,
    chapters: list[str],
    embedding_provider: Provider,
) -> None:
    upsert_book(
        book_id,
        {
            "book_id": book_id,
            "filename": library_filename,
            "base_book_id": base_book_id,
            "pdf_file_id": pdf_file_id,
            "pages": pages,
            "chunks": chunks,
            "chapters": chapters,
            "indexed_at": int(time.time()),
            "embedding_provider": embedding_provider,
        },
    )


@app.post("/books/ingest")
async def ingest_book(
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
    embedding_provider: Provider = "openai",
    max_pages: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    requests_per_second: float | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if max_pages is not None and max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be > 0")
    if max_retries < 0:
        raise HTTPException(status_code=400, detail="max_retries must be >= 0")

    if batch_size is None:
        batch_size = 12
    if requests_per_second is None:
        requests_per_second = 0.0
    if requests_per_second < 0:
        raise HTTPException(status_code=400, detail="requests_per_second must be >= 0")

    library_filename, safe_name = _display_filename_and_safe_name(display_name, file.filename)
    base_book_id = _book_id_from_filename(safe_name)
    book_id = f"{base_book_id}-{embedding_provider}"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    pdf_file_id = await asyncio.to_thread(store_pdf, base_book_id, content)

    pages = extract_pages(content)
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")
    if max_pages is not None:
        pages = pages[:max_pages]

    docs = chunk_pages(book_id, pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not docs:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    from app.config import CHUNK_OVERLAP as DEFAULT_CO
    from app.config import CHUNK_SIZE as DEFAULT_CS

    effective_cs = chunk_size if chunk_size is not None else DEFAULT_CS
    effective_co = chunk_overlap if chunk_overlap is not None else DEFAULT_CO

    doc_signature = _build_doc_signature(
        content=content,
        max_pages=max_pages,
        chunk_size=effective_cs,
        chunk_overlap=effective_co,
        embedding_provider=embedding_provider,
        book_id=book_id,
    )

    progress = load_progress(book_id)
    valid_progress = (
        progress.get("doc_signature") == doc_signature
        and progress.get("total_chunks") == len(docs)
        and progress.get("embedding_provider") == embedding_provider
        and progress.get("book_id") == book_id
    )
    next_index = int(progress.get("next_index", 0)) if valid_progress else 0

    if not valid_progress:
        next_index = 0
        clear_book_index_vectors(book_id, embedding_provider)
    elif next_index > 0 and not index_exists(book_id, embedding_provider):
        next_index = 0
        clear_book_index_vectors(book_id, embedding_provider)

    embedder = get_embedding_model(embedding_provider)
    throttle_delay = (1.0 / requests_per_second) if requests_per_second > 0 else 0.0
    started_at = time.time()

    ingest_status[safe_name] = {
        "filename": safe_name,
        "book_id": book_id,
        "status": "ingesting",
        "started_at": started_at,
        "elapsed_seconds": 0.0,
        "total_chunks": len(docs),
        "processed_chunks": next_index,
        "progress_percent": round((next_index / len(docs)) * 100, 2) if docs else 0.0,
        "embedding_provider": embedding_provider,
        "resumed": bool(valid_progress and next_index > 0),
        "message": f"Preparing embeddings ({embedding_provider})",
    }
    ingest_control[safe_name] = {"paused": False, "stop": False}

    chapters = sorted({str(d.metadata.get("chapter", "Unknown")) for d in docs})

    if (
        len(docs) > 0
        and next_index >= len(docs)
        and bool(progress.get("completed"))
        and index_exists(book_id, embedding_provider)
    ):
        elapsed = round(max(0.0, time.time() - started_at), 1)
        ingest_status[safe_name].update(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "processed_chunks": len(docs),
                "progress_percent": 100.0,
                "message": "Already indexed (same file); skipped.",
            }
        )
        _upsert_book_record(
            book_id=book_id,
            library_filename=library_filename,
            base_book_id=base_book_id,
            pdf_file_id=pdf_file_id,
            pages=len(pages),
            chunks=len(docs),
            chapters=chapters,
            embedding_provider=embedding_provider,
        )
        return {
            "filename": safe_name,
            "book_id": book_id,
            "pages": len(pages),
            "chunks_indexed": len(docs),
            "total_chunks_for_run": len(docs),
            "embedding_provider": embedding_provider,
            "status": "completed",
        }

    try:
        for start_idx in range(next_index, len(docs), batch_size):
            control = ingest_control.get(safe_name, {"paused": False, "stop": False})
            if control.get("stop"):
                elapsed = round(max(0.0, time.time() - started_at), 1)
                ingest_status[safe_name].update(
                    {
                        "status": "stopped",
                        "elapsed_seconds": elapsed,
                        "processed_chunks": start_idx,
                        "progress_percent": round((start_idx / len(docs)) * 100, 2),
                        "message": "Indexing stopped by user",
                    }
                )
                save_progress(
                    book_id,
                    {
                        "doc_signature": doc_signature,
                        "total_chunks": len(docs),
                        "next_index": start_idx,
                        "embedding_provider": embedding_provider,
                        "completed": False,
                    },
                )
                break

            while control.get("paused") and not control.get("stop"):
                ingest_status[safe_name].update(
                    {
                        "status": "paused",
                        "elapsed_seconds": round(max(0.0, time.time() - started_at), 1),
                        "processed_chunks": start_idx,
                        "progress_percent": round((start_idx / len(docs)) * 100, 2),
                        "message": "Indexing paused by user",
                    }
                )
                await asyncio.sleep(0.5)
                control = ingest_control.get(safe_name, {"paused": False, "stop": False})

            if control.get("stop"):
                elapsed = round(max(0.0, time.time() - started_at), 1)
                ingest_status[safe_name].update(
                    {
                        "status": "stopped",
                        "elapsed_seconds": elapsed,
                        "processed_chunks": start_idx,
                        "progress_percent": round((start_idx / len(docs)) * 100, 2),
                        "message": "Indexing stopped by user",
                    }
                )
                save_progress(
                    book_id,
                    {
                        "doc_signature": doc_signature,
                        "total_chunks": len(docs),
                        "next_index": start_idx,
                        "embedding_provider": embedding_provider,
                        "completed": False,
                    },
                )
                break

            end_idx = min(start_idx + batch_size, len(docs))
            batch_docs = docs[start_idx:end_idx]
            texts = [d.page_content for d in batch_docs]
            metadatas = [dict(d.metadata) for d in batch_docs]

            st = ingest_status[safe_name]
            st.update(
                {
                    "status": "ingesting",
                    "elapsed_seconds": round(max(0.0, time.time() - started_at), 1),
                    "processed_chunks": start_idx,
                    "progress_percent": round((start_idx / len(docs)) * 100, 2),
                    "message": f"Embedding chunks {start_idx + 1}-{end_idx}/{len(docs)}",
                }
            )

            retries = 0
            embeddings: list[list[float]] = []
            while True:
                try:
                    embeddings = await asyncio.to_thread(embedder.embed_documents, texts)
                    break
                except Exception as exc:
                    message = str(exc)
                    is_rate_limited = (
                        "429" in message
                        or "RESOURCE_EXHAUSTED" in message
                        or "rate_limit" in message.lower()
                        or "rate limit" in message.lower()
                    )
                    is_transient = (
                        "timed out" in message.lower()
                        or "timeout" in message.lower()
                        or "connection" in message.lower()
                    )
                    if (not is_rate_limited and not is_transient) or retries >= max_retries:
                        raise

                    retry_after = _parse_retry_delay(exc)
                    if retry_after is None:
                        retry_after = min(2**retries, 120) + random.uniform(0, 0.5)
                    st.update(
                        {
                            "status": "rate-limited-wait"
                            if is_rate_limited
                            else "temporary-wait",
                            "elapsed_seconds": round(
                                max(0.0, time.time() - started_at), 1
                            ),
                            "processed_chunks": start_idx,
                            "progress_percent": round((start_idx / len(docs)) * 100, 2),
                            "retry_in_seconds": round(retry_after, 1),
                            "message": (
                                f"Rate limited; waiting {retry_after:.1f}s before retry"
                                if is_rate_limited
                                else f"Temporary error; waiting {retry_after:.1f}s before retry"
                            ),
                        }
                    )
                    await asyncio.sleep(retry_after)
                    retries += 1

            from app.services.pinecone_store import pinecone_namespace, upsert_embedding_batch

            ns = pinecone_namespace(book_id, embedding_provider)
            await asyncio.to_thread(
                upsert_embedding_batch,
                embedding_provider,
                ns,
                start_idx,
                texts,
                embeddings,
                metadatas,
            )

            next_index = end_idx
            save_progress(
                book_id,
                {
                    "doc_signature": doc_signature,
                    "total_chunks": len(docs),
                    "next_index": next_index,
                    "embedding_provider": embedding_provider,
                    "completed": next_index >= len(docs),
                },
            )

            if next_index < len(docs) and throttle_delay > 0:
                await asyncio.sleep(throttle_delay)

    except Exception as exc:
        elapsed = round(max(0.0, time.time() - started_at), 1)
        ingest_status[safe_name].update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "processed_chunks": next_index,
                "progress_percent": round((next_index / len(docs)) * 100, 2)
                if docs
                else 0.0,
                "message": str(exc),
            }
        )
        raise HTTPException(
            status_code=400,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    if ingest_status[safe_name].get("status") not in _TERMINAL_INGEST_STATUSES:
        elapsed = round(max(0.0, time.time() - started_at), 1)
        ingest_status[safe_name].update(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "processed_chunks": next_index,
                "progress_percent": 100.0,
                "retry_in_seconds": 0,
                "message": "Ingestion completed",
            }
        )
        save_progress(
            book_id,
            {
                "doc_signature": doc_signature,
                "total_chunks": len(docs),
                "next_index": next_index,
                "embedding_provider": embedding_provider,
                "completed": True,
            },
        )
        _upsert_book_record(
            book_id=book_id,
            library_filename=library_filename,
            base_book_id=base_book_id,
            pdf_file_id=pdf_file_id,
            pages=len(pages),
            chunks=len(docs),
            chapters=chapters,
            embedding_provider=embedding_provider,
        )

    final = ingest_status[safe_name]
    return {
        "filename": safe_name,
        "book_id": book_id,
        "pages": len(pages),
        "chunks_indexed": final.get("processed_chunks", next_index),
        "total_chunks_for_run": len(docs),
        "embedding_provider": embedding_provider,
        "status": final.get("status"),
    }


@app.get("/books")
def get_books() -> dict[str, Any]:
    return {"books": list(list_books().values())}


@app.delete("/books/{book_id}")
def delete_book(book_id: str) -> dict[str, Any]:
    entry = get_book(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Book not found in library.")

    embedding_provider: Provider = entry.get("embedding_provider") or "openai"
    filename_for_ingest = entry.get("filename") or ""
    safe_name = _sanitize_filename(str(filename_for_ingest) or "upload.pdf")
    pdf_file_id = entry.get("pdf_file_id")

    st = ingest_status.get(safe_name)
    if (
        st
        and st.get("book_id") == book_id
        and st.get("status") not in _TERMINAL_INGEST_STATUSES
    ):
        control = ingest_control.setdefault(safe_name, {"paused": False, "stop": False})
        control["stop"] = True
        control["paused"] = False

    mongo_delete_book(book_id)
    clear_book_index_vectors(book_id, embedding_provider)
    removed_progress = delete_progress(book_id)

    removed_pdf = False
    if pdf_file_id and count_books_with_pdf_file_id(pdf_file_id) == 0:
        removed_pdf = delete_pdf_file(str(pdf_file_id))
        ingest_status.pop(safe_name, None)
        ingest_control.pop(safe_name, None)

    return {
        "book_id": book_id,
        "removed_pdf": removed_pdf,
        "removed_progress": removed_progress,
        "embedding_provider": embedding_provider,
    }


@app.get("/admin/books/{book_id}/chunks")
def admin_list_book_chunks(
    book_id: str,
    embedding_provider: Provider = "openai",
    offset: int = 0,
    limit: int = 50,
    _admin: None = Depends(verify_admin),
) -> dict[str, Any]:
    if not get_book(book_id):
        raise HTTPException(status_code=404, detail="Book not found in library.")
    try:
        chunks, total = list_book_documents_page(
            book_id,
            embedding_provider=embedding_provider,
            offset=offset,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "book_id": book_id,
        "embedding_provider": embedding_provider,
        "total": total,
        "offset": offset,
        "limit": limit,
        "returned": len(chunks),
        "chunks": chunks,
    }


@app.get("/books/{book_id}/pdf")
def get_book_pdf(book_id: str) -> Response:
    entry = get_book(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Book not found in library.")
    pdf_file_id = entry.get("pdf_file_id")
    if not pdf_file_id:
        raise HTTPException(status_code=404, detail="No PDF stored for this book.")
    content = read_pdf_bytes(str(pdf_file_id))
    if not content:
        raise HTTPException(status_code=404, detail="PDF file is missing in storage.")
    download_name = str(entry.get("filename") or f"{book_id}.pdf")
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{download_name}"',
        },
    )


@app.post("/tts")
def text_to_speech(body: TtsRequest) -> Response:
    try:
        wav = synthesize_openai_tts_wav(body.text, voice_name=body.voice)
        return Response(content=wav, media_type="audio/wav")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc


@app.post("/query/classify")
def classify(body: ClassifyRequest) -> dict[str, str]:
    try:
        label = classify_query(body.question, chat_provider=body.chat_provider)
        return {"question": body.question, "classification": label}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc


@app.get("/books/{book_id}/summary")
def get_book_summary(
    book_id: str, embedding_provider: Provider = "openai", chat_provider: Provider = "openai"
) -> dict[str, str]:
    try:
        store = load_book_store(book_id, embedding_provider=embedding_provider)
        return {"book_id": book_id, "summary": summarize_book(store, chat_provider=chat_provider)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary failed: {exc}") from exc


@app.get("/books/{book_id}/chapters/{chapter}/summary")
def get_chapter_summary(
    book_id: str,
    chapter: str,
    embedding_provider: Provider = "openai",
    chat_provider: Provider = "openai",
) -> dict[str, str]:
    try:
        store = load_book_store(book_id, embedding_provider=embedding_provider)
        return {
            "book_id": book_id,
            "chapter": chapter,
            "summary": summarize_chapter(store, chapter, chat_provider=chat_provider),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chapter summary failed: {exc}") from exc


@app.post("/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    try:
        intent = classify_query(body.question, chat_provider=body.chat_provider)
        store = load_book_store(body.book_id, embedding_provider=body.embedding_provider)

        if intent == "book_summary":
            answer = summarize_book(store, chat_provider=body.chat_provider)
            sources = []
        elif intent == "chapter_summary":
            chapter_match = re.search(
                r"\bchapter\s+([0-9]+|[ivxlcdm]+)\b", body.question, re.IGNORECASE
            )
            if chapter_match:
                raw = chapter_match.group(1)
                chapter_name = f"Chapter {raw.upper() if raw.isalpha() else raw}"
                answer = summarize_chapter(store, chapter_name, chat_provider=body.chat_provider)
                sources = []
            else:
                answer = "Please specify the chapter number/name in your question."
                sources = []
        else:
            history_tuples = [(t.role, t.content) for t in body.history]
            history_block = format_history_for_prompt(history_tuples)
            docs = gather_documents_for_rag(store, intent, body.question, body.k)
            context = format_context_blocks(docs)
            llm = get_chat_model(body.chat_provider, temperature=0.1)
            prompt = build_full_rag_prompt(body.question, context, history_block)
            answer = llm.invoke(prompt).content.strip()
            sources = [
                {
                    "page": d.metadata.get("page"),
                    "chapter": d.metadata.get("chapter"),
                    "preview": d.page_content[:160],
                }
                for d in docs
            ]

        return {
            "book_id": body.book_id,
            "classification": intent,
            "answer": answer,
            "sources": sources,
            "embedding_provider": body.embedding_provider,
            "chat_provider": body.chat_provider,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
