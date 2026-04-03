import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from langchain_community.vectorstores import FAISS

from app.config import (
    ADMIN_API_TOKEN,
    BOOKS_DIR,
    PINECONE_API_KEY,
    PINECONE_SERVERLESS_CLOUD,
    PINECONE_SERVERLESS_REGION,
    PROGRESS_DIR,
    public_vector_store_info,
    use_pinecone_vector_store,
)
from app.models import ChatRequest, ClassifyRequest, CreatePineconeIndexRequest, TtsRequest
from app.services.classifier_service import classify_query
from app.services.document_service import chunk_pages, extract_pages
from app.services.manifest_service import get_book, list_books, pop_book, upsert_book
from app.services.provider_service import Provider, get_chat_model, get_embedding_model
from app.services.rag_chat_service import (
    build_full_rag_prompt,
    format_context_blocks,
    format_history_for_prompt,
    gather_documents_for_rag,
)
from app.services.summary_service import summarize_book, summarize_chapter
from app.services.tts_service import synthesize_gemini_tts_wav
from app.services.vector_service import (
    clear_book_index_vectors,
    faiss_index_dir,
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


def verify_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    if not ADMIN_API_TOKEN:
        return
    if x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Admin routes require header X-Admin-Token (set ADMIN_API_TOKEN on server).",
        )


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base) or "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _book_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    safe = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return safe or f"book-{int(time.time())}"


def _manifest_filename_and_safe_name(display_name: str | None, upload_filename: str) -> tuple[str, str]:
    """(library label + download hint, ingest_status key / on-disk PDF basename)."""
    if display_name and display_name.strip():
        label = Path(display_name.strip()).name.strip()
        if label:
            if len(label) > 240:
                raise HTTPException(
                    status_code=400,
                    detail="display_name is too long (max 240 characters).",
                )
            manifest = label if label.lower().endswith(".pdf") else f"{label}.pdf"
            return manifest, _sanitize_filename(manifest)
    return upload_filename, _sanitize_filename(upload_filename)


def _load_progress(progress_file: Path) -> dict[str, Any]:
    if not progress_file.exists():
        return {}
    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_progress(progress_file: Path, payload: dict[str, Any]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(payload), encoding="utf-8")


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
    """Public server metadata for the UI (vector DB mode, Pinecone index names). No secrets."""
    return public_vector_store_info()


@app.post("/admin/pinecone/index")
def admin_create_pinecone_index(
    body: CreatePineconeIndexRequest,
    _admin: None = Depends(verify_admin),
) -> dict[str, Any]:
    """Create a serverless Pinecone index (control plane). Requires PINECONE_API_KEY and optional X-Admin-Token."""
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


@app.post("/books/ingest")
async def ingest_book(
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
    embedding_provider: Provider = "ollama",
    max_pages: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    requests_per_second: float | None = None,
    persist_every_batches: int = 20,
    max_retries: int = 3,
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if max_pages is not None and max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be > 0")
    if persist_every_batches <= 0:
        raise HTTPException(status_code=400, detail="persist_every_batches must be > 0")
    if max_retries < 0:
        raise HTTPException(status_code=400, detail="max_retries must be >= 0")

    # Same defaults as V1 ingest when query params are omitted (no UI tuning).
    if batch_size is None:
        batch_size = 12
    if requests_per_second is None:
        requests_per_second = 0.0
    if requests_per_second < 0:
        raise HTTPException(status_code=400, detail="requests_per_second must be >= 0")

    manifest_filename, safe_name = _manifest_filename_and_safe_name(display_name, file.filename)
    base_book_id = _book_id_from_filename(safe_name)
    book_id = f"{base_book_id}-{embedding_provider}"
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = BOOKS_DIR / f"{base_book_id}.pdf"
    progress_file = PROGRESS_DIR / f"{book_id}.progress.json"
    index_dir = faiss_index_dir(book_id, embedding_provider)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    path.write_bytes(content)

    pages = extract_pages(str(path))
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")
    if max_pages is not None:
        pages = pages[:max_pages]

    cs = chunk_size
    co = chunk_overlap
    docs = chunk_pages(book_id, pages, chunk_size=cs, chunk_overlap=co)
    if not docs:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    from app.config import CHUNK_OVERLAP as DEFAULT_CO
    from app.config import CHUNK_SIZE as DEFAULT_CS

    effective_cs = cs if cs is not None else DEFAULT_CS
    effective_co = co if co is not None else DEFAULT_CO

    doc_signature = _build_doc_signature(
        content=content,
        max_pages=max_pages,
        chunk_size=effective_cs,
        chunk_overlap=effective_co,
        embedding_provider=embedding_provider,
        book_id=book_id,
    )

    progress = _load_progress(progress_file)
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
    batches_since_persist = 0

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

    store: FAISS | None = None
    if next_index > 0 and not use_pinecone_vector_store():
        try:
            store = await asyncio.to_thread(load_book_store, book_id, embedding_provider)
        except Exception:
            next_index = 0
            clear_book_index_vectors(book_id, embedding_provider)
            ingest_status[safe_name]["resumed"] = False
            ingest_status[safe_name]["processed_chunks"] = 0
            ingest_status[safe_name]["progress_percent"] = 0.0

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
        upsert_book(
            book_id,
            {
                "book_id": book_id,
                "filename": manifest_filename,
                "pdf_path": str(path),
                "pages": len(pages),
                "chunks": len(docs),
                "chapters": chapters,
                "indexed_at": int(time.time()),
                "embedding_provider": embedding_provider,
            },
        )
        return {
            "filename": safe_name,
            "book_id": book_id,
            "pages": len(pages),
            "chunks_indexed": len(docs),
            "total_chunks_for_run": len(docs),
            "embedding_provider": embedding_provider,
            "progress_file": str(progress_file),
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
                _save_progress(
                    progress_file,
                    {
                        "doc_signature": doc_signature,
                        "book_id": book_id,
                        "total_chunks": len(docs),
                        "next_index": start_idx,
                        "embedding_provider": embedding_provider,
                        "completed": False,
                    },
                )
                if store is not None and not use_pinecone_vector_store():
                    await asyncio.to_thread(store.save_local, str(index_dir))
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
                _save_progress(
                    progress_file,
                    {
                        "doc_signature": doc_signature,
                        "book_id": book_id,
                        "total_chunks": len(docs),
                        "next_index": start_idx,
                        "embedding_provider": embedding_provider,
                        "completed": False,
                    },
                )
                if store is not None and not use_pinecone_vector_store():
                    await asyncio.to_thread(store.save_local, str(index_dir))
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
                    is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
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

            if use_pinecone_vector_store():
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
            else:
                pairs = list(zip(texts, embeddings))
                if store is None:
                    store = FAISS.from_embeddings(
                        pairs,
                        embedder,
                        metadatas=metadatas,
                    )
                else:
                    store.add_embeddings(pairs, metadatas=metadatas)

            next_index = end_idx

            _save_progress(
                progress_file,
                {
                    "doc_signature": doc_signature,
                    "book_id": book_id,
                    "total_chunks": len(docs),
                    "next_index": next_index,
                    "embedding_provider": embedding_provider,
                    "completed": next_index >= len(docs),
                },
            )
            if not use_pinecone_vector_store():
                batches_since_persist += 1
                if store is not None and batches_since_persist >= persist_every_batches:
                    await asyncio.to_thread(store.save_local, str(index_dir))
                    batches_since_persist = 0

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
        if store is not None and not use_pinecone_vector_store():
            await asyncio.to_thread(store.save_local, str(index_dir))
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
        _save_progress(
            progress_file,
            {
                "doc_signature": doc_signature,
                "book_id": book_id,
                "total_chunks": len(docs),
                "next_index": next_index,
                "embedding_provider": embedding_provider,
                "completed": True,
            },
        )
        upsert_book(
            book_id,
            {
                "book_id": book_id,
                "filename": manifest_filename,
                "pdf_path": str(path),
                "pages": len(pages),
                "chunks": len(docs),
                "chapters": chapters,
                "indexed_at": int(time.time()),
                "embedding_provider": embedding_provider,
            },
        )

    final = ingest_status[safe_name]
    return {
        "filename": safe_name,
        "book_id": book_id,
        "pages": len(pages),
        "chunks_indexed": final.get("processed_chunks", next_index),
        "total_chunks_for_run": len(docs),
        "embedding_provider": embedding_provider,
        "progress_file": str(progress_file),
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

    embedding_provider: Provider = entry.get("embedding_provider") or "ollama"
    pdf_path_raw = entry.get("pdf_path")
    filename_for_ingest = entry.get("filename") or ""
    safe_name = _sanitize_filename(str(filename_for_ingest) or "upload.pdf")

    other_books_share_pdf = False
    if pdf_path_raw:
        for bid, other in list_books().items():
            if bid == book_id:
                continue
            if other.get("pdf_path") == pdf_path_raw:
                other_books_share_pdf = True
                break

    st = ingest_status.get(safe_name)
    if (
        st
        and st.get("book_id") == book_id
        and st.get("status") not in _TERMINAL_INGEST_STATUSES
    ):
        control = ingest_control.setdefault(safe_name, {"paused": False, "stop": False})
        control["stop"] = True
        control["paused"] = False

    pop_book(book_id)
    clear_book_index_vectors(book_id, embedding_provider)

    progress_file = PROGRESS_DIR / f"{book_id}.progress.json"
    removed_progress = False
    if progress_file.exists():
        progress_file.unlink()
        removed_progress = True

    removed_pdf = False
    if pdf_path_raw and not other_books_share_pdf:
        path = Path(str(pdf_path_raw))
        if path.is_file():
            path.unlink()
            removed_pdf = True
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
    embedding_provider: Provider = "ollama",
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
def get_book_pdf(book_id: str) -> FileResponse:
    entry = get_book(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Book not found in library.")
    raw_path = entry.get("pdf_path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="No PDF path recorded for this book.")
    path = Path(str(raw_path))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="PDF file is missing on disk.")
    download_name = str(entry.get("filename") or path.name)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=download_name,
        content_disposition_type="inline",
    )


@app.post("/tts")
def text_to_speech(body: TtsRequest) -> Response:
    """Gemini native TTS; returns WAV for HTMLAudioElement playback."""
    try:
        wav = synthesize_gemini_tts_wav(body.text, voice_name=body.voice)
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
    book_id: str, embedding_provider: Provider = "ollama", chat_provider: Provider = "ollama"
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
    embedding_provider: Provider = "ollama",
    chat_provider: Provider = "ollama",
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
