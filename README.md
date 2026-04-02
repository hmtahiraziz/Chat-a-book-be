# BookChat API (LangChain backend)

FastAPI + LangChain for PDF ingestion, retrieval, and chat. Vectors can live on **local FAISS** or **Pinecone** (recommended for serverless / cloud).

## Features

- PDF ingestion and chunking (`page`, `chapter` metadata)
- Vector indexing per book: **FAISS** (local `data/indices/`) or **Pinecone** (hosted, one namespace per book + embedding provider)
- Query classification and intent-aware chat (MMR retrieval)
- Book / chapter summaries
- Optional Gemini native TTS (`POST /tts`)

## Vector store: FAISS vs Pinecone

| Mode | When to use |
|------|----------------|
| **FAISS** (default) | Local dev, no Pinecone account. Indexes are files under `data/indices/`. |
| **Pinecone** | Production, containers without persistent disk, or multi-instance APIs. |

**Enable Pinecone**

1. In [Pinecone](https://www.pinecone.io/), create **serverless** indexes with **cosine** metric. Defaults used in `.env.example` match this project’s embedders:

   | Pinecone index name | Dimension | Used for |
   |---------------------|-----------|----------|
   | `bookchat-ollama-nomic-768` | **768** | `OLLAMA_EMBED_MODEL=nomic-embed-text` |
   | `bookchat-google-gemini-3072` | **3072** | `GEMINI_EMBED_MODEL=gemini-embedding-001` (Gemini API default embedding size) |

2. Set in `.env` (see `.env.example` for the full template):

   ```env
   PINECONE_API_KEY=your-key
   PINECONE_INDEX=bookchat-ollama-nomic-768
   PINECONE_INDEX_OLLAMA=bookchat-ollama-nomic-768
   PINECONE_INDEX_GOOGLE=bookchat-google-gemini-3072
   ```

   If `VECTOR_STORE` is omitted, the app uses Pinecone when both `PINECONE_API_KEY` and `PINECONE_INDEX` are set; otherwise it uses FAISS.

3. Optional: `VECTOR_STORE=pinecone` or `VECTOR_STORE=faiss` to force a mode.

Re-ingest PDFs after switching from FAISS to Pinecone (indexes are not migrated automatically).

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **pip** (usually bundled with Python)
- **Ollama** ([download](https://ollama.com/download)) if you use local models
- **Google AI API key** (optional) for Gemini embeddings, chat, and TTS — set `GEMINI_API_KEY` in `.env`

---

## Setup on macOS

1. Open **Terminal** and go to this folder:

   ```bash
   cd path/to/Book-Rag/ai-book-chatbot-v2
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Your prompt should show `(venv)`. To leave the environment later: `deactivate`.

3. Upgrade pip and install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create your environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with a text editor. Typical values:

   - `OLLAMA_BASE_URL` — default `http://127.0.0.1:11434`
   - `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL` — match models you pulled
   - `GEMINI_API_KEY` — if you use Google for embeddings/chat/TTS

5. **Ollama** (if using local models): start the Ollama app, then:

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

   (Use the model names configured in `.env`.)

6. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

   If `uvicorn` is not on your PATH:

   ```bash
   python -m uvicorn app.main:app --reload --port 8001
   ```

7. Check **http://127.0.0.1:8001/docs** for the interactive API docs.

---

## Setup on Windows

1. Open **Command Prompt** or **PowerShell** and go to this folder:

   ```cmd
   cd path\to\Book-Rag\ai-book-chatbot-v2
   ```

2. Create and activate a virtual environment:

   ```cmd
   py -3 -m venv venv
   venv\Scripts\activate
   ```

   If `py` is not available, try `python -m venv venv` instead.

   Your prompt should show `(venv)`. To leave later: `deactivate`.

3. Upgrade pip and install dependencies:

   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create your environment file:

   ```cmd
   copy .env.example .env
   ```

   Edit `.env` in Notepad, VS Code, or another editor (same variables as in the macOS section).

5. **Ollama** (if using local models): install from [ollama.com/download](https://ollama.com/download), start Ollama, then in a **new** terminal:

   ```cmd
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

6. Start the API (with `venv` activated):

   ```cmd
   uvicorn app.main:app --reload --port 8001
   ```

   Or:

   ```cmd
   python -m uvicorn app.main:app --reload --port 8001
   ```

7. Open **http://127.0.0.1:8001/docs** in your browser.

### Windows notes

- If **execution policy** blocks scripts in PowerShell, run once (as your user):

  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

- If **Python was not added to PATH**, reinstall from [python.org](https://www.python.org/downloads/) and enable **Add python.exe to PATH**, or use the **py** launcher as shown above.

---

## Main endpoints

- `POST /books/ingest` — multipart PDF upload
- `GET /books` — library list
- `POST /chat` — RAG chat with history
- `POST /tts` — Gemini speech (WAV), requires `GEMINI_API_KEY`
- `POST /query/classify`
- `GET /books/{book_id}/summary`

---

## Production hint

For production, run without `--reload`, behind a reverse proxy, and set secrets only via environment variables — never commit `.env`.
