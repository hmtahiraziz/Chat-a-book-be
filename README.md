# BookChat API (LangChain backend)

FastAPI + LangChain for PDF ingestion, retrieval, and chat.

## Storage

| Layer | Technology | Contents |
|-------|------------|----------|
| **Library** | **MongoDB** (+ GridFS) | Book metadata, PDF binaries, ingest resume state, chat sessions |
| **Vectors** | **Pinecone** | Chunk embeddings (one namespace per book + embedding provider) |

Both `MONGODB_URI` and `PINECONE_API_KEY` are **required** at startup.

### MongoDB

Set `MONGODB_URI` (and optional `MONGODB_DB_NAME`, collection names) in `.env`. PDFs are stored in GridFS so deploys do not rely on local `data/books/`.

### Pinecone

1. Set in `.env` (see `.env.example`), then create **serverless** indexes with **cosine** metric:

   ```bash
   cd ai-book-chatbot-v2
   python scripts/create_pinecone_indexes.py
   ```

   Or create one preset: `python scripts/create_pinecone_indexes.py --preset openai`

   | Pinecone index name | Dimension | Used for |
   |---------------------|-----------|----------|
   | `bookchat-openai-3072` | **3072** | `OPENAI_EMBED_MODEL=text-embedding-3-large` |

2. Confirm `.env` matches your index names:

   ```env
   PINECONE_API_KEY=your-key
   PINECONE_INDEX=bookchat-openai-3072
   PINECONE_INDEX_OPENAI=bookchat-openai-3072
   OPENAI_API_KEY=your-key
   ```

## Features

- PDF ingestion and chunking (`page`, `chapter` metadata)
- Pinecone vector indexing per book (namespace per book + embedding provider)
- Query classification and intent-aware chat (MMR retrieval)
- Book / chapter summaries
- Optional OpenAI speech (`POST /tts`)
- **JWT authentication** — register, login, and subscription plans (`starter` / `pro`)
- Mock subscription checkout (no payment gateway)

## Authentication & subscriptions

| Endpoint | Description |
|----------|-------------|
| `POST /auth/register` | Create account |
| `POST /auth/login` | Returns JWT bearer token |
| `GET /auth/me` | Current user (requires `Authorization: Bearer …`) |
| `GET /auth/plans` | List plans (public) |
| `POST /auth/subscribe` | Activate plan (mock — no payment) |

Workspace routes (`/books/*`, `/chat/*`, ingest, TTS) require a signed-in user with an **active** subscription. Admins bypass plan limits.

**Seed the default admin** (after MongoDB is configured):

```bash
source venv/bin/activate
pip install -r requirements.txt   # includes python-jose, passlib
python scripts/seed_admin.py
```

Defaults: `admin@bookchat.local` / `Admin123!` (override with `SEED_ADMIN_EMAIL`, `SEED_ADMIN_PASSWORD`).

**Seed the demo account** (landing page “Try demo”, no signup):

```bash
python scripts/seed_demo.py
```

Uses `demo@bookchat.local` with Pro subscription. `POST /auth/demo-login` issues a JWT with no password. The demo book is auto-detected (filename containing “harry” and “potter”) or set `DEMO_BOOK_ID` in `.env`.

Set `JWT_SECRET` in `.env` for production.

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **MongoDB** (local, Atlas, or other) — `MONGODB_URI` in `.env`
- **Pinecone** account and indexes — see Storage above
- **OpenAI API key** for embeddings, chat, and TTS — set `OPENAI_API_KEY` in `.env`

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

   Edit `.env` with a text editor. Set `OPENAI_API_KEY`, `PINECONE_API_KEY`, `MONGODB_URI`, and Pinecone index names (see Storage above).

5. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

   If `uvicorn` is not on your PATH:

   ```bash
   python -m uvicorn app.main:app --reload --port 8001
   ```

6. Check **http://127.0.0.1:8001/docs** for the interactive API docs.

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

5. Start the API (with `venv` activated):

   ```cmd
   uvicorn app.main:app --reload --port 8001
   ```

   Or:

   ```cmd
   python -m uvicorn app.main:app --reload --port 8001
   ```

6. Open **http://127.0.0.1:8001/docs** in your browser.

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
- `GET /chat/sessions` — list chat threads (`X-Client-Id` header)
- `POST /chat/sessions` — create thread
- `PUT /chat/sessions/{session_id}` — save thread (messages, title)
- `DELETE /chat/sessions/{session_id}` — delete thread
- `POST /chat` — RAG chat with history
- `POST /tts` — OpenAI speech (WAV), requires `OPENAI_API_KEY`
- `POST /query/classify`
- `GET /books/{book_id}/summary`

---

## Production hint

For production, run without `--reload`, behind a reverse proxy, and set secrets only via environment variables — never commit `.env`.
