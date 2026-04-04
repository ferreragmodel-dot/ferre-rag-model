# Gianfranco Ferré Archive RAG System

This project builds a Retrieval-Augmented Generation (RAG) system for the Gianfranco Ferré archive, using a vector database and a Large Language Model (LLM). The system ingests Ferré PDFs, chunks them, generates embeddings, stores them in ChromaDB, and enables search and chat over the archive through a web interface.

<img src="images/ferre.png" width="800">

## Project Overview

**Workflow:**
1. Chunk Ferré archive PDFs into text segments
2. Generate embeddings for text chunks (Vertex AI)
3. Generate embeddings for fashion show images by season
4. Load chunks, embeddings, and images into ChromaDB (separate collections per season)
5. Query and chat with the archive using LLM (text and images)
6. Agent mode: LLM automatically selects the right retrieval strategy and generates a grounded answer

**Architecture:**
- `src/vector-db/` — Python CLI pipeline for chunking, embedding, and loading data
- `src/api-service/` — FastAPI backend exposing LLM, RAG, and Agent chat endpoints
- `src/frontend-react/` — Next.js frontend with chat interface
- ChromaDB vector database
- Vertex AI / Gemini 2.0 Flash for LLM and embeddings
- Docker for containerized development

## Prerequisites
- Docker installed
- Clone this repository
- Ferré archive PDFs (add to `src/vector-db/input-datasets/ferre-notes-lessons/`)
- Ferré fashion show images (add to `src/vector-db/input-datasets/ferre-designs/Dataset DataShack 2026/[SEASON]/`)
- GCP service account with Vertex AI access

## Secrets & Environment Setup
- Obtain a GCP service account key (JSON) and place it in `secrets/llm-service-account.json` (outside the repo)
- Set `GCP_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` in each `docker-shell.sh`
- **Never commit service account files to the repo**

## Setup GCP Service Account
1. Go to [GCP Console](https://console.cloud.google.com/home/dashboard)
2. Create a service account with "Vertex AI User" and "Storage Admin" roles
3. Download the JSON key and place it in `secrets/llm-service-account.json` (next to the repo, not inside it)
4. In each `docker-shell.sh`, set `GCP_PROJECT` to your project ID and `GOOGLE_APPLICATION_CREDENTIALS` to `/secrets/llm-service-account.json`

## Folder Structure
```
Desktop/  (or wherever you cloned the repo)
├── secrets/                        # Outside the repo — never committed
│   └── llm-service-account.json
└── ferre-rag-model/
    └──Dataset Datashack 2026/
           └──ALTA MODA 1986-87 FW/Womenswear/Fashion show photos
           └──ALTA MODA 1987 SS/Womenswear/Fashion show photos
           └──ALTA MODA 1987-88 FW/Womenswear/Fashion show photos
           └──ALTA MODA 1988 SS/Womenswear/Fashion show photos
           └──ALTA MODA 1988-89 FW/Womenswear/Fashion show photos
           └──ALTA MODA 1989 SS/Womenswear/Fashion show photos
    └── src/
    ├── vector-db/                  # Offline pipeline: chunk, embed, load
    │   ├── outputs/                   # Chunked and embedded data
    │   ├── metadata/                  # Archive metadata JSON files
    │   ├── cli.py
    │   ├── agent_tools.py
    │   ├── semantic_splitter.py
    │   ├── docker-shell.sh
    │   ├── docker-compose.yml
    │   └── Dockerfile
    ├── api-service/                # FastAPI backend
    │   ├── api/
    │   │   ├── routers/
    │   │   │   ├── llm_chat.py         # Plain LLM chat
    │   │   │   ├── llm_rag_chat.py     # RAG chat
    │   │   │   └── llm_agent_chat.py   # Agentic RAG chat
    │   │   ├── utils/
    │   │   │   ├── llm_utils.py
    │   │   │   ├── llm_rag_utils.py
    │   │   │   ├── llm_agent_utils.py
    │   │   │   ├── agent_tools.py
    │   │   │   └── chat_utils.py
    │   │   └── service.py
    │   ├── docker-shell.sh
    │   ├── docker-compose.yml
    │   └── Dockerfile
    └── frontend-react/             # Next.js frontend
        ├── src/
        │   ├── app/
        │   │   └── chat/           # Main chat interface
        │   ├── components/
        │   │   ├── chat/           # Chat UI components
        │   │   └── layout/         # Header, Footer
        │   └── services/
        │       └── DataService.js  # API client
        ├── docker-shell.sh
        └── package.json
```

---

## 1. Vector DB Setup

Navigate to the vector-db directory:
```bash
cd ferre-rag-model/src/vector-db
```

Build and run the container:
```bash
sh docker-shell.sh
```

This will build the Docker image, start ChromaDB, and run the chunk/embed/load pipeline automatically.

### How conditional startup works

When `docker-shell.sh` launches the container, `docker-entrypoint.sh` runs automatically and skips steps that have already been completed. Each step is checked independently:

| Step | Skip condition | What is checked |
|---|---|---|
| **Chunk** | `outputs/chunks*.jsonl` already exists | Presence of any chunk file in `outputs/`. If nothing is present, `python cli.py --chunk` is run |
| **Embed (text)** | `outputs/embeddings*.jsonl` already exists | Presence of any text embedding file in `outputs/`. If nothing is present, `python cli.py --embed` is run |
| **Load text** | At least one non-image collection exists in ChromaDB | Any collection whose name does **not** start with `images-` counts |
| **Load images** | At least one image collection exists in ChromaDB | Any collection whose name starts with `images-` counts |

> **Note:** the checks are intentionally generic — they verify whether *any* data of that type is present, not whether it matches a specific `--chunk_type`. If you want to force a specific chunk type, re-run the relevant steps manually once the container is running:
> ```bash
> python cli.py --chunk --chunk_type char-split
> python cli.py --embed --chunk_type char-split
> python cli.py --load --chunk_type char-split
> ```

### Chunking parameters

All chunking methods measure `chunk_size` and `chunk_overlap` in **tokens** (not characters), using the `cl100k_base` tokenizer (GPT-4) as a close approximation for `text-embedding-004`.

The defaults are defined at the top of [cli.py](src/vector-db/cli.py):

```python
CHUNK_SIZE_TOKENS = 350       # tokens per chunk
CHUNK_OVERLAP_TOKENS = 50     # tokens of overlap between consecutive chunks (~14%)
```

**Target chunk size guide:**

| Range (tokens) | Approx. characters | When to use |
|---|---|---|
| 128–256 | ~500–1000 | High-precision retrieval, short factual sentences |
| **256–512** | **~1000–2000** | **Recommended balance for RAG — good for Ferré archive texts** |
| 512–1024 | ~2000–4000 | Long structured documents, more context per retrieved chunk |

### Embedding parameters

```python
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
```

| Model | Best for |
|---|---|
| `text-embedding-004` | English text (default) |
| `text-multilingual-embedding-002` | Italian or mixed-language text — recommended if archive PDFs are in Italian |

**`EMBEDDING_DIMENSION`** — both models support Matryoshka representations:

| Dimension | Quality | Notes |
|---|---|---|
| **256** | good | Current default |
| **512** | better | Recommended upgrade |
| 768 | best | Maximum quality |

**Image embeddings** use `multimodalembedding@001` (Vertex AI) with dimension `1408` — independent from the text embedding constants.

### Manual CLI Usage

**Text Processing:**
```bash
python cli.py --chunk --chunk_type recursive-split
python cli.py --embed --chunk_type recursive-split
python cli.py --load --chunk_type recursive-split
```

**Image Processing (fashion show photos):**
```bash
python cli.py --embed-fashion-show-photos
python cli.py --load-fashion-show-photos
```

This generates one `.jsonl` file per season (e.g. `embeddings-images-FW1986-1987-fashion-show-photos.jsonl`) and loads them all into a single ChromaDB collection `images-fashion-show-photos`. Season is preserved as metadata on each record. Other image types (e.g. technical sheets, drawings) will use separate commands and collections when added.

### Query and Chat (CLI)

**Query** (returns raw retrieved chunks):
```bash
python cli.py --query --chunk_type recursive-split --q "What does Ferré say about elegance?"
```

**Chat** (RAG — retrieves chunks then generates an LLM answer):
```bash
python cli.py --chat --chunk_type recursive-split --q "What does Ferré say about elegance?"
```

Optional filters:
```bash
python cli.py --chat --q "..." --filter doc="Notes_White shirt"
python cli.py --chat --q "..." --contains "architecture"
python cli.py --chat --q "..." --top_k 20
```

**Agent mode** (LLM selects retrieval strategy automatically):
```bash
python cli.py --agent --q "What are Ferré's ideas on creativity?"
python cli.py --agent --q "What did Ferré say about fashion in 1997?"
python cli.py --agent --top_k 15 --q "How does Ferré describe the relationship between fashion and architecture?"
```

| Tool | When used | Filter |
|---|---|---|
| `search_archive` | General questions across all documents | None |
| `search_by_document` | Query targets a specific known document | `doc` = filename |
| `search_by_year` | Query asks about a specific year | `year` = e.g. `"1997"` |

Keep this container running while setting up the backend API service and frontend.

---

## 2. Backend API Service

Navigate to the API service directory:
```bash
cd ferre-rag-model/src/api-service
```

Build and run the container:
```bash
sh docker-shell.sh
```

Start the API service inside the container:
```bash
uvicorn_server
```

Verify the service is running at `http://localhost:9000`.

### View API Docs
FastAPI provides interactive API documentation automatically:
- Go to `http://localhost:9000/docs`
- You can test all endpoints from this tool

### Available Routes

| Prefix | Router | Description |
|---|---|---|
| `/llm` | `llm_chat.py` | Plain LLM chat with conversation history |
| `/llm-rag` | `llm_rag_chat.py` | RAG chat — retrieves archive chunks before answering |
| `/llm-agent` | `llm_agent_chat.py` | Agentic RAG — LLM selects retrieval tool automatically |

Each router exposes:
- `GET /chats` — list recent chats
- `GET /chats/{chat_id}` — get a specific chat
- `POST /chats` — start a new chat
- `POST /chats/{chat_id}` — continue an existing chat
- `GET /images/{chat_id}/{message_id}.png` — serve a chat image

Chat history is persisted to disk at `/persistent/chat-history/{model}/`.

Keep this container running while setting up the frontend.

---

## 3. Frontend

Navigate to the frontend directory:
```bash
cd ferre-rag-model/src/frontend-react
```

Build and run the container:
```bash
sh docker-shell.sh
```

First time only — install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm run dev
```

View the app at `http://localhost:3000`.

### App Structure

**Pages:**
- `/` — redirects to `/chat`
- `/chat` — main chat interface (home + active chat)

**Chat Models (selectable in the UI):**
- `Ferrè Assistant (LLM)` — plain conversational LLM
- `Ferrè Expert (RAG)` — retrieves archive chunks before answering
- `Ferrè Expert (Agent)` — agentic RAG with automatic tool selection

**Key components:**
- `ChatInput` — message input with image upload and model selector
- `ChatMessage` — renders conversation with markdown and images
- `ChatHistory` — recent chats grid on the home screen
- `ChatHistorySidebar` — chat list in the active chat view

**Data Service:**
- `src/services/DataService.js` — all API calls to the backend

---

## Architecture Diagrams

**Step 1:**
<img src="images/llm-rag-flow-1.png" width="800">

**Step 2:**
<img src="images/llm-rag-flow-2.png" width="800">

---

## Docker Cleanup

Make sure we do not have any running containers and clear up unused images:
```bash
docker container ls
# Stop any running containers
docker system prune
docker image ls
```

---

## Contributors
- Jack Webster
- Filippo Longhi
- Cecilia Zheng
- Asia Capezzuoli
- Stefan Golic

Note: the contents of `Dataset DataShack 2026` are exactly the same as the contents of the OneDrive folder shared by the Centro di Ricerca Ferr� for the purposes of this project.
