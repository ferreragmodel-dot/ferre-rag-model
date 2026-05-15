# Gianfranco Ferré Archive

This project builds a Retrieval-Augmented Generation (RAG) system for the Gianfranco Ferré archive, using a vector database and a Large Language Model (LLM). The system ingests Ferré PDFs, chunks them, generates embeddings, stores them in ChromaDB, and enables search and chat over the archive through a web interface.

<img src="images/ferre.png" width="800">

## System Overview

Three services work together at runtime:

- **`src/api-service/`** — FastAPI backend. Serves the archive browse endpoints (`/archive/*`), proxies images from GCS, and runs the agentic RAG chat (`/llm-agent/*`). Connects to ChromaDB (for vector search) and PostgreSQL (for item metadata).
- **`src/frontend-react/`** — Next.js app. Renders the image grid, filters, item detail modal, and conversation popup. Talks only to the api-service.
- **`src/vector-db/`** — Offline CLI (not a server). Used once to chunk archive PDFs, generate embeddings, and load them into ChromaDB. Can target a local or deployed ChromaDB instance.

**Infrastructure:**

| Component | Role |
|---|---|
| ChromaDB | Vector store — text chunks (`semantic-split-collection`, 256-dim) and fashion show images (`images-fashion-show-photos`, 1408-dim) |
| PostgreSQL | `FashionItem` metadata table, seeded from archive JSONL on api-service startup |
| Google Cloud Storage | Image hosting — bucket `ferre-archive`, proxied through the api-service |
| Gemini 2.0 Flash | LLM powering the agent chat |
| Vertex AI | Embedding models — `text-embedding-004` (text) and `multimodalembedding@001` (images) |

---

## Folder Structure

```
ferre-rag-model/
├── .env                        # Root env vars shared across services
├── src/
│   ├── vector-db/              # Offline embedding pipeline
│   │   ├── cli.py              # Main CLI (--chunk, --embed, --load, --query, --chat)
│   │   ├── semantic_splitter.py
│   │   ├── docker-shell.sh
│   │   ├── docker-compose.yml  # Includes local ChromaDB container
│   │   ├── docker-entrypoint.sh
│   │   └── Dockerfile
│   ├── api-service/            # FastAPI backend
│   │   ├── api/
│   │   │   ├── service.py              # App entrypoint + /archive/* routes
│   │   │   ├── routers/
│   │   │   │   └── llm_agent_chat.py   # /llm-agent/chats, /llm-agent/item-chats
│   │   │   ├── models/
│   │   │   │   └── fashion_item.py     # SQLModel FashionItem table
│   │   │   ├── seeds/
│   │   │   │   └── seed.py             # Seeds FashionItem from JSONL on startup
│   │   │   └── utils/
│   │   │       ├── agent_orchestrator.py   # Gemini function-calling agent
│   │   │       ├── retrieval_tools.py      # ChromaDB retrieval helpers
│   │   │       ├── chat_utils.py           # Chat persistence
│   │   │       └── gcs_utils.py            # GCS image fetching + proxy
│   │   ├── .env
│   │   ├── docker-shell.sh
│   │   └── docker-compose.yml      # Includes PostgreSQL container
│   └── frontend-react/         # Next.js visual archive browser
│       ├── src/
│       │   ├── app/
│       │   │   ├── page.tsx            # Home — image grid + search + filters
│       │   │   └── layout.tsx          # Root layout with password gate
│       │   ├── components/
│       │   │   ├── ImageGrid.tsx       # Infinite-scroll masonry grid
│       │   │   ├── ImageCard.tsx       # Single archive card
│       │   │   ├── FilterBar.tsx       # Season / item / colour / material chips
│       │   │   ├── SearchBar.tsx       # Floating search → opens ConversationPopup
│       │   │   ├── ConversationPopup.tsx   # Full-screen agent chat with citations
│       │   │   ├── ItemDetailModal.tsx     # Item detail + cluster carousel + item chat
│       │   │   ├── TitleBar.tsx
│       │   │   └── PasswordGate.tsx    # Session-scoped password gate
│       │   └── lib/
│       │       ├── api.ts              # All fetch calls to the backend
│       │       ├── types.ts            # Shared TypeScript types
│       │       └── chat-utils.tsx      # Shared citation rendering utility
│       ├── next.config.js
│       └── package.json
```

---

## Prerequisites

- Docker
- Node.js 18+ (for frontend local dev without Docker)
- `gcloud` CLI installed and authenticated:
  ```bash
  gcloud auth application-default login
  ```
- Root `.env` file in the repo root (see below)

---

## Environment Setup

### Root `.env` (repo root)

Controls which ChromaDB instance all services connect to and holds shared credentials:

```env
GCP_PROJECT=ferre-rag-model
POSTGRES_DB=ragdb
POSTGRES_USER=app
POSTGRES_PASSWORD=app
GOOGLE_API_KEY=<your Gemini API key>

# ChromaDB target — switch between local and deployed:
CHROMADB_HOST=ferre-chromadb-323252296985.us-central1.run.app
CHROMADB_PORT=443
CHROMADB_SSL=true
```

To point at a **local** ChromaDB container instead:
```env
CHROMADB_HOST=llm-rag-chromadb
CHROMADB_PORT=8000
CHROMADB_SSL=false
```

### `src/api-service/.env`

Mirrors the ChromaDB vars plus service-specific config:
```env
GCP_PROJECT=ferre-rag-model
GCS_BUCKET=ferre-archive
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=app
POSTGRES_PASSWORD=app
CHROMADB_HOST=ferre-chromadb-323252296985.us-central1.run.app
CHROMADB_PORT=443
CHROMADB_SSL=true
```

### Google credentials

The api-service container reads Application Default Credentials from `~/.config/gcloud` (mounted read-only at `/home/app/.config/gcloud`). No service account key file is needed — ADC from `gcloud auth application-default login` is sufficient.

---

## Running Locally

Each service can independently point at a **local** Docker container or a **deployed** Cloud Run instance. Mix and match depending on what you need:

| Service | Local | Deployed |
|---|---|---|
| **ChromaDB** | `llm-rag-chromadb` container (port 8000) — started by vector-db compose | `ferre-chromadb-323252296985.us-central1.run.app:443` |
| **api-service** | Docker container (port 9000) | `ferre-api-323252296985.us-central1.run.app` |
| **PostgreSQL** | Docker container (port 5432) — started by api-service compose | — |
| **frontend** | `npm run dev` (port 3000) | Deployed separately |
| **vector-db** | Always runs locally (offline pipeline, not a server) | N/A |

### Typical local dev setup

This is the most common configuration: api-service and frontend run locally, ChromaDB and GCS are the deployed instances.

**1. Start the api-service:**
```bash
cd src/api-service
sh docker-shell.sh        # builds image, starts container + postgres
# inside the container:
uvicorn_server
```
API available at `http://localhost:9000`. Interactive docs at `http://localhost:9000/docs`.

**2. Start the frontend:**
```bash
cd src/frontend-react
npm install               # first time only
npm run dev
```
App available at `http://localhost:3000`.

The frontend defaults to `http://localhost:9000` for the API. To point it at the deployed api-service instead, create `src/frontend-react/.env.local`:
```env
NEXT_PUBLIC_API_BASE_URL=https://ferre-api-323252296985.us-central1.run.app
```

### Using a local ChromaDB (for vector-db loading)

The local ChromaDB container (`llm-rag-chromadb`) is defined in `src/vector-db/docker-compose.yml` and starts automatically when you run `sh docker-shell.sh` there.

To switch everything to use it, update the root `.env`:
```env
CHROMADB_HOST=llm-rag-chromadb
CHROMADB_PORT=8000
CHROMADB_SSL=false
```
Or override inline without touching the file:
```bash
CHROMADB_HOST=llm-rag-chromadb CHROMADB_PORT=8000 CHROMADB_SSL=false sh docker-shell.sh
```

---

## Vector DB — Embedding Pipeline

The vector-db is an **offline CLI tool** — it is not a server. Run it once to populate ChromaDB, then point the api-service at the same ChromaDB instance.

```bash
cd src/vector-db
sh docker-shell.sh        # builds image, opens shell (starts local ChromaDB if CHROMADB_HOST=llm-rag-chromadb)
```

### Text pipeline (archive PDFs → ChromaDB)

```bash
# Inside the container:
python cli.py --chunk --chunk_type recursive-split
python cli.py --embed --chunk_type recursive-split
python cli.py --load --chunk_type recursive-split
```

Outputs are saved to `outputs/` as `.jsonl` files. Re-running skips steps whose output files already exist.

### Image pipeline (fashion show photos → ChromaDB)

```bash
python cli.py --embed-fashion-show-photos
python cli.py --load-fashion-show-photos
```

One `.jsonl` per season is written to `outputs/`. All seasons load into a single ChromaDB collection `images-fashion-show-photos` with season preserved as metadata on each document.

The image metadata generation workflow is documented in `src/vector-db/metadata/images_metadata/README.md`. Its final metadata artifact is loaded by `python cli.py --load-fashion-show-photos` and copied into the api-service seed data for Postgres.

### Query / chat (debugging)

```bash
python cli.py --query --chunk_type recursive-split --q "What does Ferré say about elegance?"
python cli.py --chat  --chunk_type recursive-split --q "What does Ferré say about elegance?"

# Optional filters:
python cli.py --chat --q "..." --filter doc="Notes_White shirt"
python cli.py --chat --q "..." --contains "architecture"
python cli.py --chat --q "..." --top_k 20
```

## API Service — Routes

| Route | Description |
|---|---|
| `GET /archive/landing-feed` | Paginated image grid with season / garment / colour / material filters |
| `GET /archive/item-detail` | Full metadata for a single archive item |
| `GET /archive/item-cluster` | Other items in the same outfit cluster |
| `GET /archive/filter-options` | Available filter values for the UI dropdowns |
| `GET /archive/image` | GCS image proxy (streams image through the backend) |
| `POST /llm-agent/chats` | Start a new multi-turn agent chat |
| `POST /llm-agent/chats/{id}` | Continue an existing chat |
| `POST /llm-agent/item-chats` | Start a chat scoped to a specific archive item |
| `POST /llm-agent/item-chats/{id}` | Continue an item-scoped chat |

Interactive docs: `http://localhost:9000/docs`

---

## Docker Cleanup

```bash
docker container ls
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

> The contents of `Dataset DataShack 2026` are provided by the Centro di Ricerca Ferré for the purposes of this project.
