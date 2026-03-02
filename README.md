# Gianfranco Ferré Archive RAG System

This project builds a Retrieval-Augmented Generation (RAG) system for the Gianfranco Ferré archive, using a vector database and a Large Language Model (LLM). The system ingests Ferré PDFs, chunks them, generates embeddings, stores them in ChromaDB, and enables search and chat over the archive.

<img src="images/ferre.png" width="800">

## Project Overview

**Workflow:**
1. Chunk Ferré archive PDFs into text segments
2. Generate embeddings for text chunks (Vertex AI, OpenAI, or other)
3. Generate embeddings for fashion show images by season
4. Load chunks, embeddings, and images into ChromaDB (separate collections per season)
5. Query and chat with the archive using LLM (text and images)
6. Agent mode: LLM automatically selects the right retrieval strategy and generates a grounded answer

**Architecture:**
- Python CLI pipeline (cli.py)
- ChromaDB vector database (local or managed)
- Vertex AI for embeddings (requires GCP service account)
- Docker for containerized development and deployment

## Prerequisites
- Docker installed
- Clone this repository
- Ferré archive PDFs (add to `input-datasets/ferre_notes_lessons/`)
- Ferré fashion show images (add to `input-datasets/ferre-designs/ALTA-MODA/[SEASON]/`)
- GCP service account with Vertex AI access (for text and image embeddings)

## Secrets & Environment Setup
- Copy `.env.example` to `.env` and fill in your GCP project ID
- Obtain a GCP service account key (JSON) and place it in `secrets/llm-service-account.json`
- **Never commit .env or service account files to the repo**

## Folder Structure
```
llm-rag/
├── input-datasets/
│   ├── ferre_notes_lessons/   # Place Ferré PDFs here
│   └── ferre-designs/         # Place fashion show images organized by season
│       └── ALTA-MODA/
│           ├── FW1986-87/     # Fall/Winter 1986-87 images
│           ├── SS1987/        # Spring/Summer 1987 images
│           └── ...
├── outputs/                   # Chunked and embedded data
├── secrets/
│   └── llm-service-account.json
├── cli.py
├── agent_tools.py
├── semantic_splitter.py
├── docker-shell.sh
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .env
```

## Setup GCP Service Account
1. Go to [GCP Console](https://console.cloud.google.com/home/dashboard)
2. Create a service account with "Vertex AI User" and "Storage Admin" roles
3. Download the JSON key and place it in `secrets/llm-service-account.json`
4. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to point to this file

## Running the Pipeline

### 1. Start the containers
```sh
./docker-shell.sh
```
This will build the Docker image, start ChromaDB, and run the chunk/embed/load pipeline automatically.

### How conditional startup works

When `docker-shell.sh` launches the container, `docker-entrypoint.sh` runs automatically and skips steps that have already been completed. Each step is checked independently:

| Step | Skip condition | What is checked |
|---|---|---|
| **Chunk** | `outputs/chunks*.jsonl` already exists | Presence of any chunk file in `outputs/`. If nothing is present, "python cli.py --chunk" is run (so chunking with default method defined in cli.py) |
| **Embed (text)** | `outputs/embeddings*.jsonl` already exists | Presence of any text embedding file in `outputs/` . If nothing is present, "python cli.py --embed" is run (so embedding with default method defined in cli.py)|
| **Load text** | At least one non-image collection exists in ChromaDB | ChromaDB is queried at startup; any collection whose name does **not** start with `images-` counts. If no non-image collection exists, "python cli.py --load" is run (so loading with default method defined in cli.py)|
| **Load images** | At least one image collection exists in ChromaDB | ChromaDB is queried at startup; any collection whose name starts with `images-` counts. If no image collection are present,  "python cli.py --load-images" is run (unique, not multiple methods possible) |

> **Note:** the checks are intentionally generic — they verify whether *any* data of that type is present, not whether it matches a specific `--chunk_type`. This means that if you want to ensure a particular chunk type (e.g. `char-split` or `semantic-split`) is chunked, embedded, and loaded, the automatic skip may trigger falsely because a previous run with a different type already populated the outputs.
>
> In that case, once the container has started and you are at the `/bin/bash` prompt, simply re-run the relevant steps manually:
> ```bash
> # Example: force chunking, embedding and loading for a specific type
> python cli.py --chunk --chunk_type char-split
> python cli.py --embed --chunk_type char-split
> python cli.py --load --chunk_type char-split
> ```

### Chunking parameters

All chunking methods measure `chunk_size` and `chunk_overlap` in **tokens** (not characters), using the `cl100k_base` tokenizer (GPT-4) as a close approximation for `text-embedding-004`. This ensures consistent semantic density regardless of the content and avoids silently exceeding the embedding model's 2048-token input limit.

The defaults are defined at the top of [cli.py](cli.py) and can be changed there:

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

For the Ferré archive (lecture notes, essays with dense paragraphs), **300–400 tokens** with **overlap 50–75 tokens** is a good starting point. After changing the constants, re-run the full pipeline for that chunk type:

```bash
python cli.py --chunk --chunk_type recursive-split
python cli.py --embed  --chunk_type recursive-split
python cli.py --load   --chunk_type recursive-split
```

> `semantic-split` does not use these constants — it determines boundaries from semantic similarity between sentences and produces variable-length chunks by design.

### Embedding parameters

The two relevant constants in [cli.py](cli.py) are:

```python
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
```

**`EMBEDDING_MODEL`**

| Model | Best for | Notes |
|---|---|---|
| `text-embedding-004` | English text | Default. Strong general-purpose model. |
| `text-multilingual-embedding-002` | Italian or mixed-language text | Recommended if Ferré archive PDFs are in Italian — handles 100+ languages natively, same API interface |

If the archive texts are in Italian, changing to `text-multilingual-embedding-002` will produce meaningfully better embeddings. The change is a one-liner in [cli.py](cli.py); then the full pipeline (embed + load) must be re-run.

**`EMBEDDING_DIMENSION`**

Both models support Matryoshka representations — the output vector can be truncated to a lower dimension without retraining, trading some quality for smaller storage and faster search.

| Dimension | Storage per chunk | Quality | When to use |
|---|---|---|---|
| 128 | minimal | lower | Not recommended for this use case |
| **256** | small | good | **Current default** — acceptable for prototyping |
| **512** | moderate | better | **Recommended upgrade** — meaningful quality gain at modest cost |
| 768 | full | best | Maximum quality; use if storage is not a concern |

For the Ferré archive, **512** is a practical improvement over 256 with no code changes beyond the constant. After changing either constant, delete existing embedding files in `outputs/` and re-run `--embed` and `--load`.

**Image embeddings**

Image embeddings use a separate model and are independent from the text embedding constants above:

- **Model:** `multimodalembedding@001` (Vertex AI) — dedicated multimodal model that embeds both images and text in the same vector space, enabling text-to-image similarity search
- **Dimension:** hardcoded `1408` in [cli.py](cli.py) — does **not** use `EMBEDDING_DIMENSION`. 1408 is the maximum available for this model (supports 128, 256, 512, 1408)

> Because text embeddings (`text-embedding-004`, 256-dim) and image embeddings (`multimodalembedding@001`, 1408-dim) live in different vector spaces, they cannot be searched together in a single query. To search images via a text query, the query must be embedded with `multimodalembedding@001` — not `text-embedding-004`.

### 2. Manual CLI Usage
You can run individual steps if needed:

**Text Processing:**
```bash
# Chunk PDFs
python cli.py --chunk --chunk_type recursive-split

# Generate embeddings for text chunks
python cli.py --embed --chunk_type recursive-split

# Load text embeddings into ChromaDB
python cli.py --load --chunk_type recursive-split
```

**Image Processing:**
```bash
# Generate embeddings for fashion show images (organized by season)
python cli.py --embed-images

# Load image embeddings into ChromaDB (creates separate collections per season)
python cli.py --load-images
```

### 3. Query and Chat
**Query** (returns raw retrieved chunks):
```bash
python cli.py --query --chunk_type recursive-split --q "What does Ferré say about elegance?"
```

**Chat** (RAG — retrieves chunks then generates an LLM answer):
```bash
python cli.py --chat --chunk_type recursive-split --q "What does Ferré say about elegance?"
```

Both support optional filters:
```bash
# Restrict to a specific document
python cli.py --chat --q "..." --filter doc="Notes_White shirt"

# Restrict by metadata field (requires --load to have been run after the metadata fix)
python cli.py --chat --q "..." --filter type=article

# Lexical filter on chunk text
python cli.py --chat --q "..." --contains "architecture"

# Increase retrieved chunks
python cli.py --chat --q "..." --top_k 20
```

### 4. Agent Mode
Agent mode uses a two-step agentic pipeline: the LLM first decides which retrieval tool to call (and with what arguments), then generates a grounded answer from the retrieved chunks. No manual filters are needed.

```bash
python cli.py --agent --q "What did Ferré write about his experience in India?"
python cli.py --agent --q "What are Ferré's ideas on creativity?"
python cli.py --agent --q "What did Ferré say about fashion in 1997?"
python cli.py --agent --top_k 15 --q "How does Ferré describe the relationship between fashion and architecture?"
```

**Available retrieval tools (selected automatically by the LLM):**

| Tool | When used | Filter |
|---|---|---|
| `search_archive` | General questions across all documents | None |
| `search_by_document` | Query targets a specific known document | `doc` = filename |
| `search_by_year` | Query asks about a specific year | `year` = e.g. `"1997"` |

## Secrets Management
- Use `.env.example` as a template for `.env`
- Store GCP credentials in `secrets/llm-service-account.json`
- Do not commit secrets to the repo
- For CI/CD, use GitHub Actions secrets

## Team Setup Instructions
1. Clone the repo
2. Add Ferré PDFs to `input-datasets/ferre_notes_lessons` (do not commit if restricted)
3. Create `.env` from `.env.example` and set your GCP project ID
4. Obtain your own GCP service account key and place in `secrets/llm-service-account.json`
5. Run `./docker-shell.sh` to start the pipeline

## Project Plan & Checklist
- [x] Vector DB setup (ChromaDB)
- [x] Data ingestion & chunking (PDFs)
- [x] Text embedding generation (Vertex AI)
- [x] Image embedding generation (Vertex AI MultiModalEmbeddingModel)
- [x] Load text embeddings to vector DB
- [x] Load image embeddings to vector DB (organized by season)
- [x] Text-based image search
- [x] Query & retrieval logic (text) with metadata and lexical filters
- [x] RAG chat endpoint
- [x] Metadata filtering (doc, year, type via `--filter` CLI arg)
- [x] Agent architecture (automatic tool selection via LLM function calling)
- [ ] Backend API (FastAPI/Flask)
- [ ] Deployment (Render/GCP/Vercel)
- [ ] Frontend chat UI
- [ ] Evaluation pipeline
- [ ] Documentation & onboarding


## Architecture Diagrams

**Step 1:**
<img src="images/llm-rag-flow-1.png" width="800">

**Step 2:**
<img src="images/llm-rag-flow-2.png" width="800">

## Contributors
- Jack Webster 
- Filippo Longhi
- Cecilia Zheng
- Asia Capezzuoli
- Stefan Golic
