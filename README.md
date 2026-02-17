# Gianfranco Ferré Archive RAG System

This project builds a Retrieval-Augmented Generation (RAG) system for the Gianfranco Ferré archive, using a vector database and a Large Language Model (LLM). The system ingests Ferré PDFs, chunks them, generates embeddings, stores them in ChromaDB, and enables search and chat over the archive.

<img src="images/ferre.png" width="800">

## Project Overview

**Workflow:**
1. Chunk Ferré archive PDFs into text segments
2. Generate embeddings for each chunk (Vertex AI, OpenAI, or other)
3. Load chunks and embeddings into ChromaDB
4. Query and chat with the archive using LLM

**Architecture:**
- Python CLI pipeline (cli.py)
- ChromaDB vector database (local or managed)
- Vertex AI for embeddings (requires GCP service account)
- Docker for containerized development and deployment

## Prerequisites
- Docker installed
- Clone this repository
- Ferré archive PDFs (not included; add to input-datasets/ferre_notes_lessons)
- GCP service account with Vertex AI access

## Secrets & Environment Setup
- Copy `.env.example` to `.env` and fill in your GCP project ID
- Obtain a GCP service account key (JSON) and place it in `secrets/llm-service-account.json`
- **Never commit .env or service account files to the repo**

## Folder Structure
```
llm-rag/
├── input-datasets/
│   └── ferre_notes_lessons/   # Place Ferré PDFs here
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

### 2. Manual CLI Usage
You can run individual steps if needed:

**Chunk PDFs:**
```
python cli.py --chunk --chunk_type recursive-split
```
**Generate Embeddings:**
```
python cli.py --embed --chunk_type recursive-split
```
**Load into ChromaDB:**
```
python cli.py --load --chunk_type recursive-split
```

### 3. Query and Chat
**Query:**
```
python cli.py --query --chunk_type recursive-split
```
**Chat:**
```
python cli.py --chat --chunk_type recursive-split
```

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
- [x] Embedding generation (Vertex AI)
- [x] Load to vector DB
- [ ] Query & retrieval logic
- [ ] RAG chat endpoint
- [ ] Metadata filtering
- [ ] Backend API (FastAPI/Flask)
- [ ] Deployment (Render/GCP/Vercel)
- [ ] Frontend chat UI
- [ ] Agent architecture
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
