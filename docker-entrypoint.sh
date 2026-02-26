#!/bin/bash
set -e

echo "Container is running!!!"
echo "Architecture: $(uname -m)"

# Dependencies are already installed via uv sync in Dockerfile
echo "Dependencies already installed via uv"

# Activate virtual environment
echo "Activating virtual environment..."
source /.venv/bin/activate

echo "Python version: $(python --version)"
echo "Checking fitz import..."
python -c "import fitz; print('✓ fitz available')" || echo "✗ fitz import failed"

# Check if pipeline outputs exist
CHUNK_OUTPUT="outputs/chunks*.jsonl"
EMBED_OUTPUT="outputs/embeddings*.jsonl"
IMAGE_EMBED_OUTPUT="outputs/embeddings-images*.jsonl"

echo ""
echo "Checking pipeline status..."
if ls $CHUNK_OUTPUT 1> /dev/null 2>&1; then
    echo "✓ Chunk output already exists, skipping chunk step"
    SKIP_CHUNK=1
else
    echo "✗ Chunk output not found, will run chunk step"
    SKIP_CHUNK=0
fi

if ls $EMBED_OUTPUT 1> /dev/null 2>&1; then
    echo "✓ Embedding output already exists, skipping embed step"
    SKIP_EMBED=1
else
    echo "✗ Embedding output not found, will run embed step"
    SKIP_EMBED=0
fi

if ls $IMAGE_EMBED_OUTPUT 1> /dev/null 2>&1; then
    echo "✓ Image embedding output already exists, skipping image embed step"
    SKIP_IMAGE_EMBED=1
else
    echo "✗ Image embedding output not found, will run image embed step"
    SKIP_IMAGE_EMBED=0
fi

# Run pipeline conditionally (but always load)
echo ""
if [ $SKIP_CHUNK -eq 0 ] || [ $SKIP_EMBED -eq 0 ]; then
    echo "Running Ferré archive pipeline..."

    if [ $SKIP_CHUNK -eq 0 ]; then
        echo "→ Chunking documents..."
        python cli.py --chunk || echo "⚠ Chunk step failed"
    fi

    if [ $SKIP_EMBED -eq 0 ]; then
        echo "→ Generating embeddings..."
        python cli.py --embed || echo "⚠ Embed step failed"
    fi
else
    echo "✓ All pipeline outputs already exist, skipping chunk and embed"
fi

# Check if ChromaDB collections already exist (tied to the persistent volume lifecycle)
echo ""
echo "Checking ChromaDB collection status..."
SKIP_LOAD_TEXT=$(python -c "
import chromadb
try:
    c = chromadb.HttpClient(host='llm-rag-chromadb', port=8000)
    cols = [col.name for col in c.list_collections()]
    has_text = any(not col.startswith('images-') for col in cols)
    print('1' if has_text else '0')
except Exception:
    print('0')
" 2>/dev/null)

SKIP_LOAD_IMAGES=$(python -c "
import chromadb
try:
    c = chromadb.HttpClient(host='llm-rag-chromadb', port=8000)
    cols = [col.name for col in c.list_collections()]
    print('1' if any(col.startswith('images-') for col in cols) else '0')
except Exception:
    print('0')
" 2>/dev/null)

if [ "$SKIP_LOAD_TEXT" = "1" ]; then
    echo "✓ Text collection already exists in ChromaDB, skipping text load"
else
    echo "✗ Text collection not found in ChromaDB, will load text embeddings"
fi

if [ "$SKIP_LOAD_IMAGES" = "1" ]; then
    echo "✓ Image collections already exist in ChromaDB, skipping image load"
else
    echo "✗ Image collections not found in ChromaDB, will load image embeddings"
fi

# Load embeddings into ChromaDB conditionally
if [ "$SKIP_LOAD_TEXT" = "0" ]; then
    echo "→ Loading text embeddings to ChromaDB..."
    python cli.py --load || echo "⚠ Load text embeddings step failed"
fi

if [ "$SKIP_LOAD_IMAGES" = "0" ]; then
    echo "→ Loading image embeddings to ChromaDB..."
    python cli.py --load-images || echo "⚠ Load image embeddings step failed"
fi

echo "✓ Pipeline complete"
echo ""
echo "Container ready! ChromaDB is available at http://localhost:8000"
echo "Keeping container running..."
exec /bin/bash