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
        python cli.py --chunk --chunk_type $CHUNK_TYPE || echo "⚠ Chunk step failed"
    fi

    if [ $SKIP_EMBED -eq 0 ]; then
        echo "→ Generating embeddings..."
        python cli.py --embed --chunk_type $CHUNK_TYPE || echo "⚠ Embed step failed"
    fi
else
    echo "✓ All pipeline outputs already exist, skipping chunk and embed"
fi

# Always load embeddings into ChromaDB
echo "→ Loading text embeddings to ChromaDB..."
python cli.py --load || echo "⚠ Load text embeddings step failed"

echo "→ Loading image embeddings to ChromaDB..."
python cli.py --load-images || echo "⚠ Load image embeddings step failed"

echo "✓ Pipeline complete"
echo ""
echo "Container ready! ChromaDB is available at http://localhost:8000"
echo "Keeping container running..."
exec /bin/bash