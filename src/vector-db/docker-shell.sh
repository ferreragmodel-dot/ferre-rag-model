#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Load environment variables from root .env file
ROOT_ENV="../../.env"
if [ -f "$ROOT_ENV" ]; then
    echo "Loading environment from $ROOT_ENV..."
    export $(cat "$ROOT_ENV" | grep -v '^#' | xargs)
else
    echo "⚠ Warning: $ROOT_ENV not found"
fi

# Set variables with defaults
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="${GCP_PROJECT:?Error: GCP_PROJECT not set in .env}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:?Error: GOOGLE_APPLICATION_CREDENTIALS not set in .env}"
export IMAGE_NAME="llm-rag-cli"

echo "Configuration:"
echo "  GCP_PROJECT: $GCP_PROJECT"
echo "  CHROMADB_HOST: ${CHROMADB_HOST:-llm-rag-chromadb}"
echo "  CHROMADB_PORT: ${CHROMADB_PORT:-8000}"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
MSYS_NO_PATHCONV=1 docker-compose run --rm --service-ports $IMAGE_NAME