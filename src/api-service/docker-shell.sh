#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export MSYS_NO_PATHCONV=1

# Load environment variables from root .env file
ROOT_ENV="../../.env"
if [ -f "$ROOT_ENV" ]; then
    echo "Loading environment from $ROOT_ENV..."
    export $(cat "$ROOT_ENV" | grep -v '^#' | xargs)
else
    echo "⚠ Warning: $ROOT_ENV not found"
fi

# Define some environment variables with defaults
export IMAGE_NAME="llm-rag-api-service"
export BASE_DIR=$(pwd)
export GCP_PROJECT="${GCP_PROJECT:?Error: GCP_PROJECT not set in .env}"
export DEV=${DEV:-1}

echo "Configuration:"
echo "  IMAGE_NAME: $IMAGE_NAME"
echo "  GCP_PROJECT: $GCP_PROJECT"
echo "  CHROMADB_HOST: ${CHROMADB_HOST:-llm-rag-chromadb}"
echo "  POSTGRES_HOST: ${POSTGRES_HOST:-localhost}"

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run all containers via docker-compose
MSYS_NO_PATHCONV=1 docker-compose run --rm --service-ports $IMAGE_NAME