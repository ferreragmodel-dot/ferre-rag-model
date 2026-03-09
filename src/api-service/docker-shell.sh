#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export MSYS_NO_PATHCONV=1

# Define some environment variables
export IMAGE_NAME="llm-rag-api-service"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export GCP_PROJECT="project-296db417-a890-492a-80a" # CHANGE TO YOUR PROJECT ID
export CHROMADB_HOST="llm-rag-chromadb"
export CHROMADB_PORT=8000

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-v "${APPDATA}/gcloud:/home/app/.config/gcloud" \
-p 9000:9000 \
-e DEV=1 \
-e GOOGLE_APPLICATION_CREDENTIALS=/home/app/.config/gcloud/application_default_credentials.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e CHROMADB_HOST=$CHROMADB_HOST \
-e CHROMADB_PORT=$CHROMADB_PORT \
--network llm-rag-network \
$IMAGE_NAME