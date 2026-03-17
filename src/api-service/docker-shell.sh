#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

export MSYS_NO_PATHCONV=1

# Define some environment variables
export IMAGE_NAME="llm-rag-api-service"
export BASE_DIR=$(pwd)
export GCP_PROJECT="idyllic-psyche-487701-u7" # CHANGE TO YOUR PROJECT ID
export DEV=1

# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run all containers via docker-compose
MSYS_NO_PATHCONV=1 docker-compose run --rm --service-ports $IMAGE_NAME