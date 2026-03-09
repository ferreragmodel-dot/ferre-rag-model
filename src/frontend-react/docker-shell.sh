#!/bin/bash

set -e

export MSYS_NO_PATHCONV=1
export IMAGE_NAME="llm-rag-frontend-react"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile.dev .

# Run the container
docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME