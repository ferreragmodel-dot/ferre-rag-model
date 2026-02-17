#!/bin/bash

echo "Container is running!!!"
echo "Architecture: $(uname -m)"

echo "Environment ready! Virtual environment activated."
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

# Activate virtual environment
echo "Activating virtual environment..."
source /.venv/bin/activate

echo "Running Ferré archive pipeline: chunk, embed, load"
python cli.py --chunk --chunk_type recursive-split
python cli.py --embed --chunk_type recursive-split
python cli.py --load --chunk_type recursive-split
echo "Pipeline complete. Container exiting."