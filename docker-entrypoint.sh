#!/bin/bash

echo "Container is running!!!"
echo "Architecture: $(uname -m)"

# Activate virtual environment
echo "Activating virtual environment..."
source /.venv/bin/activate

echo "Environment ready! Virtual environment activated."
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"

# If "pipeline" arg is passed, run the full chunk/embed/load pipeline.
# Otherwise drop into an interactive bash shell (default for docker-shell.sh).
if [ "$1" = "pipeline" ]; then
    echo "Running Ferré archive pipeline: chunk, embed, load"
    python cli.py --chunk --chunk_type recursive-split
    python cli.py --embed --chunk_type recursive-split
    python cli.py --load --chunk_type recursive-split
    echo "Pipeline complete. Container exiting."
else
    echo "Dropping into interactive shell. Run 'python cli.py --help' to get started."
    exec /bin/bash
fi
