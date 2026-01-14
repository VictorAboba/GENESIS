#!/bin/bash

# conda check
export PATH="/opt/conda/bin:$PATH"
python_path=$(which python)
echo "Using python from: $python_path"

# run the main program
python /app/src/main.py 2>&1 | tee /app/logs/entrypoint.log