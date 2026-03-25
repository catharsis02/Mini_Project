#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Create logs folder
mkdir -p logs

# Run preprocessing
echo "Running Stockwell preprocessing..."
python preprocess_stockwell_vectorized_v8.py \
  2>&1 | tee logs/preprocess_v8.log

# Run ML pipeline
echo "Running ML pipeline..."
python ml_pipeline_v10.py \
  2>&1 | tee logs/ml_pipeline_v10.log

echo "✅ Done"
