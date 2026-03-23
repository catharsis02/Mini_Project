#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Create logs folder
mkdir -p logs

# Run preprocessing
echo "Running Stockwell preprocessing..."
python preprocess_stockwell_vectorized.py \
  2>&1 | tee logs/preprocess.log

# Run ML pipeline
echo "Running ML pipeline..."
python ml_pipeline.py \
  2>&1 | tee logs/ml_pipeline.log

echo "✅ Done"