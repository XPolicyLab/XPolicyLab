#!/bin/bash

# H-RDT Pretrain Data Processing Pipeline
# This script runs the complete pretrain data processing pipeline

# Setup environment (source the setup script if not already done)
if [ -z "$EGODEX_DATA_ROOT" ]; then
    source "$(dirname "$0")/setup_pretrain.sh"
fi

# Change to project root
cd "$HRDT_PROJECT_ROOT"

# Define output paths
export STATS_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/egodex_stat.json"
export LARGE_VALUES_LOG="${HRDT_OUTPUT_DIR}/egodex_large_values.txt"

echo "Starting pretrain data processing pipeline..."
echo "Data Root: $EGODEX_DATA_ROOT"
echo "Output Dir: $HRDT_OUTPUT_DIR"

# Step 1: Precompute 48D actions
echo "Step 1: Precomputing 48D actions..."
python datasets/pretrain/precompute_48d_actions.py \
    --data_root "$EGODEX_DATA_ROOT" \
    --num_processes "$NUM_PROCESSES" \
    $([ "$FORCE_OVERWRITE" = "true" ] && echo "--force_overwrite")

# Step 2: Calculate statistics
echo "Step 2: Calculating dataset statistics..."
python datasets/pretrain/calc_stat.py \
    --data_root "$EGODEX_DATA_ROOT" \
    --output_path "$STATS_OUTPUT_PATH" \
    --large_values_log "$LARGE_VALUES_LOG"

# Step 3: Encode language embeddings
echo "Step 3: Encoding language embeddings..."
python datasets/pretrain/encode_lang_batch.py

echo "Pretrain pipeline completed!"
echo "Generated files:"
echo "  - 48D action data: Added to HDF5 files as 'actions_48d' key"
echo "  - Statistics: $STATS_OUTPUT_PATH"
echo "  - Large values log: $LARGE_VALUES_LOG"
echo "  - Language embeddings: *.pt files alongside HDF5 files" 