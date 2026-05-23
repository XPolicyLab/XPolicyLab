#!/bin/bash

# H-RDT RobotWin2 Data Processing Pipeline
# This script runs the complete RobotWin2 data processing pipeline

# Configuration: Set to "true" to enable steps, "false" to skip
ENABLE_STATS_CALCULATION=${ENABLE_STATS_CALCULATION:-false}
ENABLE_LANGUAGE_ENCODING=${ENABLE_LANGUAGE_ENCODING:-false}

# Setup environment (source the setup script if not already done)
if [ -z "$ROBOTWIN2_DATA_ROOT" ]; then
    source "$(dirname "$0")/setup_robotwin2.sh"
fi

# Change to project root
cd "$HRDT_PROJECT_ROOT"

# Define output paths
export STATS_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/stats.json"
export OUTLIER_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/outlier_files.txt"
export LANG_EMBEDDINGS_DIR="${HRDT_OUTPUT_DIR}/lang_embeddings"

echo "Starting RobotWin2 data processing pipeline..."
echo "Data Root: $ROBOTWIN2_DATA_ROOT"
echo "Output Dir: $HRDT_OUTPUT_DIR"
echo "Stats calculation enabled: $ENABLE_STATS_CALCULATION"
echo "Language encoding enabled: $ENABLE_LANGUAGE_ENCODING"

# Step 1: Calculate dataset statistics (Optional)
if [ "$ENABLE_STATS_CALCULATION" = "true" ]; then
    echo "Step 1: Calculating dataset statistics..."
    python datasets/robotwin2/calc_stat.py \
        --root_dir "$ROBOTWIN2_DATA_ROOT" \
        --output_path "$STATS_OUTPUT_PATH" \
        --outlier_path "$OUTLIER_OUTPUT_PATH" \
        --num_processes "$NUM_PROCESSES"
    
    if [ $? -eq 0 ]; then
        echo "✓ Statistics calculation completed successfully"
    else
        echo "✗ Statistics calculation failed"
        exit 1
    fi
else
    echo "Step 1: Skipping statistics calculation (already available or disabled)"
fi

# Step 2: Encode language embeddings (Optional)
if [ "$ENABLE_LANGUAGE_ENCODING" = "true" ]; then
    echo "Step 2: Encoding language embeddings..."
    python datasets/robotwin2/encode_lang_batch.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Language encoding completed successfully"
    else
        echo "✗ Language encoding failed"
        exit 1
    fi
else
    echo "Step 2: Skipping language encoding (already available or disabled)"
    if [ -d "$LANG_EMBEDDINGS_DIR" ]; then
        echo "✓ Language embeddings found at: $LANG_EMBEDDINGS_DIR"
    else
        echo "⚠ Warning: Language embeddings directory not found: $LANG_EMBEDDINGS_DIR"
        echo "  Please run with ENABLE_LANGUAGE_ENCODING=true or ensure embeddings are available"
    fi
fi

echo ""
echo "RobotWin2 pipeline completed!"
echo "Available files:"
echo "  - Statistics: $STATS_OUTPUT_PATH"
echo "  - Outlier files: $OUTLIER_OUTPUT_PATH"
echo "  - Language embeddings: $LANG_EMBEDDINGS_DIR/*.pt"
echo ""
echo "To run specific steps, use:"
echo "  ENABLE_STATS_CALCULATION=true ./run_robotwin2_pipeline.sh"
echo "  ENABLE_LANGUAGE_ENCODING=true ./run_robotwin2_pipeline.sh" 