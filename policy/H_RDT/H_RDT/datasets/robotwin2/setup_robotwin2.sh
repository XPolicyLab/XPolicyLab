#!/bin/bash

# H-RDT RobotWin2 Data Processing Setup
# Set your paths here

# Required paths - modify these according to your environment
export ROBOTWIN2_DATA_ROOT="/share/hongzhe/datasets/robotwin2/dataset/aloha-agilex"
export T5_MODEL_PATH="/data/lingxuan/weights/t5-v1_1-xxl"

# Project structure (auto-detected)
export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_finetune.yaml"
export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/robotwin2"

# Processing parameters
export NUM_PROCESSES=64
export NUM_GPUS=8
export PROCESSES_PER_GPU=6

# Create output directory
mkdir -p "$HRDT_OUTPUT_DIR"

# Add project to Python path
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

echo "RobotWin2 environment setup completed"
echo "Data Root: $ROBOTWIN2_DATA_ROOT"
echo "T5 Model: $T5_MODEL_PATH"
echo "Output Dir: $HRDT_OUTPUT_DIR" 