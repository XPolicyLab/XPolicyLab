#!/usr/bin/env bash
# Build a TensorRT engine from a DreamZero checkpoint.
#
# Usage (recommended — with real calibration data):
#   bash scripts/inference/build_trt_engine.sh \
#       --model-path ./checkpoints/DreamZero-DROID \
#       --tensorrt nvfp4 \
#       --dataset-path ./data/droid_lerobot \
#       --cuda-device 0
#
# Usage (without dataset — acceptable for fp16, not recommended for nvfp4/fp8):
#   bash scripts/inference/build_trt_engine.sh \
#       --model-path ./checkpoints/DreamZero-DROID \
#       --tensorrt nvfp4 \
#       --cuda-device 0
#
# The engine is saved to:
#   {model_path}/tensorrt/wan/WanModel_{precision}.trt
#
# Supported precisions: nvfp4 (recommended), fp8, fp16
#
# For quantized precisions (nvfp4, fp8), ModelOpt calibrates quantization
# parameters using real forward passes.  Providing --dataset-path is strongly
# recommended — random dummy inputs are used as fallback but reduce accuracy.
#
# ENABLE_TENSORRT=true must be set before any groot modules are imported
# (it controls flash-attention compatibility mode for ONNX/TRT export).
# This script sets it and launches the Python build script via torchrun so
# that RANK / WORLD_SIZE env vars are available for GrootSimPolicy init.

# export HF_HUB_CACHE=/mnt/aws-lfs-02/shared/ckpts
set -euo pipefail

MODEL_PATH=""
TENSORRT_PRECISION=""
CUDA_DEVICE="0"
DATASET_PATH=""
NUM_CALIBRATION_TRAJS="2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tensorrt)
            TENSORRT_PRECISION="$2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --num-calibration-trajs)
            NUM_CALIBRATION_TRAJS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-path <path> --tensorrt <precision> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH             Path to DreamZero checkpoint directory"
            echo "  --tensorrt PRECISION          TRT precision: nvfp4 (recommended), fp8, fp16"
            echo "  --dataset-path PATH           LeRobot dataset for real calibration (recommended for nvfp4/fp8)"
            echo "  --num-calibration-trajs N     Number of calibration trajectories (default: 2)"
            echo "  --cuda-device ID              CUDA device index (default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required" >&2
    exit 1
fi

if [[ -z "$TENSORRT_PRECISION" ]]; then
    echo "Error: --tensorrt is required (e.g. nvfp4, fp8, fp16)" >&2
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi

ENGINE_PATH="${MODEL_PATH}/tensorrt/wan/WanModel_${TENSORRT_PRECISION}.trt"

echo "=========================================="
echo "DreamZero TensorRT Engine Builder"
echo "  Checkpoint         : $MODEL_PATH"
echo "  Precision          : $TENSORRT_PRECISION"
echo "  CUDA device        : $CUDA_DEVICE"
echo "  Dataset (calibrate): ${DATASET_PATH:-<none — using dummy inputs>}"
echo "  Calibration trajs  : $NUM_CALIBRATION_TRAJS"
echo "  Output             : $ENGINE_PATH"
echo "=========================================="

# ENABLE_TENSORRT must be set before Python imports any groot model modules
# (it activates flash-attention compatibility mode required for ONNX/TRT export).
export ENABLE_TENSORRT=true
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export ATTENTION_BACKEND="TE"
export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Build the Python argument list.
PYTHON_ARGS=(
    --model-path "$MODEL_PATH"
    --tensorrt   "$TENSORRT_PRECISION"
    --num-calibration-trajs "$NUM_CALIBRATION_TRAJS"
)
if [[ -n "$DATASET_PATH" ]]; then
    PYTHON_ARGS+=(--dataset-path "$DATASET_PATH")
fi

# torchrun sets RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT which are
# required by GrootSimPolicy's distributed init.
torchrun \
    --standalone \
    --nproc_per_node=1 \
    "${REPO_ROOT}/scripts/inference/build_trt_engine_droid.py" \
    "${PYTHON_ARGS[@]}"

echo "=========================================="
echo "Engine built successfully: $ENGINE_PATH"
echo ""
echo "Run inference with:"
echo "  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \\"
echo "      socket_test_optimized_AR.py --port 5000 --enable-dit-cache \\"
echo "      --model-path ${MODEL_PATH} --tensorrt ${TENSORRT_PRECISION}"
echo "=========================================="
