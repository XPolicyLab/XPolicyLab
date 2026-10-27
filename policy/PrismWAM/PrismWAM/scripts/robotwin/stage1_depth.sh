#!/usr/bin/env bash
# RoboTwin, Stage 1: depth warm-up  ->  configs/robotwin/stage1_depth.yaml
# Global batch 256, 15K steps.
#
#   bash scripts/robotwin/stage1_depth.sh                  # all GPUs on this node
#   bash scripts/robotwin/stage1_depth.sh --dry-run        # check the config, print the launch command
#   bash scripts/robotwin/stage1_depth.sh max_steps=200    # override any YAML key
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK and MASTER_ADDR set.
# The global batch is kept on any GPU count via gradient accumulation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="${DATA_ROOT:-./data}"           # parent of robotwin/, see docs/benchmarks/robotwin.md
WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"  # see docs/models.md
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

exec bash scripts/train.sh configs/robotwin/stage1_depth.yaml \
    paths.data_root="$DATA_ROOT" \
    paths.weights_root="$WEIGHTS_ROOT" \
    paths.output_root="$OUTPUT_ROOT" \
    "$@"
