#!/usr/bin/env bash
# RoboDojo, Stage 1: depth warm-up  ->  configs/robodojo/stage1_depth.yaml
# Global batch 1024, 15K steps.
#
#   bash scripts/robodojo/stage1_depth.sh                  # all GPUs on this node
#   bash scripts/robodojo/stage1_depth.sh --dry-run        # check the config, print the launch command
#   bash scripts/robodojo/stage1_depth.sh max_steps=200    # override any YAML key
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK and MASTER_ADDR set.
# The global batch is kept on any GPU count via gradient accumulation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="${DATA_ROOT:-./data}"           # parent of robodojo/, see docs/benchmarks/robodojo.md
WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"  # see docs/models.md
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-./external/RoboDojo}"  # RoboDojo checkout, needed to decode its RGB

exec bash scripts/train.sh configs/robodojo/stage1_depth.yaml \
    paths.data_root="$DATA_ROOT" \
    paths.weights_root="$WEIGHTS_ROOT" \
    paths.output_root="$OUTPUT_ROOT" \
    paths.robodojo_root="$ROBODOJO_ROOT" \
    "$@"
