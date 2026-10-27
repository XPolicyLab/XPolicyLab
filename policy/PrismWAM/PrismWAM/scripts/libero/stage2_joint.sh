#!/usr/bin/env bash
# LIBERO, Stage 2: joint RGB / depth / action training  ->  configs/libero/stage2_joint.yaml
# Global batch 256, 15K steps.
# Starts from $OUTPUT_ROOT/libero/stage1_depth; pass model.depth_dit_checkpoint_path=<file>
# to start from another Stage-1 checkpoint.
#
#   bash scripts/libero/stage2_joint.sh                  # all GPUs on this node
#   bash scripts/libero/stage2_joint.sh --dry-run        # check the config, print the launch command
#   bash scripts/libero/stage2_joint.sh resume=$OUTPUT_ROOT/libero/stage2_joint/checkpoints/state/step_005000
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK and MASTER_ADDR set.
# The global batch is kept on any GPU count via gradient accumulation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="${DATA_ROOT:-./data}"           # parent of libero/, see docs/benchmarks/libero.md
WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"  # see docs/models.md
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"    # Stage 1 output is read from here

exec bash scripts/train.sh configs/libero/stage2_joint.yaml \
    paths.data_root="$DATA_ROOT" \
    paths.weights_root="$WEIGHTS_ROOT" \
    paths.output_root="$OUTPUT_ROOT" \
    "$@"
