#!/usr/bin/env bash
# RoboDojo, Stage 2: joint RGB / depth / action training  ->  configs/robodojo/stage2_joint.yaml
# Global batch 1024, 40K steps.
# Starts from $OUTPUT_ROOT/robodojo/stage1_depth; pass model.depth_dit_checkpoint_path=<file>
# to start from another Stage-1 checkpoint.
#
#   bash scripts/robodojo/stage2_joint.sh                  # all GPUs on this node
#   bash scripts/robodojo/stage2_joint.sh --dry-run        # check the config, print the launch command
#   bash scripts/robodojo/stage2_joint.sh resume=$OUTPUT_ROOT/robodojo/stage2_joint/checkpoints/state/step_005000
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK and MASTER_ADDR set.
# The global batch is kept on any GPU count via gradient accumulation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="${DATA_ROOT:-./data}"           # parent of robodojo/, see docs/benchmarks/robodojo.md
WEIGHTS_ROOT="${WEIGHTS_ROOT:-./weights}"  # see docs/models.md
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"    # Stage 1 output is read from here
ROBODOJO_ROOT="${ROBODOJO_ROOT:-./external/RoboDojo}"  # RoboDojo checkout, needed to decode its RGB

exec bash scripts/train.sh configs/robodojo/stage2_joint.yaml \
    paths.data_root="$DATA_ROOT" \
    paths.weights_root="$WEIGHTS_ROOT" \
    paths.output_root="$OUTPUT_ROOT" \
    paths.robodojo_root="$ROBODOJO_ROOT" \
    "$@"
