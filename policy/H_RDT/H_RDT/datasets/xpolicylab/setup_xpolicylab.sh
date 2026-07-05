#!/bin/bash

# H-RDT XPolicyLab Data Processing Setup
# Adjust these paths for your environment when needed.

export XPOLICYLAB_DATA_ROOT="${XPOLICYLAB_DATA_ROOT:-/vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud}"
export XPOLICYLAB_RAW_BENCH_NAME="${XPOLICYLAB_RAW_BENCH_NAME:-RoboDojo}"
export XPOLICYLAB_ENV_CFG_TYPE="${XPOLICYLAB_ENV_CFG_TYPE:-arx_x5}"
export XPOLICYLAB_ACTION_TYPE="${XPOLICYLAB_ACTION_TYPE:-joint}"
export T5_MODEL_PATH="${T5_MODEL_PATH:-/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT/t5-v1_1-xxl}"

export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_finetune.yaml"
export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/xpolicylab"

export NUM_PROCESSES="${NUM_PROCESSES:-64}"
export HRDT_LANG_GPU="${HRDT_LANG_GPU:-0}"

XPOLICYLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
DEMO_ENV_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"

mkdir -p "$HRDT_OUTPUT_DIR"

export PYTHONPATH="${DEMO_ENV_ROOT}:${XPOLICYLAB_ROOT}:${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

echo "XPolicyLab environment setup completed"
echo "Data Root: $XPOLICYLAB_DATA_ROOT"
echo "Raw Dataset: $XPOLICYLAB_RAW_BENCH_NAME"
echo "Env Config: $XPOLICYLAB_ENV_CFG_TYPE"
echo "Action Type: $XPOLICYLAB_ACTION_TYPE"
echo "T5 Model: $T5_MODEL_PATH"
echo "Output Dir: $HRDT_OUTPUT_DIR"
