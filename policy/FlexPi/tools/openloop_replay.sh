#!/usr/bin/env bash
# Open-loop replay of dataset episodes through the FlexPi eval pipeline, in the
# same environment setup_eval_policy_server.sh gives the policy server.
#
#   bash openloop_replay.sh --dataset-root <34D LeRobot root> --ckpt-dir <run dir> --episodes 1200 1201
#   ... --set eef_frame=arm_base_grasp      # negative control: must get worse
#   ... --stride 32                         # the eval replan cadence
set -euo pipefail

# The benchmark checkout this adapter sits in (<bench>/XPolicyLab/policy/FlexPi/tools/).
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
FLEXPI_DIR="${BENCH_ROOT}/XPolicyLab/policy/FlexPi/FlexPi"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV:-flexpi}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTHONPATH="${BENCH_ROOT}:${FLEXPI_DIR}:${FLEXPI_DIR}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FLEXPI_DIR}/checkpoints}" \
    python -u "${BENCH_ROOT}/XPolicyLab/policy/FlexPi/tools/openloop_replay.py" "$@"
