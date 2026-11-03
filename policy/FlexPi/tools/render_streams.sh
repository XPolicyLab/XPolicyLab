#!/usr/bin/env bash
# Render a FLEXPI_DUMP_STREAMS dump (pred vs actual future video / DINO / pointmap),
# in the same environment setup_eval_policy_server.sh gives the policy server.
#
#   bash render_streams.sh <run>/streams --ckpt-dir <run dir> --step 110000
set -euo pipefail

# The benchmark checkout this adapter sits in (<bench>/XPolicyLab/policy/FlexPi/tools/).
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
FLEXPI_DIR="${BENCH_ROOT}/XPolicyLab/policy/FlexPi/FlexPi"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${POLICY_ENV:-flexpi}"
# The renderer loads the policy itself; it must not dump anything of its own.
unset FLEXPI_DUMP_STREAMS FLEXPI_DUMP_ACTIONS FLEXPI_DUMP_OBS

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTHONPATH="${BENCH_ROOT}:${FLEXPI_DIR}:${FLEXPI_DIR}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FLEXPI_DIR}/checkpoints}" \
    python -u "${BENCH_ROOT}/XPolicyLab/policy/FlexPi/tools/render_streams.py" "$@"
