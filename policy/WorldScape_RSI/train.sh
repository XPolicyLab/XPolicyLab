#!/usr/bin/env bash
set -euo pipefail

# Public entry point: supply machine-specific paths through the environment.
# Keep personal launchers outside this adapter's release directory.
usage() {
    cat <<'EOF'
Usage:
  bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

Post-trains RoboDojo Flash (H24/C1) or Memory (H24/C24).
Required: DATA_ROOT, ROBODOJO_ZSCORE_STATS, ROBODOJO_INDEX_PATH,
PRETRAINED_MODEL_PATH. Set ROBODOJO_MODEL=flash (default) or memory.
WSP_INITIALIZATION=checkpoint initializes from the pretrained model.
Use RUN_NAME and OUTPUT_DIR for external run naming and checkpoint output.
WSP_PYTHON or CONDA_ENV selects Python; gpu_id sets CUDA_VISIBLE_DEVICES.
Extra arguments are forwarded as configuration overrides.

Example, after configuring data, statistics, index and pretrained weights:
  bash train.sh RoboDojo baseline arx_x5 joint 62 0,1,2,3

EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [ "$#" -lt 6 ]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[WorldScape_RSI-train] only bench_name=RoboDojo is supported" >&2
    exit 2
fi
recipe="${worldscape_root}/recipes/posttrain/posttrain_robodojo.sh"
if [[ ! -f "${recipe}" ]]; then
    echo "[WorldScape_RSI-train] recipe not found: ${recipe}" >&2
    exit 1
fi
if [[ "${action_type}" != "joint" ]]; then
    echo "[WorldScape_RSI-train] RoboDojo requires action_type=joint" >&2
    exit 1
fi
action_dim="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "$(dirname "${XPL_ROOT}")" "${env_cfg_type}")"
if [[ "${action_dim}" != "14" ]]; then
    echo "[WorldScape_RSI-train] RoboDojo requires a 14-dim robot, got ${action_dim}" >&2
    exit 1
fi
run_name="${RUN_NAME:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
output_dir="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/${run_name}}"
cd "${worldscape_root}"
exec env \
    RUN_NAME="${run_name}" \
    OUTPUT_DIR="${output_dir}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    SEED="${seed}" \
    bash "${recipe}" "$@"
