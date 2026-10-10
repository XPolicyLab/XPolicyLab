#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval_hf_robodojo.sh <oft|groot|pi_v3> <task_name> <seed> <policy_gpu> <env_gpu> \
    <starvla_env> <robodojo_sim_env> [eval_num|native]

Example (10-episode build_tower verification):
  bash scripts/eval_hf_robodojo.sh oft build_tower 0 0 1 /path/to/starvla-env /path/to/robodojo-env 10

The script downloads the pinned Hugging Face snapshot, verifies its framework,
normalization files, size and SHA256, then starts the vendored XPolicyLab
StarVLA server. Do not start a second upstream StarVLA server manually.

Optional environment variables:
  STARVLA_HF_ROOT             Download root (default: policy/StarVLA/checkpoints/huggingface)
  STARVLA_HF_LOCAL_FILES_ONLY Set to 1 for offline/cache-only use
  STARVLA_HF_VERIFY_ONLY      Set to 1 for an already-materialized run directory
  STARVLA_HF_SKIP_WEIGHT_HASH Set to 1 to skip the 10 GB weight SHA after checking its size
  STARVLA_HF_DRY_RUN          Set to 1 to validate and print the RoboDojo command only
  STARVLA_BASE_VLM            Optional local Qwen3-VL-4B-Instruct path for offline use
  STARVLA_CPU_THREADS         Policy-server CPU import threads (default: 8)
  STARVLA_ROBODOJO_NUM_ENVS   Isaac environments in parallel (default: 1; archived PI protocol: 5)
  STARVLA_ROBODOJO_KIT_ARGS   Override the safe single-GPU Isaac Kit arguments
EOF
}

if [[ $# -lt 7 || $# -gt 8 ]]; then
    usage >&2
    exit 2
fi

variant_input=$1
task_name=$2
seed=$3
policy_gpu_id=$4
env_gpu_id=$5
policy_env_ref=$6
eval_env_ref=$7
eval_num=${8:-${EVAL_NUM:-10}}

case "${variant_input,,}" in
    oft|qwen_oft|qwenoft)
        variant="oft"
        ckpt_label="hf_qwenoft_100k"
        ;;
    groot|gr00t|qwen_groot|qwengroot)
        variant="groot"
        ckpt_label="hf_qwengroot_130k"
        ;;
    pi|pi_v3|piv3|qwenpi_v3)
        variant="pi_v3"
        ckpt_label="hf_qwenpi_v3"
        ;;
    *)
        echo "[starVLA HF][ERROR] unknown variant: ${variant_input}; choose oft, groot, or pi_v3" >&2
        exit 2
        ;;
esac

# This entry point launches exactly one policy process and one simulator
# process.  Cluster login shells can retain torchrun/Accelerate variables from
# a training allocation; PI-v3 imports DistributedOverwatch and will otherwise
# wait forever for ranks that are not part of this evaluation.
starvla_distributed_env_vars=(
    RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
    PMI_RANK PMI_SIZE MPI_LOCALRANKID MPI_LOCALNRANKS
    OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
    OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
    MV2_COMM_WORLD_RANK MV2_COMM_WORLD_SIZE
    MV2_COMM_WORLD_LOCAL_RANK MV2_COMM_WORLD_LOCAL_SIZE
)
stale_distributed_env_vars=()
for variable_name in "${starvla_distributed_env_vars[@]}"; do
    if [[ -v "${variable_name}" ]]; then
        stale_distributed_env_vars+=("${variable_name}")
    fi
    unset "${variable_name}"
done
if (( ${#stale_distributed_env_vars[@]} > 0 )); then
    echo "[starVLA HF] cleared stale distributed-training variables: ${stale_distributed_env_vars[*]}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
hf_root="${STARVLA_HF_ROOT:-${POLICY_DIR}/checkpoints/huggingface}"
run_dir="${hf_root}/${variant}"

if [[ ! -f "${BENCH_ROOT}/scripts/robodojo.sh" ]]; then
    echo "[starVLA HF][ERROR] XPolicyLab must be checked out as RoboDojo/XPolicyLab." >&2
    echo "[starVLA HF][ERROR] Missing ${BENCH_ROOT}/scripts/robodojo.sh" >&2
    exit 1
fi

required_asset_paths=(
    "Assets/Robots/x5/robot_config.yml"
    "Assets/Object/RoboDojo"
    "Assets/Eval_Layout/RoboDojo"
    "Assets/Material"
)
missing_asset_paths=()
for asset_path in "${required_asset_paths[@]}"; do
    if [[ ! -e "${BENCH_ROOT}/${asset_path}" ]]; then
        missing_asset_paths+=("${asset_path}")
    fi
done
if (( ${#missing_asset_paths[@]} > 0 )); then
    echo "[starVLA HF][ERROR] RoboDojo's official benchmark assets are incomplete:" >&2
    printf '[starVLA HF][ERROR]   %s\n' "${missing_asset_paths[@]}" >&2
    echo "[starVLA HF][ERROR] Run this once from the RoboDojo root:" >&2
    echo "[starVLA HF][ERROR]   bash scripts/init_assets.sh" >&2
    exit 1
fi

if [[ "${eval_env_ref}" == */* && ! -x "${eval_env_ref}/bin/python" ]]; then
    echo "[starVLA HF][ERROR] RoboDojo sim environment is not a valid environment prefix: ${eval_env_ref}" >&2
    echo "[starVLA HF][ERROR] Pass a Conda environment name or a prefix containing bin/python, not the RoboDojo source directory." >&2
    exit 2
fi

# A prefix path lets us validate the simulator's dynamic-library view before
# loading the 10 GB policy checkpoint.  Conda-name environments are checked by
# the client helper immediately after activation.
if [[ -x "${eval_env_ref}/bin/python" ]]; then
    "${eval_env_ref}/bin/python" "${SCRIPT_DIR}/check_isaac_runtime.py"
    export STARVLA_ISAAC_RUNTIME_PREFLIGHTED=1
fi

prepare_args=(
    --variant "${variant}"
    --output-dir "${run_dir}"
)
if [[ "${STARVLA_HF_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    prepare_args+=(--local-files-only)
fi
if [[ "${STARVLA_HF_VERIFY_ONLY:-0}" == "1" ]]; then
    prepare_args+=(--verify-only)
fi
if [[ "${STARVLA_HF_SKIP_WEIGHT_HASH:-0}" == "1" ]]; then
    prepare_args+=(--skip-weight-hash)
fi

echo "[starVLA HF] XPolicyLab commit: $(git -C "${XPL_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[starVLA HF] RoboDojo commit: $(git -C "${BENCH_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[starVLA HF] preparing variant=${variant}, task=${task_name}, eval_num=${eval_num}"

# Keep the resolved policy environment active in this parent shell.  Current
# XPolicyLab's sim-client bootstrap reads deploy.yml with `python` before it
# activates the RoboDojo Conda environment, so a path-only venv must remain on
# PATH until that hand-off.
# shellcheck source=activate_policy_env.sh
source "${SCRIPT_DIR}/activate_policy_env.sh"
starvla_activate_policy_env "${policy_env_ref}"
checkpoint_path="$(
    "${STARVLA_POLICY_PYTHON}" "${SCRIPT_DIR}/prepare_hf_checkpoint.py" "${prepare_args[@]}" \
        | tail -n 1
)"

if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[starVLA HF][ERROR] validated checkpoint was not found: ${checkpoint_path}" >&2
    exit 1
fi

export STARVLA_CKPT_PATH="${checkpoint_path}"
export STARVLA_INCLUDE_STATE="True"
export STARVLA_UNNORM_KEY="arx_x5"
export STARVLA_EXECUTE_HORIZON="16"
export STARVLA_IMAGE_SIZE="[224,224]"
export STARVLA_DIRECT_ROBODOJO_CLIENT="1"
if [[ "${variant}" == "pi_v3" ]]; then
    export STARVLA_REQUIRED_PI_V3_FORWARD="canonical_interleaved"
else
    unset STARVLA_REQUIRED_PI_V3_FORWARD || true
fi

unset EVAL_NUM || true

echo "[starVLA HF] runtime: RGB, raw 14D state -> server q99 normalization, h50/execute16"
echo "[starVLA HF] checkpoint: ${STARVLA_CKPT_PATH}"

robodojo_args=(
    eval
    --policy-dir "${POLICY_DIR}"
    --task "${task_name}"
    --ckpt "${ckpt_label}"
    --env-cfg arx_x5
    --action-type joint
    --seed "${seed}"
    --policy-gpu "${policy_gpu_id}"
    --env-gpu "${env_gpu_id}"
    --policy-env "${policy_env_ref}"
    --eval-env "${eval_env_ref}"
    --eval-num "${eval_num}"
)
if [[ "${STARVLA_HF_DRY_RUN:-0}" == "1" ]]; then
    robodojo_args+=(--dry-run)
fi

exec bash "${BENCH_ROOT}/scripts/robodojo.sh" "${robodojo_args[@]}"
