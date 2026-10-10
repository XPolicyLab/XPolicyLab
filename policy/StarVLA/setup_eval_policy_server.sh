#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <policy_server_port> [policy_server_host]"
    exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
# Run-dir resolution: the default run dir is the 5-tuple
# <bench>-<ckpt>-<env>-<action>-<seed> under results/Checkpoints/ or checkpoints/.
# ckpt_name may also be a path (absolute, or relative containing '/', resolved
# under this policy dir), and checkpoints/<ckpt_name> is kept as a verbatim
# fallback. STARVLA_CKPT_PATH stays the highest-priority explicit override.
run_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
result_run_dir="${SCRIPT_DIR}/results/Checkpoints/${ckpt_name}"
local_run_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"

if [[ "${ckpt_name}" == /* ]]; then
    candidate_run_dirs=("${ckpt_name}")
elif [[ "${ckpt_name}" == */* ]]; then
    candidate_run_dirs=("${SCRIPT_DIR}/${ckpt_name}")
else
    candidate_run_dirs=(
        "${SCRIPT_DIR}/results/Checkpoints/${run_id}"
        "${SCRIPT_DIR}/checkpoints/${run_id}"
        "${result_run_dir}"
        "${local_run_dir}"
    )
fi

resolve_starvla_checkpoint() {
    local run_dir=$1
    local candidates=()

    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi

    shopt -s nullglob
    candidates=(
        "${run_dir}"/final_model/*.pt
        "${run_dir}"/final_model/*.safetensors
        "${run_dir}"/checkpoints/*.pt
        "${run_dir}"/checkpoints/*.safetensors
    )
    shopt -u nullglob

    if (( ${#candidates[@]} > 0 )); then
        printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
        return 0
    fi

    return 1
}

checkpoint_path="${STARVLA_CKPT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    for run_dir in "${candidate_run_dirs[@]}"; do
        if checkpoint_path=$(resolve_starvla_checkpoint "${run_dir}"); then
            break
        fi
        checkpoint_path=""
    done
fi
if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="${local_run_dir}/checkpoints/<checkpoint>.pt"
fi
if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[SERVER][ERROR] checkpoint file does not exist: ${checkpoint_path}" >&2
    echo "[SERVER][ERROR] set STARVLA_CKPT_PATH=/path/to/pytorch_model.pt to override checkpoint lookup" >&2
    echo "[SERVER][ERROR] expected a .pt or .safetensors file under one of:" >&2
    echo "[SERVER][ERROR]   ${result_run_dir}/final_model/" >&2
    echo "[SERVER][ERROR]   ${result_run_dir}/checkpoints/" >&2
    echo "[SERVER][ERROR]   ${local_run_dir}/final_model/" >&2
    echo "[SERVER][ERROR]   ${local_run_dir}/checkpoints/" >&2
    exit 1
fi
# Keep the final checkpoint symlink unresolved. Hugging Face snapshots place
# the weight under <snapshot>/checkpoints/ as a symlink into the blob cache;
# resolving that symlink loses the adjacent config YAML and normalization
# statistics that StarVLA intentionally reads from <snapshot>/.
checkpoint_parent="$(cd -L "$(dirname "${checkpoint_path}")" && pwd -L)"
checkpoint_path="${checkpoint_parent}/$(basename "${checkpoint_path}")"
checkpoint_parent_name="$(basename "${checkpoint_parent}")"
if [[ "${checkpoint_parent_name}" == "checkpoints" || "${checkpoint_parent_name}" == "final_model" ]]; then
    checkpoint_run_dir="$(dirname "${checkpoint_parent}")"
else
    checkpoint_run_dir="${checkpoint_parent}"
fi
for sidecar in config.yaml config.full.yaml dataset_statistics.json; do
    if [[ ! -f "${checkpoint_run_dir}/${sidecar}" ]]; then
        echo "[SERVER][ERROR] missing checkpoint sidecar: ${checkpoint_run_dir}/${sidecar}" >&2
        echo "[SERVER][ERROR] keep the complete Hugging Face run-directory layout; do not pass a blob-cache path" >&2
        exit 1
    fi
done
echo "[SERVER] resolved StarVLA checkpoint: ${checkpoint_path}"
starvla_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
starvla_server_host="127.0.0.1"
starvla_include_state="${STARVLA_INCLUDE_STATE:-auto}"
starvla_unnorm_key="${STARVLA_UNNORM_KEY:-arx_x5}"
starvla_execute_horizon="${STARVLA_EXECUTE_HORIZON:-16}"
starvla_image_size="${STARVLA_IMAGE_SIZE:-[224,224]}"
starvla_cpu_threads="${STARVLA_CPU_THREADS:-8}"
starvla_required_pi_v3_forward="${STARVLA_REQUIRED_PI_V3_FORWARD:-none}"

# Defensive counterpart to scripts/eval_hf_robodojo.sh for callers that use the
# generic XPolicyLab launcher directly.  This server is single-process; stale
# torchrun/MPI variables make StarVLA's DistributedOverwatch wait for peers.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
unset PMI_RANK PMI_SIZE MPI_LOCALRANKID MPI_LOCALNRANKS
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
unset MV2_COMM_WORLD_RANK MV2_COMM_WORLD_SIZE
unset MV2_COMM_WORLD_LOCAL_RANK MV2_COMM_WORLD_LOCAL_SIZE

if ! [[ "${starvla_cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[SERVER][ERROR] STARVLA_CPU_THREADS must be a positive integer: ${starvla_cpu_threads}" >&2
    exit 2
fi

# Accelerate otherwise chooses half of all visible host CPUs for each Python
# process.  On large shared nodes the StarVLA websocket server and the XPolicy
# adapter import concurrently, and that default can turn a ~30 s PI-v3 import
# into a policy-server timeout.  Respect explicit BLAS settings, with a modest
# reproducible fallback for evaluation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${starvla_cpu_threads}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${starvla_cpu_threads}}"

cleanup() {
    if [[ -n "${STARVLA_SERVER_PID:-}" ]]; then
        echo "[SERVER] kill StarVLA websocket server ${STARVLA_SERVER_PID}"
        kill "${STARVLA_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, starvla_port=${starvla_server_port}"
echo "[SERVER] starVLA overrides: include_state=${starvla_include_state}, unnorm_key=${starvla_unnorm_key}, execute_horizon=${starvla_execute_horizon}, image_size=${starvla_image_size}"
echo "[SERVER] required PI-v3 forward: ${starvla_required_pi_v3_forward}"
echo "[SERVER] CPU thread limits: OMP_NUM_THREADS=${OMP_NUM_THREADS}, MKL_NUM_THREADS=${MKL_NUM_THREADS}"
if [[ -n "${STARVLA_BASE_VLM:-}" ]]; then
    echo "[SERVER] StarVLA base VLM override: ${STARVLA_BASE_VLM}"
fi

# Accept the venv path used by the released StarVLA setup as well as a Conda
# environment name.  This keeps the documented eval command identical across
# the two installation styles.
# shellcheck source=scripts/activate_policy_env.sh
source "${SCRIPT_DIR}/scripts/activate_policy_env.sh"
starvla_activate_policy_env "${policy_conda_env}"

(
    cd "${STARVLA_ROOT}"
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${STARVLA_POLICY_PYTHON}" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${checkpoint_path}" \
        --port "${starvla_server_port}" \
        --use_bf16
) &
STARVLA_SERVER_PID=$!

sleep 6

PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
"${STARVLA_POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${policy_server_port}" \
        host="${policy_server_host}" \
        bench_name="${bench_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        checkpoint_path="${checkpoint_path}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        include_state="${starvla_include_state}" \
        unnorm_key="${starvla_unnorm_key}" \
        execute_horizon="${starvla_execute_horizon}" \
        image_size="${starvla_image_size}" \
        required_pi_v3_forward="${starvla_required_pi_v3_forward}" \
        starvla_root="${STARVLA_ROOT}" \
        starvla_server_host="${starvla_server_host}" \
        starvla_server_port="${starvla_server_port}"
