#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_conda_env=${8:?policy conda env required}
policy_server_port=${9:?policy server port required}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
yaml_file="${OPENDM_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
if [[ "${yaml_file}" != /* ]]; then
    yaml_file="${SCRIPT_DIR}/${yaml_file}"
fi
if [[ ! -f "${yaml_file}" ]]; then
    echo "[SERVER] OpenDM deploy config not found: ${yaml_file}" >&2
    exit 1
fi

if [[ "${action_type}" != "joint" ]]; then
    echo "OpenDM currently supports action_type=joint, got ${action_type}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

OVERRIDES=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name=OpenDM
    action_type="${action_type}"
)

[[ -z "${MODEL_PATH:-}" ]] || OVERRIDES+=(model_path="${MODEL_PATH}")
[[ -z "${NORM_STATS_PATH:-}" ]] || OVERRIDES+=(norm_stats_path="${NORM_STATS_PATH}")
[[ -z "${OPENDM_ACTION_STEPS:-}" ]] || OVERRIDES+=(action_steps="${OPENDM_ACTION_STEPS}")
[[ -z "${OPENDM_DIFFUSION_STEPS:-}" ]] || OVERRIDES+=(diffusion_steps="${OPENDM_DIFFUSION_STEPS}")
[[ -z "${OPENDM_DIFFUSION_NOISE_SEED:-}" ]] || OVERRIDES+=(diffusion_noise_seed="${OPENDM_DIFFUSION_NOISE_SEED}")

echo "[SERVER] policy=OpenDM task=${task_name} port=${policy_server_port} config=${yaml_file}"
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${OVERRIDES[@]}"
