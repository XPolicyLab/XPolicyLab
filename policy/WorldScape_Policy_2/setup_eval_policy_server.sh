#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?}
task_name=${2:?}
ckpt_name=${3:?}
env_cfg_type=${4:?}
action_type=${5:?}
seed=${6:?}
policy_gpu_id=${7:?}
policy_conda_env=${8:?}
policy_server_port=${9:?}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${WSP_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
[[ "${yaml_file}" = /* ]] || yaml_file="${SCRIPT_DIR}/${yaml_file}"
export WSP_DEPLOY_CONFIG="${yaml_file}"

if [[ "${action_type}" != "joint" ]]; then
    echo "[SERVER][ERROR] ${policy_name} requires action_type=joint." >&2
    exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# Resolve after activating the policy environment (PyYAML is a policy dependency).
# Only explicitly set legacy env variables override the selected YAML.
resolved=$(python "${SCRIPT_DIR}/xpolicylab_adapter/runtime_config.py" "${yaml_file}" "${policy_gpu_id}")
mapfile -t config_lines <<< "${resolved}"
worldscape_root="${config_lines[0]}"
echo "[SERVER] config=${yaml_file} ${config_lines[1]}"
overrides=(
    "port=${policy_server_port}" "host=${policy_server_host}"
    "bench_name=${bench_name}" "task_name=${task_name}" "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}" "action_type=${action_type}" "seed=${seed}"
    "policy_name=${policy_name}" "worldscape_root=${worldscape_root}"
)
for ((i=2; i<${#config_lines[@]}; i++)); do
    overrides+=("${config_lines[i]}")
done
exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${BENCH_ROOT}:${worldscape_root}:${worldscape_root}/src:${PYTHONPATH:-}" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" --overrides "${overrides[@]}"
