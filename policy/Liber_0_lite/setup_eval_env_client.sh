#!/bin/bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${EVAL_ENV_TYPE:-sim}" == "debug" ]]; then
    client_python="${eval_env_conda_env}/bin/python"
    [[ -x "$client_python" ]] || { echo "Debug mode requires an environment prefix as argument 8." >&2; exit 1; }
    export PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    eval_batch=$("$client_python" -c 'import sys,yaml; print(str(yaml.safe_load(open(sys.argv[1]))["eval_batch"]).lower())' "$yaml_file")
    exec "$client_python" "${UTILS_DIR}/debug_env_client.py" \
        --bench_name "$bench_name" --task_name "$task_name" \
        --env_cfg_type "$env_cfg_type" --policy_name "$policy_name" \
        --protocol ws --host "$policy_server_ip" --port "$policy_server_port" --eval_batch "$eval_batch"
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
