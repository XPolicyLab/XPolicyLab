#!/bin/bash
set -e

UTILS_DIR="${1}"
yaml_file="${2}"
eval_env_conda_env="${3}"
policy_server_port="${4}"
bench_name="${5}"
task_name="${6}"
env_cfg_type="${7}"
policy_name="${8}"
additional_info="${9}"
ROOT_DIR="${10}"
seed="${11}"
env_gpu_id="${12}"
policy_server_ip="${13:-localhost}"
protocol_override="${14:-}"

# shellcheck source=resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1

read eval_batch yaml_protocol env_client_mode < <(python - <<PY
import yaml
with open("${yaml_file}", "r") as f:
    data = yaml.safe_load(f)
print(
    str(data.get("eval_batch", False)).lower(),
    data.get("protocol", "ws"),
    data.get("env_client_mode", "run-once"),
)
PY
)
protocol="${protocol_override:-${yaml_protocol}}"

run_mode="${15:-${ROBODOJO_ENV_CLIENT_RUN_MODE:-${env_client_mode}}}"
if [[ "${run_mode}" == "run-once" ]]; then
    run_mode="--run-once"
fi

if [[ -z "${EVAL_ENV_TYPE:-}" ]]; then
    echo "[CLIENT] EVAL_ENV_TYPE=(default sim) -> ${eval_env_mode}"
else
    echo "[CLIENT] EVAL_ENV_TYPE=${EVAL_ENV_TYPE} -> ${eval_env_mode}"
fi

COMMON_ARGS=(
    "${eval_batch}"
    "${eval_env_conda_env}"
    "${policy_server_port}"
    "${bench_name}"
    "${task_name}"
    "${env_cfg_type}"
    "${policy_name}"
    "${additional_info}"
    "${ROOT_DIR}"
    "${seed}"
    "${env_gpu_id}"
    "${policy_server_ip}"
)

if [[ "${eval_env_mode}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}" "${run_mode}"
elif [[ "${eval_env_mode}" == "sim" ]]; then
    if [[ "${run_mode}" != "--run-once" ]]; then
        echo "[WARN] env_client_mode=daemon is not supported for EVAL_ENV_TYPE=sim; running the one-shot sim eval client instead." >&2
    fi
    bash "${UTILS_DIR}/run_sim_env_client.sh" "${COMMON_ARGS[@]}"
elif [[ "${eval_env_mode}" == "real_world" ]]; then
    echo -e "\033[31m[WARN] EVAL_ENV_TYPE=real: real-world evaluation is not supported in the open-source release; continuing to real env client.\033[0m" >&2
    bash "${UTILS_DIR}/run_real_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}" "${run_mode}"
else
    echo "[ERROR] Unknown eval env mode: ${eval_env_mode}" >&2
    exit 1
fi
