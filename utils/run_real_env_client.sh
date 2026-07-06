#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"
run_mode="${14:---run-once}"
artifact_root="${ROBODOJO_ARTIFACT_ROOT:-${TMPDIR:-/tmp}/robodojo-artifacts}"

IFS='|' read -r action_type base_cfg < <(ADDITIONAL_INFO="${additional_info}" python - <<'PY'
import os

fields = {}
for part in os.environ.get("ADDITIONAL_INFO", "").split(","):
    if "=" not in part:
        continue
    key, value = part.split("=", 1)
    fields[key.strip()] = value.strip()
print(fields.get("action_type", "") + "|" + fields.get("base_cfg", ""))
PY
)
base_cfg="${base_cfg:-${ROBODOJO_BASE_CFG:-}}"

echo -e "\033[31m[WARN] Real-world evaluation is not supported in the open-source release; attempting real env client startup anyway.\033[0m" >&2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port} (real env)...\033[0m"

export PYTHONPATH="${root_dir}/src:${root_dir}/XPolicyLab/integrations:${root_dir}/XPolicyLab:${root_dir}:${PYTHONPATH:-}"

CLIENT_ARGS=(
    --bench_name "${bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --protocol "${protocol}"
    --host "${policy_server_ip}"
    --port "${free_port}"
    --eval_batch "${eval_batch}"
    --eval-env-type real_world
    --root-dir "${root_dir}"
)

if [[ "${run_mode}" == "--run-once" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real requires daemon mode; set env_client_mode: daemon in deploy.yml"
    exit 1
fi

if [[ -z "${action_type}" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real requires action_type in additional_info (e.g. action_type=ee)" >&2
    exit 1
fi

if [[ -z "${base_cfg}" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real requires base_cfg in additional_info (e.g. base_cfg=arx_x5_station) or ROBODOJO_BASE_CFG" >&2
    exit 1
fi

PYTHONWARNINGS=ignore::UserWarning \
python -m eval_station.servers.env_client_server \
    "${CLIENT_ARGS[@]}" \
    --action-type "${action_type}" \
    --base-cfg "${base_cfg}" \
    --serve-host 0.0.0.0 \
    --serve-port 19200 \
    --artifact-root "${artifact_root}"
