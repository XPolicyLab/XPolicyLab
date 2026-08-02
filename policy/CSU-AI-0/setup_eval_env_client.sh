#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

CONDA_ENVS_ROOT="${CONDA_ENVS_ROOT:-/root/autodl-tmp/conda_envs}"
if [[ "${eval_env_conda_env}" = /* ]]; then
    EVAL_ENV_DIR="${eval_env_conda_env}"
else
    EVAL_ENV_DIR="${CONDA_ENVS_ROOT}/${eval_env_conda_env}"
fi
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ENV_DIR}/bin/python}"
[[ -x "${EVAL_PYTHON}" ]] || {
    echo "[CSU-AI-0][ERROR] eval python missing: ${EVAL_PYTHON}" >&2
    exit 1
}

eval_batch="$(${EVAL_PYTHON} - "${yaml_file}" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    data = yaml.safe_load(stream)
print(str(data.get("eval_batch", False)).lower())
PY
)"

TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/root/autodl-tmp/tmp/xdg-runtime}"
mkdir -p "${TMPDIR}" "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true

echo "[CSU-AI-0] client task=${task_name} action_type=${action_type} server=${policy_server_ip}:${policy_server_port}"

exec env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    EVAL_ENV_TYPE=sim \
    PATH="${EVAL_ENV_DIR}/bin:${PATH}" \
    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/XPolicyLab:${PYTHONPATH:-}" \
    TMPDIR="${TMPDIR}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost \
    bash "${ROOT_DIR}/scripts/eval_policy.sh" \
        --bench_name "${bench_name}" \
        --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" \
        --policy_name "${policy_name}" \
        --host "${policy_server_ip}" \
        --port "${policy_server_port}" \
        --protocol ws \
        --eval_batch "${eval_batch}" \
        --root_dir "${ROOT_DIR}" \
        --device_id "${env_gpu_id}" \
        --additional_info "${additional_info}" \
        --seed "${seed}"

