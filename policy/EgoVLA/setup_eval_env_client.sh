#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name is required}
task_name=${2:?task_name is required}
ckpt_name=${3:?finalized SparkArena checkpoint is required}
env_cfg_type=${4:?env_cfg_type is required}
action_type=${5:?action_type is required}
seed=${6:?seed is required}
env_gpu_id=${7:?env gpu id is required}
eval_env_conda_env=${8:?eval env is required}
additional_info=${9:-}
policy_server_port=${10:?port is required}
policy_server_ip=${11:-localhost}

[[ "${bench_name}" == "SparkArena" ]] || {
  echo "SparkArena EgoVLA simulation client requires bench_name=SparkArena" >&2; exit 2;
}
[[ "${env_cfg_type}" == "tianji_marvin_wuji" && "${action_type}" == "joint" ]] || {
  echo "SparkArena EgoVLA evaluation requires tianji_marvin_wuji/joint" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_ROOT="${EGOVLA_EVAL_ROOT:-${EVAL_MAIN_ROOT:-}}"
[[ -n "${EVAL_ROOT}" ]] || { echo "set EGOVLA_EVAL_ROOT to the evaluator checkout" >&2; exit 2; }
if [[ "${EVAL_ROOT}" != /* ]]; then
  echo "EGOVLA_EVAL_ROOT must be an absolute path: ${EVAL_ROOT}" >&2
  exit 2
fi
[[ -d "${EVAL_ROOT}" ]] || {
  echo "EgoVLA evaluator root does not exist: ${EVAL_ROOT}" >&2; exit 2;
}
EVAL_ROOT="$(cd "${EVAL_ROOT}" && pwd -P)"
EVAL_XPL_ROOT="${EVAL_ROOT}/XPolicyLab"
UTILS_DIR="${EVAL_XPL_ROOT}/utils"
EVAL_POLICY_DIR="${EVAL_XPL_ROOT}/policy/egovla"

required_eval_files=(
  "${EVAL_ROOT}/scripts/eval_policy.sh"
  "${EVAL_ROOT}/env_cfg/${env_cfg_type}.yml"
  "${EVAL_XPL_ROOT}/XPolicyLab.py"
  "${UTILS_DIR}/setup_env_client.sh"
  "${EVAL_POLICY_DIR}/__init__.py"
  "${EVAL_POLICY_DIR}/deploy.py"
)
for required_file in "${required_eval_files[@]}"; do
  [[ -f "${required_file}" ]] || {
    echo "incomplete SparkArena evaluator checkout; missing regular file: ${required_file}" >&2
    exit 2
  }
done
DEPLOY_FILE="${SCRIPT_DIR}/deploy.yml"
POLICY_PYTHON="${EGOVLA_PYTHON_BIN:-}"
if [[ -z "${POLICY_PYTHON}" && -n "${EGOVLA_CONDA_ENV:-}" && -x "${EGOVLA_CONDA_ENV}/bin/python" ]]; then
  POLICY_PYTHON="${EGOVLA_CONDA_ENV}/bin/python"
fi
if [[ -z "${POLICY_PYTHON}" ]]; then
  POLICY_PYTHON="$(command -v python3 || command -v python || true)"
fi

[[ -f "${DEPLOY_FILE}" ]] || { echo "missing policy deploy contract: ${DEPLOY_FILE}" >&2; exit 2; }
[[ -x "${POLICY_PYTHON}" ]] || { echo "missing policy Python: ${POLICY_PYTHON}" >&2; exit 2; }
export EGOVLA_STRICT_ACTION_CONTRACT=1

# Verify the SparkArena sidecar/model hashes, deploy contract, and listening
# endpoint before allowing the environment client to send an observation.
PYTHONPATH="${SCRIPT_DIR}/.deps:${SCRIPT_DIR}/EgoVLA_Release/VILA:${SCRIPT_DIR}/EgoVLA_Release:${SCRIPT_DIR}:${POLICY_XPL_ROOT}" \
  "${POLICY_PYTHON}" - "${ckpt_name}" "${DEPLOY_FILE}" "${policy_server_ip}" "${policy_server_port}" <<'PY'
import socket
import sys
from pathlib import Path

import yaml

from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.utils.sparkarena_provenance import verify_sparkarena_checkpoint

checkpoint, deploy_path, host, port_text = sys.argv[1:]
checked = verify_sparkarena_checkpoint(checkpoint)
deploy = yaml.safe_load(Path(deploy_path).read_text(encoding="utf-8"))
expected = {
    "protocol": "ws",
    "bench_name": "SparkArena",
    "env_cfg_type": "tianji_marvin_wuji",
    "action_type": "joint",
    "runtime_module": "XPolicyLab.policy.EgoVLA.model",
    "runtime_class": "EgoVLASparkArenaInference",
    "camera_key": "cam_head",
    "raw_input_resolution": [480, 640],
    "input_resolution": [384, 384],
    "input_color_order": "RGB",
    "image_geometry": "full_frame_resize_480x640_to_384x384",
    "channel_swap": False,
    "eval_batch": True,
    "history_length": 6,
    "history_stride": 5,
}
bad = {key: (deploy.get(key), value) for key, value in expected.items() if deploy.get(key) != value}
if bad:
    raise ValueError(f"SparkArena deploy contract mismatch: {bad}")
port = int(port_text)
if not host or not 1 <= port <= 65535:
    raise ValueError(f"invalid policy endpoint {host!r}:{port}")
with socket.create_connection((host, port), timeout=5.0):
    pass
print(f"[egovla-sparkarena] client preflight ok: {checked['root']} -> {host}:{port}")
PY

# Activate the requested evaluator only after the policy-side verification.
if [[ "${eval_env_conda_env}" != "uv" && "${eval_env_conda_env}" != "none" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${eval_env_conda_env}"
  elif command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      source "${conda_base}/etc/profile.d/conda.sh"
      conda activate "${eval_env_conda_env}"
    fi
  elif [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    export PATH="${eval_env_conda_env}/bin:${PATH}"
  fi
fi

CLIENT_PYTHONPATH="${EVAL_XPL_ROOT}:${EVAL_ROOT}"
env PYTHONPATH="${CLIENT_PYTHONPATH}" bash "${UTILS_DIR}/setup_env_client.sh" \
  "${UTILS_DIR}" "${DEPLOY_FILE}" "${eval_env_conda_env}" \
  "${policy_server_port}" "${bench_name}" "${task_name}" \
  "${env_cfg_type}" egovla "${additional_info}" "${EVAL_ROOT}" \
  "${seed}" "${env_gpu_id}" "${policy_server_ip}"
