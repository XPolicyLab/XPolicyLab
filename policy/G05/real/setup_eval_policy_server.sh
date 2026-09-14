#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_python=${8:?policy Python executable required}
policy_server_port=${9:?policy_server_port required}
policy_server_host=${10:-0.0.0.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_PATH="${G05_REAL_CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/${env_cfg_type}}"
RUNTIME_ROOT="${CHECKPOINT_PATH}/inference_runtime"

case "${env_cfg_type}" in
  arx_x5) embodiment=robodojo_arx_x5 ;;
  piper) embodiment=robodojo_piper ;;
  piper_x) embodiment=robodojo_piper_x ;;
  *) echo "unsupported RoboDojo real robot: ${env_cfg_type}" >&2; exit 2 ;;
esac

[[ "${action_type}" == joint ]] || { echo "real policy requires action_type=joint" >&2; exit 2; }
[[ -x "${policy_python}" ]] || { echo "invalid policy Python: ${policy_python}" >&2; exit 2; }

"${policy_python}" "${SCRIPT_DIR}/validate_bundle.py" \
  --bundle "${CHECKPOINT_PATH}" --embodiment "${embodiment}"

exec env \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${XPL_ROOT}:${RUNTIME_ROOT}/src:${RUNTIME_ROOT}:${RUNTIME_ROOT}/third_party/galaxea_dataset/src:${RUNTIME_ROOT}/third_party/galaxea_tokenizer/src:${PYTHONPATH:-}" \
  "${policy_python}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides \
      host="${policy_server_host}" \
      port="${policy_server_port}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      action_type="${action_type}" \
      checkpoint_path="${CHECKPOINT_PATH}" \
      eval_embodiment="${embodiment}"
