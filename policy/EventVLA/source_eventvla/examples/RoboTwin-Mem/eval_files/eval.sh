#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTVLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$EVENTVLA_ROOT/.." && pwd)"

resolve_default_robotwin_mem_root() {
  local candidate
  for candidate in \
    "${REPO_ROOT}/RoboTwin-Mem" \
    "${REPO_ROOT}/RoboTwin"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

policy_name="model2robotwin_mem_interface"
task_name=${1:-}
task_config=${2:-}
ckpt_setting=${3:-eventvla_demo}
seed=${4:-0}
gpu_id=${5:-0}

if [[ -z "${task_name}" || -z "${task_config}" ]]; then
  echo "[eval][error] Missing task arguments."
  echo "Usage: bash examples/RoboTwin-Mem/eval_files/eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]"
  exit 1
fi

ROBOTWIN_MEM_ROOT_PATH="${ROBOTWIN_MEM_ROOT:-}"
if [[ -z "${ROBOTWIN_MEM_ROOT_PATH}" ]]; then
  ROBOTWIN_MEM_ROOT_PATH="$(resolve_default_robotwin_mem_root || true)"
fi
if [[ -z "${ROBOTWIN_MEM_ROOT_PATH}" || ! -d "${ROBOTWIN_MEM_ROOT_PATH}" ]]; then
  echo "[eval][error] RoboTwin-Mem root not found."
  echo "[eval][error] Set ROBOTWIN_MEM_ROOT to your RoboTwin-Mem checkout."
  exit 1
fi

DEPLOY_POLICY_PATH="${DEPLOY_POLICY_PATH:-$SCRIPT_DIR/deploy_policy.yml}"
AUTO_START_POLICY_SERVER=${AUTO_START_POLICY_SERVER:-1}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-120}

if [[ -n "${ROBOTWIN_MEM_PYTHON:-}" ]]; then
  EVAL_PYTHON_BIN="${ROBOTWIN_MEM_PYTHON}"
elif [[ -n "${EVAL_PYTHON:-}" ]]; then
  EVAL_PYTHON_BIN="${EVAL_PYTHON}"
else
  EVAL_PYTHON_BIN="python"
fi

read_deploy_value() {
  local key="$1"
  awk -F': ' -v lookup_key="$key" '
    $1 == lookup_key {
      value=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "$DEPLOY_POLICY_PATH"
}

is_port_open() {
  local host="$1"
  local port="$2"
  (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_for_spawned_server() {
  local host="$1"
  local port="$2"
  local server_pid="$3"
  local timeout="$4"
  local elapsed=0

  while (( elapsed < timeout )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[eval][error] Policy server exited before becoming ready."
      return 1
    fi
    if is_port_open "${host}" "${port}"; then
      return 0
    fi
    sleep 1
    ((elapsed++))
  done

  echo "[eval][error] Timed out waiting for policy server on ${host}:${port}."
  return 1
}

wait_for_existing_server() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local elapsed=0

  while (( elapsed < timeout )); do
    if is_port_open "${host}" "${port}"; then
      return 0
    fi
    sleep 1
    ((elapsed++))
  done

  echo "[eval][error] Timed out waiting for an existing policy server on ${host}:${port}."
  return 1
}

cleanup_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}

if [[ ! -f "${DEPLOY_POLICY_PATH}" ]]; then
  echo "[eval][error] Missing deploy policy config: ${DEPLOY_POLICY_PATH}"
  exit 1
fi

policy_ckpt_path="${POLICY_CKPT_PATH:-$(read_deploy_value policy_ckpt_path)}"
host="${HOST:-$(read_deploy_value host)}"
port="${PORT:-$(read_deploy_value port)}"
unnorm_key="${UNNORM_KEY:-$(read_deploy_value unnorm_key)}"
action_mode="${ACTION_MODE:-$(read_deploy_value action_mode)}"
host=${host:-127.0.0.1}
port=${port:-5800}
unnorm_key=${unnorm_key:-robotwin_mem}
action_mode=${action_mode:-abs}

if [[ -z "${policy_ckpt_path}" ]]; then
  echo "[eval][error] Missing policy_ckpt_path in ${DEPLOY_POLICY_PATH}"
  exit 1
fi

if [[ ! -f "${policy_ckpt_path}" ]]; then
  echo "[eval][error] Checkpoint not found: ${policy_ckpt_path}"
  exit 1
fi

if [[ "${AUTO_START_POLICY_SERVER}" == "1" ]]; then
  policy_server_gpu_id=${POLICY_SERVER_GPU_ID:-$gpu_id}
  trap cleanup_server EXIT INT TERM
  echo "[eval] starting policy server from deploy_policy.yml ckpt=${policy_ckpt_path}"
  echo "[eval] server host=${host} port=${port} gpu=${policy_server_gpu_id}"
  bash "$SCRIPT_DIR/run_policy_server.sh" "${policy_ckpt_path}" "${policy_server_gpu_id}" "${port}" &
  server_pid=$!
  wait_for_spawned_server "${host}" "${port}" "${server_pid}" "${SERVER_READY_TIMEOUT}"
else
  echo "[eval] AUTO_START_POLICY_SERVER=0, expecting an already-running server on ${host}:${port}"
  echo "[eval] client will still use deploy_policy.yml ckpt=${policy_ckpt_path}"
  wait_for_existing_server "${host}" "${port}" "${SERVER_READY_TIMEOUT}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export PYTHONPATH="${ROBOTWIN_MEM_ROOT_PATH}:${EVENTVLA_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

cd "$ROBOTWIN_MEM_ROOT_PATH"

echo "PYTHONPATH: $PYTHONPATH"
echo "[eval] client/server checkpoint source=${policy_ckpt_path}"
echo "[eval] host=${host} port=${port}"
echo "[eval] python=${EVAL_PYTHON_BIN}"

PYTHONWARNINGS=ignore::UserWarning \
"${EVAL_PYTHON_BIN}" script/eval_policy.py --config "$DEPLOY_POLICY_PATH" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --host "${host}" \
    --port "${port}" \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --unnorm_key "${unnorm_key}" \
    --action_mode "\"${action_mode}\""
