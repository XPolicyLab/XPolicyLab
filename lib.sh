#!/usr/bin/env bash
set -euo pipefail

# ---------- colors ----------
_c_reset="\033[0m"
_c_red="\033[31m"
_c_green="\033[32m"
_c_yellow="\033[33m"
_c_blue="\033[34m"

logi() { echo -e "${_c_yellow}[INFO]  $*${_c_reset}"; }
logs() { echo -e "${_c_green}[SERV]  $*${_c_reset}"; }
logc() { echo -e "${_c_blue}[CLNT]  $*${_c_reset}"; }
loge() { echo -e "${_c_red}[ERR ]  $*${_c_reset}"; }

die() { loge "$*"; exit 2; }

# ---------- ports ----------
get_free_port() {
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PY
}

# ---------- conda ----------
conda_bootstrap() {
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

conda_on()  { conda activate "$1"; }
conda_off() { conda deactivate >/dev/null 2>&1 || true; }

# ---------- background process mgmt ----------
_BG_PIDS=()

bg_run() {  # usage: bg_run <cmd...>   -> echo pid
  "$@" &
  local pid=$!
  _BG_PIDS+=("$pid")
  echo "$pid"
}

cleanup_bg() {
  # kill in reverse order
  for ((i=${#_BG_PIDS[@]}-1; i>=0; i--)); do
    local pid="${_BG_PIDS[i]}"
    if kill -0 "$pid" 2>/dev/null; then
      logi "Cleanup: killing PID=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done
}

setup_trap_cleanup() {
  trap 'cleanup_bg' EXIT INT TERM
}