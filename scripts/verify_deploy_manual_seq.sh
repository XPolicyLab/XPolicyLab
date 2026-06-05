#!/usr/bin/env bash
# 按顺序手动验证：分别起 setup_eval_policy_server + setup_eval_env_client（不经 eval.sh）
set -uo pipefail

XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-/mnt/nfs/niantian/robodojo_test/XPolicyLab}"
ROOT_DIR="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
LOG_ROOT="${LOG_ROOT:-/root/deploy_manual}"
RESULTS="${LOG_ROOT}/results_manual.txt"
SERVER_WAIT="${SERVER_WAIT:-600}"
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-900}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-0}"
VERIFY_START="${VERIFY_START:-}"
VERIFY_END="${VERIFY_END:-}"

mkdir -p "${LOG_ROOT}"

CONDA_BASE="${CONDA_BASE:-/mnt/nfs/miniconda3}"
export PATH="${CONDA_BASE}/bin:${CONDA_BASE}/condabin:${PATH}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate XPolicyLab
export PYTHONUNBUFFERED=1

if [[ -z "${VERIFY_START}" ]]; then
  : > "${RESULTS}"
fi

# policy | ckpt | expert_num | action | policy_env
POLICIES=(
  "Abot_M0|sim|100|joint|ABot"
  "GigaWorldPolicy|RoboDojo_sim_arx_seed_0|3500|joint|gigaworld-policy"
  "InternVLA_A1|RoboDojo_sim_seed_0|3500|joint|internvla_a1"
  "LingBot_VA|RoboDojo-cotrain-arx_x5-3500-joint-0|3500|joint|lingbot_va"
  "RDT_1B|RoboDojo_sim_seed_0|3500|joint|RDT"
  "Motus|RoboDojo-cotrain-arx_x5-3500-joint-0|3500|joint|motus"
  "OpenVLA_OFT|RoboDojo-cotrain-arx_x5-3500-joint-0|3500|joint|openvla_oft"
  "SmolVLA|RoboDojo_sim_arx-x5_seed_0|3500|joint|smolvla"
  "X_VLA|XVLA_sim_arx-x5|3500|ee|XVLA"
  "Spirit_v15|RoboDojo_sim_arx-x5_seed_0|3500|joint|uv"
  "MolmoACT2|RoboDojo-cotrain-arx_x5-3500-joint-0|3500|joint|uv"
  "Pi_05|Pi_05_sim_arx-x5_seed_1|3500|joint|uv"
  "Pi_0|RoboDojo_sim_arx_seed_0|3500|joint|uv"
  "Pi_0_Fast|RoboDojo_sim_arx_seed_0|3500|joint|uv"
  "Dexbotic_DM0|RoboDojo-cotrain-arx_x5-3500-ee-0|3500|ee|DM0"
  "Xiaomi_Robotics_0|RoboDojo-cotrain-arx_x5-100-ee-0|100|ee|mibot"
  "GR00T_N17|cotrain|3500|joint|uv"
)

verify_one() {
  local policy="$1" ckpt="$2" num="$3" act="$4" env="$5"
  local policy_dir="${XPOLICYLAB_ROOT}/policy/${policy}"
  local slog="${LOG_ROOT}/${policy}_server.log"
  local clog="${LOG_ROOT}/${policy}_client.log"
  local port
  local server_pid
  local rc=0

  echo "======== ${policy} ========" | tee -a "${RESULTS}"

  if [[ ! -d "${policy_dir}" ]]; then
    echo "FAIL ${policy} (no policy dir)" | tee -a "${RESULTS}"
    return 1
  fi

  port="$(bash "${UTILS_DIR}/get_free_port.sh")"
  : > "${slog}"
  : > "${clog}"

  cd "${policy_dir}" || return 1
  echo "[${policy}] server port=${port} env=${env} ckpt=${ckpt}" | tee -a "${RESULTS}"

  setsid bash setup_eval_policy_server.sh \
    RoboDojo stack_bowls "${ckpt}" arx_x5 "${num}" "${act}" 0 \
    "${POLICY_GPU}" "${env}" "${port}" localhost \
    > "${slog}" 2>&1 &
  server_pid=$!

  local server_wait="${SERVER_WAIT}"
  case "${policy}" in
    MolmoACT2|Pi_05|Pi_0|Pi_0_Fast|GR00T_N17) server_wait="${SLOW_POLICY_SERVER_WAIT:-1800}" ;;
  esac

  if ! bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${port}" "${server_pid}" "Policy server" "${server_wait}" >> "${slog}" 2>&1; then
    echo "FAIL ${policy} SERVER (wait port)" | tee -a "${RESULTS}"
    tail -20 "${slog}" | tee -a "${RESULTS}"
    kill -TERM -- -"${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  if grep -qE "Traceback|Error handling request|FileNotFoundError|ModuleNotFoundError|ImportError" "${slog}"; then
    echo "FAIL ${policy} SERVER (traceback in log)" | tee -a "${RESULTS}"
    grep -E "Traceback|Error:|FileNotFoundError|ModuleNotFoundError|ImportError" "${slog}" | tail -5 | tee -a "${RESULTS}"
    kill -TERM -- -"${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 1
  fi

  local add_info="ckpt_name=${ckpt},action_type=${act}"
  set +e
  timeout "${CLIENT_TIMEOUT}" bash setup_eval_env_client.sh \
    RoboDojo stack_bowls "${ckpt}" arx_x5 "${act}" 0 \
    "${ENV_GPU}" XPolicyLab "${add_info}" "${port}" localhost \
    > "${clog}" 2>&1
  rc=$?
  set -e

  kill -TERM -- -"${server_pid}" 2>/dev/null || kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  if [[ "${rc}" -eq 124 ]]; then
    echo "FAIL ${policy} CLIENT (timeout ${CLIENT_TIMEOUT}s)" | tee -a "${RESULTS}"
    tail -8 "${clog}" | tee -a "${RESULTS}"
    return 1
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL ${policy} CLIENT (exit ${rc})" | tee -a "${RESULTS}"
    tail -15 "${clog}" | tee -a "${RESULTS}"
    return 1
  fi

  if grep -qE "Communication error|Connection refused" "${clog}" && ! grep -q "Running Episode 9" "${clog}"; then
    echo "FAIL ${policy} CLIENT (connection errors)" | tee -a "${RESULTS}"
    tail -8 "${clog}" | tee -a "${RESULTS}"
    return 1
  fi

  echo "PASS ${policy}" | tee -a "${RESULTS}"
  return 0
}

echo "=== manual verify start $(date -Iseconds) VERIFY_START=${VERIFY_START:-0} VERIFY_END=${VERIFY_END:-0} ===" | tee -a "${RESULTS}"
started=false
if [[ -z "${VERIFY_START}" ]]; then
  started=true
fi
for spec in "${POLICIES[@]}"; do
  IFS='|' read -r policy ckpt num act env <<< "${spec}"
  if [[ "${started}" == false ]]; then
    if [[ "${policy}" == "${VERIFY_START}" ]]; then
      started=true
    else
      echo "SKIP ${policy} (resume from ${VERIFY_START})" | tee -a "${RESULTS}"
      continue
    fi
  fi
  verify_one "${policy}" "${ckpt}" "${num}" "${act}" "${env}" || true
  if [[ -n "${VERIFY_END}" && "${policy}" == "${VERIFY_END}" ]]; then
    break
  fi
done
echo "=== manual verify done $(date -Iseconds) ===" | tee -a "${RESULTS}"
