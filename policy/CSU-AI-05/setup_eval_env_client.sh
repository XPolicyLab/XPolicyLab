#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
env_gpu_id=${7:?env_gpu_id required}
eval_env_conda_env=${8:?eval_env_conda_env required}
additional_info=${9:-}
policy_server_port=${10:?policy_server_port required}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
default_bench_root="$(cd "${XPL_ROOT}/.." && pwd)"
BENCH_ROOT="${XPOLICYLAB_BENCH_ROOT:-${default_bench_root}}"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# XPolicyLab's shared setup_env_client.sh invokes `python` before activating
# the simulator environment. Minimal/non-interactive shells may expose only
# python3, so make the explicitly configured policy/client Python available.
if ! command -v python >/dev/null 2>&1; then
  raw_client_python="${XPOLICYLAB_CLIENT_PYTHON:-${G05_PYTHON:-}}"
  resolved_client_python=""
  if [[ -n "${raw_client_python}" ]]; then
    if [[ -x "${raw_client_python}" ]]; then
      resolved_client_python="${raw_client_python}"
    else
      resolved_client_python="$(command -v "${raw_client_python}" 2>/dev/null || true)"
    fi
  fi
  if [[ -z "${resolved_client_python}" ]]; then
    echo "The XPolicyLab environment client requires a 'python' command." >&2
    echo "Set XPOLICYLAB_CLIENT_PYTHON or G05_PYTHON to an environment bin/python." >&2
    exit 3
  fi
  export PATH="$(dirname "${resolved_client_python}"):${PATH}"
fi
if ! command -v python >/dev/null 2>&1; then
  echo "Configured client environment does not provide a 'python' executable." >&2
  exit 3
fi

# The shared debug/sim launchers activate the requested evaluator environment
# with conda. Make a configured conda executable discoverable in minimal SSH
# shells, where the installation's bin directory is often absent from PATH.
if ! command -v conda >/dev/null 2>&1; then
  raw_conda="${XPOLICYLAB_CONDA_EXE:-${CONDA_EXE:-}}"
  resolved_conda=""
  if [[ -n "${raw_conda}" ]]; then
    if [[ -x "${raw_conda}" ]]; then
      resolved_conda="${raw_conda}"
    else
      resolved_conda="$(command -v "${raw_conda}" 2>/dev/null || true)"
    fi
  fi
  if [[ -z "${resolved_conda}" ]]; then
    echo "The XPolicyLab evaluator client requires conda." >&2
    echo "Set XPOLICYLAB_CONDA_EXE or CONDA_EXE to bin/conda." >&2
    exit 3
  fi
  export PATH="$(dirname "${resolved_conda}"):${PATH}"
fi

if [[ ! -d "${BENCH_ROOT}" ]]; then
  echo "RoboDojo/RoboTwin root does not exist: ${BENCH_ROOT}" >&2
  exit 3
fi
if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" && ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
  echo "Simulator root is missing scripts/eval_policy.sh: ${BENCH_ROOT}" >&2
  echo "Set XPOLICYLAB_BENCH_ROOT to the RoboDojo or RoboTwin checkout." >&2
  exit 3
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] benchmark_root=${BENCH_ROOT}"

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
