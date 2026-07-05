#!/bin/bash
set -e

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_xpolicylab_root() {
    local dir
    dir="$(cd "${1}" && pwd)"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/setup_policy_server.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "[SERVER][ERROR] XPolicyLab root (setup_policy_server.py) not found above ${1}" >&2
    return 1
}
XPL_ROOT="$(find_xpolicylab_root "${CURRENT_DIR}")"
ROOT_DIR="$(dirname "${XPL_ROOT}")"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${CURRENT_DIR}/deploy.yml"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"

policy_name="$("${YAML_PYTHON}" - <<PYENV
import yaml
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
print(cfg["policy_name"])
PYENV
)"

ensure_policy_symlink() {
    local policy_real
    policy_real="$(readlink -f "${CURRENT_DIR}")"
    local policy_link="${XPL_ROOT}/policy/${policy_name}"

    if [[ -L "${policy_link}" ]]; then
        local link_target
        link_target="$(readlink -f "${policy_link}")"
        if [[ "${link_target}" != "${policy_real}" ]]; then
            echo "[SERVER][ERROR] ${policy_link} points to ${link_target}, not ${policy_real}" >&2
            exit 1
        fi
    elif [[ -e "${policy_link}" ]]; then
        local link_target
        link_target="$(readlink -f "${policy_link}")"
        if [[ "${link_target}" != "${policy_real}" ]]; then
            echo "[SERVER][ERROR] ${policy_link} exists at ${link_target}, not ${policy_real}; refusing to overwrite." >&2
            exit 1
        fi
    else
        mkdir -p "$(dirname "${policy_link}")"
        ln -s "${policy_real}" "${policy_link}"
        echo "[SERVER] Linked ${policy_link} -> ${policy_real}"
    fi
}

ensure_policy_symlink

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

resolve_uv_env() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        "${YAML_PYTHON}" - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    else
        "${YAML_PYTHON}" - <<PYENV
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
path = Path("${raw_path}").expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    fi
}

policy_uv_env_path="$(resolve_uv_env "${policy_uv_env}")"
if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
    echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
    echo "[SERVER][ERROR] Run: bash ${CURRENT_DIR}/install.sh" >&2
    exit 1
fi

echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    chmod +x "${policy_uv_env_path}/.venv/bin/python"* 2>/dev/null || true
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[SERVER][ERROR] venv python not executable: ${PYTHON_BIN}" >&2
    echo "[SERVER][ERROR] Run: bash ${CURRENT_DIR}/install.sh" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import jax" >/dev/null 2>&1; then
    echo "[SERVER][ERROR] jax not found in ${PYTHON_BIN}" >&2
    echo "[SERVER][ERROR] Run: bash ${CURRENT_DIR}/install.sh" >&2
    exit 1
fi
OPEN_SF_ROOT="${policy_uv_env_path}"
OPENPI_SRC="${OPEN_SF_ROOT}/src"
OPENPI_CLIENT_SRC="${OPEN_SF_ROOT}/packages/openpi-client/src"
VGGT_SRC="${OPEN_SF_ROOT}/src/vggt"
echo "[SERVER] Using python: ${PYTHON_BIN}"

PYTHONPATH_PARTS=("${ROOT_DIR}")
for path in "${OPENPI_SRC}" "${OPENPI_CLIENT_SRC}" "${VGGT_SRC}"; do
    if [[ -d "${path}" ]]; then
        PYTHONPATH_PARTS+=("${path}")
    fi
done

TJY_ROOT="$(cd "${CURRENT_DIR}/../../../../.." && pwd)"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${TJY_ROOT}/.cache/openpi}"
mkdir -p "${OPENPI_DATA_HOME}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    TF_CPP_MIN_LOG_LEVEL=2 \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=offline \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
