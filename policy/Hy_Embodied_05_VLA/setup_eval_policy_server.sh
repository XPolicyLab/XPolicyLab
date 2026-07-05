#!/bin/bash
set -e

# Policy-side startup for Hy_Embodied_05_VLA. Activates the Hy-Embodied uv venv (torch +
# HunYuanVLMoT fork + flash_attn), puts the repo root on PYTHONPATH so the
# `hy_vla` package and the `robotwin_eval` adapter import, and launches the
# XPolicyLab policy server.

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=$9
policy_server_host=${10:-"localhost"}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"

# Resolve the Hy-Embodied uv venv root. "uv" -> read policy_uv_env_path from
# deploy.yml; otherwise treat the arg as a path. Relative paths resolve
# against the policy dir.
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

# Resolve the Hy-Embodied source tree root (mirrors model.py._resolve_hy_root):
# deploy.yml `hy_root` -> $HY_VLA_ROOT -> ./Hy-Embodied-0.5-VLA. Relative paths
# resolve against the policy dir. Needed to locate `checkpoints/<ckpt_name>`.
resolve_hy_root() {
    "${YAML_PYTHON}" - <<PYHY
import os, yaml
from pathlib import Path
script_dir = Path("${CURRENT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8")) or {}
hy_root = cfg.get("hy_root") or os.environ.get("HY_VLA_ROOT") or str(script_dir / "Hy-Embodied-0.5-VLA")
p = Path(hy_root).expanduser()
if not p.is_absolute():
    p = (script_dir / p).resolve()
print(p)
PYHY
}
hy_root="$(resolve_hy_root)"

# Turn the eval `ckpt_name` into an optional `ckpt_path=` override for
# setup_policy_server.py (overrides deploy.yml's default ckpt_path). Priority:
#   1. $HY_VLA_CKPT_PATH (highest)
#   2. ckpt_name if absolute / already exists
#   3. ${hy_root}/checkpoints/${ckpt_name}, then ${POLICY_DIR}/checkpoints/${ckpt_name}
# Empty / placeholder ckpt_name (or nothing resolvable) -> no override, so the
# deploy.yml ckpt_path default is used unchanged.
resolve_ckpt_override() {
    HY_VLA_CKPT_PATH="${HY_VLA_CKPT_PATH:-}" \
    CKPT_NAME="${ckpt_name}" \
    HY_ROOT="${hy_root}" \
    POLICY_DIR="${CURRENT_DIR}" \
    "${YAML_PYTHON}" - <<'PYCKPT'
import os
from pathlib import Path

env_override = (os.environ.get("HY_VLA_CKPT_PATH") or "").strip()
if env_override:
    print(Path(env_override).expanduser())
    raise SystemExit(0)

ckpt_name = (os.environ.get("CKPT_NAME") or "").strip()
_PLACEHOLDERS = {"", "null", "none", "default", "ckpt", "ckpt_name", "-"}
if ckpt_name.lower() in _PLACEHOLDERS:
    raise SystemExit(0)

cand = Path(ckpt_name).expanduser()
if cand.is_absolute() or cand.exists():
    print(cand)
    raise SystemExit(0)

hy_root = Path(os.environ["HY_ROOT"])
policy_dir = Path(os.environ["POLICY_DIR"])
for base in (hy_root / "checkpoints" / ckpt_name, policy_dir / "checkpoints" / ckpt_name):
    if base.exists():
        print(base)
        raise SystemExit(0)
PYCKPT
}
ckpt_path_override="$(resolve_ckpt_override)"

if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
    echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
    echo "[SERVER][ERROR] Run: bash ${CURRENT_DIR}/install.sh" >&2
    exit 1
fi

echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
source "${policy_uv_env_path}/.venv/bin/activate"
PYTHON_BIN="$(command -v python)"
echo "[SERVER] Using python: ${PYTHON_BIN}"

# ROOT_DIR -> XPolicyLab importable; hy_root -> hy_vla + robotwin_eval
# (pyproject only packages hy_vla, so robotwin_eval needs the repo on path).
PYTHONPATH_PARTS=("${ROOT_DIR}" "${policy_uv_env_path}")

overrides=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    action_dim="${action_dim}"
)
if [[ -n "${ckpt_path_override}" ]]; then
    echo "[SERVER] ckpt override: ckpt_path=${ckpt_path_override}"
    overrides+=(ckpt_path="${ckpt_path_override}")
elif [[ -n "${ckpt_name}" ]]; then
    echo "[SERVER] ckpt_name='${ckpt_name}' not resolved to a checkpoint dir; using deploy.yml ckpt_path"
fi

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
