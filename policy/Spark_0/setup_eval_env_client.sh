#!/bin/bash
set -euo pipefail

if [[ $# -lt 10 || $# -gt 11 ]]; then
    echo "Usage: bash setup_eval_env_client.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <env_gpu_id> <eval_env_conda_env> <additional_info> <policy_server_port> [policy_server_host]"
    exit 1
fi

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
policy_server_host=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Configuration parsing must use the requested environment too: the base
# Python on remote workers may not have PyYAML. Conda activation hooks can
# legitimately read unset variables, so relax nounset only during activation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"
set -u
echo "[CLIENT] Python: $(command -v python) (Conda: ${CONDA_PREFIX})"

# FAST_OBS_UPDATE (deploy.py) must read the SAME live model config as the
# policy server. Copy of setup_eval_policy_server.sh checkpoint lookup +
# _model_config_path. A pre-set SPARK_0_MODEL_CONFIG wins; otherwise ckpt_name
# yaml, then the checkpoint's recorded model_config.
run_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
result_run_dir="${SCRIPT_DIR}/results/Checkpoints/${ckpt_name}"
local_run_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"

if [[ "${ckpt_name}" == /* ]]; then
    candidate_run_dirs=("${ckpt_name}")
elif [[ "${ckpt_name}" == */* ]]; then
    candidate_run_dirs=("${SCRIPT_DIR}/${ckpt_name}")
else
    shopt -s nullglob
    tag_glob_dirs=(
        "${SCRIPT_DIR}/results/Checkpoints/${bench_name}-${ckpt_name}-"*"-${action_type}-${seed}"
        "${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-"*"-${action_type}-${seed}"
    )
    shopt -u nullglob
    candidate_run_dirs=(
        "${SCRIPT_DIR}/results/Checkpoints/${run_id}"
        "${SCRIPT_DIR}/checkpoints/${run_id}"
        ${tag_glob_dirs[@]+"${tag_glob_dirs[@]}"}
        "${result_run_dir}"
        "${local_run_dir}"
    )
fi

resolve_Spark_0_checkpoint() {
    local run_dir=$1
    local candidates=()

    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi

    shopt -s nullglob
    candidates=(
        "${run_dir}"/final_model/*.pt
        "${run_dir}"/final_model/*.safetensors
        "${run_dir}"/checkpoints/*.pt
        "${run_dir}"/checkpoints/*.safetensors
    )
    shopt -u nullglob

    if (( ${#candidates[@]} > 0 )); then
        printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
        return 0
    fi

    return 1
}

checkpoint_path="${SPARK_0_CKPT_PATH:-}"
if [[ -z "${checkpoint_path}" ]]; then
    for run_dir in "${candidate_run_dirs[@]}"; do
        if checkpoint_path=$(resolve_Spark_0_checkpoint "${run_dir}"); then
            break
        fi
        checkpoint_path=""
    done
fi

model_config_env="${SPARK_0_MODEL_CONFIG:-}"
spark0_resolved=$(python - "${SCRIPT_DIR}" "${ckpt_name}" "${checkpoint_path}" "${model_config_env}" <<'PY'
import sys
from pathlib import Path

import yaml

script_dir, ckpt_name, checkpoint_path, model_config_env = sys.argv[1:5]
config_dir = Path(script_dir) / "model_configs"

_NULLS = (None, "", "null", "None", "auto")


def _stem(text):
    text = str(text)
    return text[:-5] if text.endswith(".yaml") else text


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _checkpoint_value(key):
    """Top-level key from the checkpoint's saved training config."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return None
    run_dir = path.parents[1]
    for name in ("config.full.yaml", "config.yaml"):
        cfg_path = run_dir / name
        if cfg_path.is_file():
            value = _load_yaml(cfg_path).get(key)
            if value not in _NULLS:
                return value
    return None


def _model_config_path():
    """(path, why) of the live model config — same order as the policy server."""
    if model_config_env:
        candidate = Path(model_config_env).expanduser()
        if candidate.is_file():
            return candidate, "SPARK_0_MODEL_CONFIG"
        candidate = config_dir / f"{_stem(model_config_env)}.yaml"
        if candidate.is_file():
            return candidate, "SPARK_0_MODEL_CONFIG"
        print(
            f"[CLIENT][WARN] SPARK_0_MODEL_CONFIG={model_config_env} not found; "
            "falling back to ckpt_name / the checkpoint's recorded model_config.",
            file=sys.stderr,
        )
    candidate = config_dir / f"{_stem(ckpt_name)}.yaml"
    if candidate.is_file():
        return candidate, "ckpt_name"
    recorded = _checkpoint_value("model_config")
    if recorded:
        candidate = config_dir / f"{_stem(recorded)}.yaml"
        if candidate.is_file():
            return candidate, "the checkpoint's recorded model_config"
        print(
            f"[CLIENT][WARN] the checkpoint was trained from model_config="
            f"{recorded}, which no longer exists under {config_dir}.",
            file=sys.stderr,
        )
    return None, ""


cfg_path, cfg_why = _model_config_path()
if cfg_path is not None:
    print(cfg_path)
    print(f"[CLIENT] model config: {cfg_path.name} (via {cfg_why})", file=sys.stderr)
PY
)

if [[ -n "${spark0_resolved}" ]]; then
    export SPARK_0_MODEL_CONFIG="${spark0_resolved}"
else
    unset SPARK_0_MODEL_CONFIG || true
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, ckpt_name=${ckpt_name}, server=${policy_server_host}:${policy_server_port}"

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
    "${policy_server_host}"
