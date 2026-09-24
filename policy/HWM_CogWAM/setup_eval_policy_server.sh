#!/bin/bash
set -Eeuo pipefail

# HWM_CogWAM is a two-hop server: this script starts the GPU-heavy CogWAM
# backend (`cogwam.serve.policy_server`, which loads the checkpoint) first,
# waits for it to pass its WebSocket handshake, and only then starts the
# standard XPolicyLab policy server (`setup_policy_server.py`, unmodified)
# hosting the lightweight `model.Model` bridge that talks to it. See
# README.md for the full picture.

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# --- Resolve the CogWAM checkpoint directory --------------------------------
#
# Routed through the shared XPolicyLab.utils.checkpoint_resolver
# (AGENTS.md: "Checkpoints resolve through ... never by re-deriving
# checkpoints/<bench>-<ckpt>-... by hand"), not a private join. Precedence:
#   1. COGWAM_ARTIFACT_DIR — explicit full path to a released artifact dir
#      (model.safetensors + dataset_statistics.json + inference_config.yaml)
#      or a raw training checkpoint directory. Passed as the resolver's
#      "checkpoint_path" explicit key.
#   2. checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
#   3. checkpoints/<ckpt_name>/ — the conventional symlink target used by
#      every other XPolicyLab adapter. Never a real copy of the (multi-GB)
#      weights.
if ! cogwam_artifact_dir="$(
    COGWAM_RESOLVE_BENCH_NAME="${bench_name}" \
    COGWAM_RESOLVE_CKPT_NAME="${ckpt_name}" \
    COGWAM_RESOLVE_ENV_CFG_TYPE="${env_cfg_type}" \
    COGWAM_RESOLVE_ACTION_TYPE="${action_type}" \
    COGWAM_RESOLVE_SEED="${seed}" \
    COGWAM_RESOLVE_CHECKPOINTS_DIR="${SCRIPT_DIR}/checkpoints" \
    COGWAM_RESOLVE_POLICY_DIR="${SCRIPT_DIR}" \
    COGWAM_RESOLVE_EXPLICIT_PATH="${COGWAM_ARTIFACT_DIR:-}" \
    PYTHONPATH="${XPL_ROOT}:${PYTHONPATH:-}" \
    python3 - <<'PY'
import os
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

model_cfg = {
    "bench_name": os.environ.get("COGWAM_RESOLVE_BENCH_NAME"),
    "ckpt_name": os.environ.get("COGWAM_RESOLVE_CKPT_NAME"),
    "env_cfg_type": os.environ.get("COGWAM_RESOLVE_ENV_CFG_TYPE"),
    "action_type": os.environ.get("COGWAM_RESOLVE_ACTION_TYPE"),
    "seed": os.environ.get("COGWAM_RESOLVE_SEED"),
    # COGWAM_ARTIFACT_DIR maps onto the resolver's own explicit-key
    # precedence instead of bypassing it.
    "checkpoint_path": os.environ.get("COGWAM_RESOLVE_EXPLICIT_PATH") or None,
}
root = resolve_checkpoint_root(
    model_cfg,
    os.environ["COGWAM_RESOLVE_CHECKPOINTS_DIR"],
    policy_dir=os.environ["COGWAM_RESOLVE_POLICY_DIR"],
    must_exist=True,
)
print(root.resolve())
PY
)"; then
    echo "[SERVER][ERROR] Could not resolve a CogWAM checkpoint directory." >&2
    echo "[SERVER][ERROR] export COGWAM_ARTIFACT_DIR=/path/to/artifact, or:" >&2
    echo "[SERVER][ERROR]   mkdir -p ${SCRIPT_DIR}/checkpoints && ln -sfn <artifact_dir> ${SCRIPT_DIR}/checkpoints/${ckpt_name}" >&2
    exit 1
fi

# --- Backbones are not redistributed with the checkpoint --------------------
#
# Both must be *local directories*, not Hugging Face repo ids. CogWAM passes
# these straight to `AutoModel.from_pretrained` and only sets
# `local_files_only=True` when the value is a directory
# (cogwam/models/dino_v3.py). A bare repo id therefore turns into a hub
# download — and for DINOv3 that repo is gated with manual approval, so the
# run would die with a 401 deep inside model construction instead of here.
: "${COGWAM_BASE_VLM:?export COGWAM_BASE_VLM=/path/to/RynnBrain1.1-2B (backbone weights are not redistributed; see download_checkpoint.py)}"
: "${COGWAM_DINO_MODEL:?export COGWAM_DINO_MODEL=/path/to/dinov3-vitb16 (backbone weights are not redistributed; see download_checkpoint.py)}"

for _backbone_var in COGWAM_BASE_VLM COGWAM_DINO_MODEL; do
    _backbone_dir="${!_backbone_var}"
    if [[ ! -d "${_backbone_dir}" ]]; then
        echo "[SERVER][ERROR] ${_backbone_var}='${_backbone_dir}' is not a directory." >&2
        echo "[SERVER][ERROR] It must be a local snapshot directory, not a Hugging Face repo id." >&2
        echo "[SERVER][ERROR] Fetch both backbones with: python ${SCRIPT_DIR}/download_checkpoint.py --dest <assets_dir>" >&2
        exit 1
    fi
done

# inference_config.yaml carries the full training-run config for provenance,
# including a couple of run/data-root interpolations that OmegaConf resolves
# eagerly even though nothing at inference time reads them. Point them at a
# scratch directory rather than requiring a real training run root.
export COGWAM_RUN_ROOT="${COGWAM_RUN_ROOT:-/tmp/hwm_cogwam_scratch/run_root}"
export COGWAM_DATA_ROOT="${COGWAM_DATA_ROOT:-/tmp/hwm_cogwam_scratch/data_root}"
mkdir -p "${COGWAM_RUN_ROOT}" "${COGWAM_DATA_ROOT}"

# RynnBrain1.1-2B interleaves linear-attention layers with full attention;
# the fast kernels (flash-linear-attention==0.3.2, causal_conv1d==1.5.0.post8)
# are pinned exactly in CogWAM's requirements.txt and are not guaranteed to be
# built on every inference host. Default to the same "torch_safe" fallback
# scripts/serve_policy.sh uses on inference hosts; override to 0 once the
# pinned kernels are actually installed.
export COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D="${COGWAM_QWEN35_DISABLE_CAUSAL_CONV1D:-1}"
export COGWAM_QWEN35_DISABLE_FLA="${COGWAM_QWEN35_DISABLE_FLA:-1}"
export COGWAM_QWEN35_ATTN_IMPLEMENTATION="${COGWAM_QWEN35_ATTN_IMPLEMENTATION:-sdpa}"

# --- Start the CogWAM backend server -----------------------------------------
backend_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
echo "[SERVER] starting CogWAM backend: artifact=${cogwam_artifact_dir} port=${backend_port}"

BACKEND_PID=""
POLICY_PID=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    [[ -z "${POLICY_PID}" ]] || kill "${POLICY_PID}" 2>/dev/null || true
    [[ -z "${BACKEND_PID}" ]] || kill "${BACKEND_PID}" 2>/dev/null || true
    [[ -z "${POLICY_PID}" ]] || wait "${POLICY_PID}" 2>/dev/null || true
    [[ -z "${BACKEND_PID}" ]] || wait "${BACKEND_PID}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT INT TERM

# COGWAM_ROOT is only needed here when CogWAM has not been `pip install -e`'d
# into this conda env (see install.sh, the reproducible path). Harmless when
# already installed: it just points PYTHONPATH at the same package.
(
    export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
    export PYTHONPATH="${COGWAM_ROOT:+${COGWAM_ROOT}:}${PYTHONPATH:-}"
    exec python -u -m cogwam.serve.policy_server \
        --artifact-dir "${cogwam_artifact_dir}" \
        --port "${backend_port}" \
        --idle_timeout -1 \
        --use_bf16
) &
BACKEND_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    127.0.0.1 "${backend_port}" "${BACKEND_PID}" "CogWAM backend server" 1200

# --- Start the standard XPolicyLab policy server, pointed at the backend ----
env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
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
            policy_server_host=127.0.0.1 \
            policy_server_port="${backend_port}" \
            expected_checkpoint_path="${cogwam_artifact_dir}" &
POLICY_PID=$!
wait "${POLICY_PID}"
