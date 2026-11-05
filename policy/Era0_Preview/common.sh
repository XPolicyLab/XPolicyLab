#!/usr/bin/env bash
# Source in a launcher; never activate an environment in the caller's shell.
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
export ROBODOJO_ROOT="${ROBODOJO_ROOT:-${BENCH_ROOT}}"
export PYTHONPATH="${ROBODOJO_ROOT}:${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
# Image timing compatibility lives in the policy; SDK rendering stays unchanged.
export HY_OBS_LAG="${HY_OBS_LAG:-1}"
export ROBODOJO_EVAL_LEGACY_RENDER="${ROBODOJO_EVAL_LEGACY_RENDER:-0}"
resolve_python() {
    local requested=$1
    if [[ -x "${requested}" && ! -d "${requested}" ]]; then
        PYTHON_BIN="$(cd "$(dirname "${requested}")" && pwd)/$(basename "${requested}")"
    elif [[ -x "${requested}/bin/python" ]]; then
        PYTHON_BIN="$(cd "${requested}" && pwd)/bin/python"
    else
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${requested}"
        PYTHON_BIN="$(command -v python)"
    fi
}
# Fixed inference optimization settings used by the original full sweep.
export XVLA_QWEN_ATTN_IMPLEMENTATION=sdpa
export XVLA_QWEN_VISION_ATTN_IMPLEMENTATION="${XVLA_QWEN_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
export XVLA_QWEN_FIXED_IMAGE_VISION=1
export XVLA_G05_REUSE_PREFIX_PROJECTIONS=1
export XVLA_G05_TERMINAL_PREFIX_ONLY=1
