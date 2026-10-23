#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVO1_SOURCE_DIR="${EVO1_SOURCE_DIR:-${SCRIPT_DIR}/upstream}"
EVO1_REVISION=5fd14b015013c4fd0aacf5f8f48f868ca9b870a2

if [[ ! -e "${EVO1_SOURCE_DIR}" ]]; then
    git clone --depth 1 --filter=blob:none --sparse --branch evo1-flash \
        https://github.com/MINT-SJTU/Evo-1.git "${EVO1_SOURCE_DIR}"
    git -C "${EVO1_SOURCE_DIR}" fetch --depth 1 origin "${EVO1_REVISION}"
    git -C "${EVO1_SOURCE_DIR}" checkout --detach "${EVO1_REVISION}"
    git -C "${EVO1_SOURCE_DIR}" sparse-checkout set Evo_1 RoboTwin_evaluation
elif [[ ! -f "${EVO1_SOURCE_DIR}/Evo_1/scripts/Evo1_server.py" ]]; then
    echo "[ERROR] Existing source directory is incomplete: ${EVO1_SOURCE_DIR}" >&2
    exit 1
fi
if [[ "$(git -C "${EVO1_SOURCE_DIR}" rev-parse HEAD)" != "${EVO1_REVISION}" ]]; then
    echo "[ERROR] Expected Evo-1 ${EVO1_REVISION}; existing checkout was left unchanged." >&2
    exit 1
fi
if ! git -C "${EVO1_SOURCE_DIR}" diff --quiet HEAD -- Evo_1 RoboTwin_evaluation; then
    echo "[ERROR] Source has tracked edits; use a clean pinned checkout." >&2
    exit 1
fi
if [[ "${1:-}" == "--source-only" ]]; then
    echo "[Evo_1] Pinned source ready: ${EVO1_SOURCE_DIR}"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash install.sh [--source-only] (inside the Evo-1 policy environment)" >&2
    exit 2
fi
EVO1_PLATFORM="${EVO1_PLATFORM:-cuda}"
if [[ "${EVO1_PLATFORM}" == "rocm" ]]; then
    # A ROCm build of the pinned Torch satisfies torch==2.5.1 below, so pip keeps it.
    python -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url "https://download.pytorch.org/whl/${EVO1_ROCM_WHEEL:-rocm6.2}"
elif [[ "${EVO1_PLATFORM}" != "cuda" ]]; then
    echo "[ERROR] EVO1_PLATFORM must be cuda or rocm, got ${EVO1_PLATFORM}" >&2
    exit 2
fi
# Resolve shared dependencies together so their upgrades retain numpy<2 and
# the policy's Torch/transformers constraints.
python -m pip install -r "${SCRIPT_DIR}/requirements.txt" -e "${XPL_ROOT}"
# The released weights were evaluated with FlashAttention.
if [[ "${EVO1_PLATFORM}" == "rocm" ]]; then
    # The ROCm build takes its Triton backend only with this set, at install and at import.
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
fi
MAX_JOBS="${EVO1_MAX_JOBS:-4}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
EVO1_SOURCE_DIR="$(cd "${EVO1_SOURCE_DIR}" && pwd)"
# A python -c launched in the policy directory would let its model.py shadow
# upstream's namespace package named model.
cd "${XPL_ROOT}"
PYTHONPATH="${EVO1_SOURCE_DIR}/Evo_1:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -c "import flash_attn; from scripts.Evo1_server import Normalizer; from XPolicyLab.policy.Evo_1.model import Model; print('[Evo_1] Imports passed')"
