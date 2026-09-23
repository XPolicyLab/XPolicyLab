#!/usr/bin/env bash
# Griffin Alpha-S: conda env + the LeRobot policy plugin (griffinlabs-ai/alpha-s) + XPolicyLab.
#
# The plugin is cloned into policy/Griffin_Alpha_S/alpha-s/ (git-ignored) and installed editable with
# its [train] extra, so `lerobot-train` and the plugin's `scripts/make_finetune_base.py` (used by
# process_data.sh) are both available in the policy environment.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
ALPHA_S_ROOT="${GRIFFIN_ALPHA_S_ROOT:-${POLICY_DIR}/alpha-s}"

CONDA_ENV="${GRIFFIN_CONDA_ENV:-griffin_alpha_s}"
PYTHON_VERSION="${GRIFFIN_PYTHON_VERSION:-3.12}"
ALPHA_S_REPO="${GRIFFIN_ALPHA_S_REPO:-https://github.com/griffinlabs-ai/alpha-s.git}"
ALPHA_S_REF="${GRIFFIN_ALPHA_S_REF:-main}"

echo "[Griffin_Alpha_S] POLICY_DIR=${POLICY_DIR}"
echo "[Griffin_Alpha_S] conda env=${CONDA_ENV} (python=${PYTHON_VERSION})"
echo "[Griffin_Alpha_S] plugin -> ${ALPHA_S_ROOT} (${ALPHA_S_REPO} @ ${ALPHA_S_REF})"

if ! command -v conda >/dev/null 2>&1; then
  echo "[Griffin_Alpha_S] ERROR: conda not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[Griffin_Alpha_S] Creating conda env: ${CONDA_ENV}"
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"

if [[ -f "${ALPHA_S_ROOT}/pyproject.toml" ]]; then
  echo "[Griffin_Alpha_S] plugin checkout already present: ${ALPHA_S_ROOT}"
  if [[ "${GRIFFIN_UPDATE_PLUGIN:-0}" == "1" && -d "${ALPHA_S_ROOT}/.git" ]]; then
    git -C "${ALPHA_S_ROOT}" fetch --depth 1 origin "${ALPHA_S_REF}"
    git -C "${ALPHA_S_ROOT}" checkout FETCH_HEAD
  fi
elif [[ -d "${ALPHA_S_ROOT}" && -n "$(ls -A "${ALPHA_S_ROOT}" 2>/dev/null)" ]]; then
  echo "[Griffin_Alpha_S] ERROR: ${ALPHA_S_ROOT} exists but is not an alpha-s checkout." >&2
  exit 1
else
  git clone --branch "${ALPHA_S_REF}" --depth 1 "${ALPHA_S_REPO}" "${ALPHA_S_ROOT}"
fi

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${GRIFFIN_TORCH_INDEX:-}" ]]; then
  echo "[Griffin_Alpha_S] Installing PyTorch from ${GRIFFIN_TORCH_INDEX}"
  pip install torch torchvision --index-url "${GRIFFIN_TORCH_INDEX}"
fi

# lerobot[dataset,training]==0.6.1, transformers 5.5.x, torch, the plugin itself.
pip install -e "${ALPHA_S_ROOT}[train]"

# XPolicyLab (websocket server, msgpack, the shared observation/action helpers).
pip install -e "${XPL_ROOT}"

python - <<'PY'
import lerobot
import lerobot_policy_griffin_alpha  # noqa: F401  registers the two policy types
from lerobot.policies.factory import get_policy_class
print("[Griffin_Alpha_S] lerobot", lerobot.__version__)
print("[Griffin_Alpha_S] griffin_alpha:", get_policy_class("griffin_alpha").__name__)
print("[Griffin_Alpha_S] griffin_alpha_fast:", get_policy_class("griffin_alpha_fast").__name__)
PY
python -c "import XPolicyLab; print('[Griffin_Alpha_S] XPolicyLab ok')"

cat <<EOT

[Griffin_Alpha_S] Installation finished.
  conda activate ${CONDA_ENV}

Optional, recommended on Ampere-or-newer GPUs (the policy falls back to SDPA with a warning otherwise):
  pip install flash-attn --no-build-isolation      # or a prebuilt wheel, see ${ALPHA_S_ROOT}/pyproject.toml
Optional, faster dataset video decoding for training:
  conda install -c conda-forge ffmpeg
EOT
