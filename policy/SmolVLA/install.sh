# XPolicyLab deploy: policy server env=smolvla; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
# SmolVLA: conda env + auto-clone HuggingFace LeRobot into policy/SmolVLA/smovla.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOVLA_ROOT="${POLICY_DIR}/smovla"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

CONDA_ENV="${SMOVLA_CONDA_ENV:-smolvla}"
PYTHON_VERSION="${SMOVLA_PYTHON_VERSION:-3.10}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
LEROBOT_REF="${LEROBOT_REF:-v0.4.4}"

echo "[SmolVLA] POLICY_DIR=${POLICY_DIR}"
echo "[SmolVLA] conda env=${CONDA_ENV} (python=${PYTHON_VERSION})"
echo "[SmolVLA] LeRobot -> ${SMOVLA_ROOT} (${LEROBOT_REF})"

if ! command -v conda >/dev/null 2>&1; then
  echo "[SmolVLA] ERROR: conda not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${SMOVLA_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "[SmolVLA] Creating conda env: ${CONDA_ENV}"
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "[SmolVLA] ERROR: conda env '${CONDA_ENV}' not found." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"

ensure_lerobot_repo() {
  if [[ -f "${SMOVLA_ROOT}/pyproject.toml" ]]; then
    echo "[SmolVLA] LeRobot repo already present: ${SMOVLA_ROOT}"
    if [[ "${SMOVLA_UPDATE_LEROBOT:-0}" == "1" ]] && [[ -d "${SMOVLA_ROOT}/.git" ]]; then
      echo "[SmolVLA] Fetching ${LEROBOT_REF} ..."
      git -C "${SMOVLA_ROOT}" fetch --tags --depth 1 origin
      git -C "${SMOVLA_ROOT}" checkout "${LEROBOT_REF}"
    fi
    return 0
  fi

  if [[ -d "${SMOVLA_ROOT}" ]] && [[ -n "$(ls -A "${SMOVLA_ROOT}" 2>/dev/null)" ]]; then
    echo "[SmolVLA] ERROR: ${SMOVLA_ROOT} exists but is not a LeRobot checkout (no pyproject.toml)." >&2
    echo "[SmolVLA] Remove it or set SMOVLA_ROOT elsewhere, then re-run install.sh" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${SMOVLA_ROOT}")"
  echo "[SmolVLA] Cloning ${LEROBOT_REPO} (${LEROBOT_REF}) ..."
  if ! git clone --branch "${LEROBOT_REF}" --depth 1 "${LEROBOT_REPO}" "${SMOVLA_ROOT}"; then
    echo "[SmolVLA] Retry clone with tag ${LEROBOT_REF#v} ..."
    rm -rf "${SMOVLA_ROOT}"
    git clone --branch "${LEROBOT_REF#v}" --depth 1 "${LEROBOT_REPO}" "${SMOVLA_ROOT}"
  fi
}

ensure_lerobot_repo

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${SMOVLA_TORCH_INDEX:-}" ]]; then
  echo "[SmolVLA] Installing PyTorch from ${SMOVLA_TORCH_INDEX}"
  pip install torch torchvision --index-url "${SMOVLA_TORCH_INDEX}"
fi

echo "[SmolVLA] Installing LeRobot editable with [smolvla] extra ..."
cd "${SMOVLA_ROOT}"
pip install -e ".[smolvla]"

cd "${XPOLICYLAB_ROOT}"
pip install -e .
pip install h5py

python -c "
import lerobot
from lerobot.policies.factory import get_policy_class
print('[SmolVLA] lerobot', getattr(lerobot, '__version__', 'unknown'))
print('[SmolVLA] smolvla policy:', get_policy_class('smolvla'))
"
python -c "import XPolicyLab; print('[SmolVLA] XPolicyLab ok')"

if command -v lerobot-train >/dev/null 2>&1; then
  echo "[SmolVLA] lerobot-train: $(command -v lerobot-train)"
else
  echo "[SmolVLA] WARNING: lerobot-train not on PATH; check conda env activation." >&2
fi

cat <<EOF

[SmolVLA] Installation finished.
  conda activate ${CONDA_ENV}
  LeRobot source: ${SMOVLA_ROOT}

Optional system packages (video decode):
  sudo apt-get install -y ffmpeg

Train / eval: use conda env name "${CONDA_ENV}" in eval.sh and deploy.sh.
EOF
