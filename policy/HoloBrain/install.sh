#!/bin/bash
# HoloBrain installation for XPolicyLab.
# Idempotent: safe to re-run; skips work that's already done.
#
# Usage:
#   bash install.sh [conda_env_name]   # default: holobrain
#
# Required system packages (install manually before running this):
#   - CUDA toolkit (>=12.1; flash-attn wheel must match)
#   - g++ / build-essential (for pytorch3d source build)
#   - git, wget, conda

set -euo pipefail

policy_conda_env="${1:-holobrain}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
RO_LAB_DIR="${POLICY_DIR}/RoboOrchardLab"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${policy_conda_env}"; then
    echo "[INFO] Conda env '${policy_conda_env}' already exists, skipping creation."
else
    echo "[INFO] Creating conda env '${policy_conda_env}' (python=3.10)..."
    conda create -n "${policy_conda_env}" python=3.10 -y
fi
conda activate "${policy_conda_env}"

python -m pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. PyTorch (must match the CUDA toolkit on this host)
# ---------------------------------------------------------------------------
# holobrain_0 requires torch>=2.6.0. Default to cu124+2.6.0; override via env.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"

if ! python -c "import torch" 2>/dev/null; then
    echo "[INFO] Installing torch==${TORCH_VERSION} from ${TORCH_INDEX_URL}..."
    python -m pip install \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
else
    echo "[INFO] torch already installed: $(python -c 'import torch;print(torch.__version__)')"
fi

# ---------------------------------------------------------------------------
# 3. Ensure VERSION_POSTFIX exists (snapshot has no .git so `make version`
#    cannot be used). Use a static postfix matching the upstream commit.
# ---------------------------------------------------------------------------
if [[ ! -f "${RO_LAB_DIR}/VERSION_POSTFIX" ]]; then
    echo "+local65607bf" > "${RO_LAB_DIR}/VERSION_POSTFIX"
fi

# ---------------------------------------------------------------------------
# 4. robo_orchard_lab[holobrain_0]
#
#    NOTE: pytorch3d>=0.7.8 and flash-attn<=2.8.3 are listed in the
#    holobrain_0 extras but pytorch3d builds from source and flash-attn
#    needs a CUDA-matched wheel. We install holobrain_0 with --no-build-isolation
#    so it picks up the already-installed torch; if pytorch3d/flash-attn
#    fail here, the script falls through to step 5/6.
# ---------------------------------------------------------------------------
if ! python -c "import robo_orchard_lab" 2>/dev/null; then
    echo "[INFO] Installing robo_orchard_lab[holobrain_0] (editable)..."
    if ! python -m pip install -e "${RO_LAB_DIR}[holobrain_0]" --no-build-isolation; then
        echo "[WARN] holobrain_0 extras install failed (likely pytorch3d/flash-attn build)."
        echo "[WARN] Installing core package only; pytorch3d/flash-attn handled in next steps."
        python -m pip install -e "${RO_LAB_DIR}" --no-build-isolation
        python -m pip install \
            "transformers<=4.57.1" "pytorch-kinematics" ninja diffusers \
            lmdb h5py terminaltables flask gevent "imageio[ffmpeg]"
    fi
else
    echo "[INFO] robo_orchard_lab already installed: $(python -c 'import robo_orchard_lab as r;print(getattr(r,\"__version__\",\"?\"))')"
fi

# ---------------------------------------------------------------------------
# 5. pytorch3d (from source)
# ---------------------------------------------------------------------------
if ! python -c "import pytorch3d" 2>/dev/null; then
    echo "[INFO] Installing pytorch3d==0.7.8 from source (~10 min build)..."
    python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
else
    echo "[INFO] pytorch3d already installed: $(python -c 'import pytorch3d;print(pytorch3d.__version__)')"
fi

# ---------------------------------------------------------------------------
# 6. flash-attn (CUDA/torch-matched prebuilt wheel)
# ---------------------------------------------------------------------------
if ! python -c "import flash_attn" 2>/dev/null; then
    cat <<'EOF'
[WARN] flash-attn not installed. Pick the wheel matching your torch+CUDA at:
       https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3

       For torch 2.8 / cu12 / py310 / cxx11ABI=TRUE:
         pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

       Install manually before running training/inference.
EOF
else
    echo "[INFO] flash-attn already installed: $(python -c 'import flash_attn;print(flash_attn.__version__)')"
fi

# ---------------------------------------------------------------------------
# 7. XPolicyLab (this repo)
# ---------------------------------------------------------------------------
echo "[INFO] Installing XPolicyLab (editable)..."
python -m pip install -e "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "[DONE] Conda env: ${policy_conda_env}"
echo "       Activate with: conda activate ${policy_conda_env}"
echo ""
python -c "import torch; print(f'  torch: {torch.__version__}, cuda={torch.cuda.is_available()}')" 2>/dev/null || true
python -c "import robo_orchard_lab; print(f'  robo_orchard_lab: ok')" 2>/dev/null || true
python -c "import pytorch3d; print(f'  pytorch3d: {pytorch3d.__version__}')" 2>/dev/null || echo "  pytorch3d: MISSING"
python -c "import flash_attn; print(f'  flash_attn: {flash_attn.__version__}')" 2>/dev/null || echo "  flash_attn: MISSING (install manually, see above)"
python -c "import XPolicyLab" 2>/dev/null && echo "  XPolicyLab: ok" || echo "  XPolicyLab: MISSING"
echo "================================================================"
