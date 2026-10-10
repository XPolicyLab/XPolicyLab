#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

INSTALL_ENV="${EGOVLA_CONDA_ENV:-egovla}"
PYTHON_BIN="${EGOVLA_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${INSTALL_ENV}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" && -x "${INSTALL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${INSTALL_ENV}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" && -n "${EGOVLA_CONDA_HOME:-}" && -f "${EGOVLA_CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
  source "${EGOVLA_CONDA_HOME}/etc/profile.d/conda.sh"
  conda activate "${INSTALL_ENV}"
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" && -z "${EGOVLA_PYTHON_BIN:-}" && -n "${CONDA_EXE:-}" ]]; then
  if [[ "${INSTALL_ENV}" == /* ]]; then
    "${CONDA_EXE}" install -p "${INSTALL_ENV}" python=3.11 pip -y
  else
    "${CONDA_EXE}" install -n "${INSTALL_ENV}" python=3.11 pip -y
  fi
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${INSTALL_ENV}" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ "${INSTALL_ENV}" == /* ]]; then
    PYTHON_BIN="${INSTALL_ENV}/bin/python"
  else
    PYTHON_BIN="$("${CONDA_EXE}" run -n "${INSTALL_ENV}" python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
  fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || {
  echo "No Python interpreter found for ${INSTALL_ENV}; install Python 3.11 in that environment first" >&2; exit 2;
}

TORCH_INDEX_URL="${EGOVLA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
if ! "${PYTHON_BIN}" -c 'import torch, torchvision; assert torch.version.cuda == "12.8"' >/dev/null 2>&1; then
  echo "[egovla] installing PyTorch/CUDA 12.8 runtime"
  "${PYTHON_BIN}" -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX_URL}"
fi
"${PYTHON_BIN}" -c 'import torch; print("[egovla] torch:", torch.__version__, "cuda:", torch.version.cuda)'

DEPS_DIR="${POLICY_DIR}/.deps"
DEPS_STAGING="${POLICY_DIR}/.deps_$$"
rm -rf "${DEPS_STAGING}"
mkdir -p "${DEPS_STAGING}"
cleanup_staging() {
  rm -rf "${DEPS_STAGING}"
}
trap cleanup_staging EXIT

# MANO is deliberately opt-in.  The direct Inspire-12 adapter does not need
# the licensed model files or the legacy ``chumpy`` dependency, and keeping
# this switch off preserves the lightweight/default training path.  When the
# official EgoVLA MANO representation is requested, install the two pure
# Python packages into the adapter-local dependency directory (never into the
# shared CUDA/PyTorch conda environment) and verify that both supplied pkl
# files can actually be loaded.
install_mano=0
case "${EGOVLA_INSTALL_MANO:-0}" in
  1|true|TRUE|yes|YES|on|ON) install_mano=1 ;;
esac

find_mano_models() {
  local configured="${EGOVLA_MANO_ROOT:-}"
  local candidate
  local candidates=(
    "${configured}"
    "${configured}/models"
    "${configured}/mano_v1_2/models"
    "${POLICY_DIR}/mano_v1_2/models"
    "${POLICY_DIR}/EgoVLA_Release/mano_v1_2/models"
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}/MANO_LEFT.pkl" && -f "${candidate}/MANO_RIGHT.pkl" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

# Keep adapter dependencies under the user's workspace.  PyTorch and
# torchvision are installed in the selected egovla environment above; the
# remaining Python packages stay isolated under the adapter-local .deps
# directory.  --no-deps prevents pip from replacing the CUDA/PyTorch
# runtime with a second incompatible stack.
# The transitive runtime packages are listed explicitly below for the same reason.
"${PYTHON_BIN}" -m pip install --upgrade --no-deps --target "${DEPS_STAGING}" \
  "numpy<2" \
  "transformers>=4.44,<4.46" \
  "accelerate>=0.30,<1.0" \
  "datasets>=2.16,<3" \
  "hydra-core>=1.3,<1.4" \
  "sentencepiece>=0.1.99" \
  "safetensors>=0.4" \
  "scipy>=1.10" \
  "h5py>=3.8" \
  "wandb==0.16.6" \
  "loguru>=0.7,<1" \
  "shortuuid>=1,<2" \
  "einops>=0.6,<1" \
  "einops-exts>=0.0.4,<1" \
  "timm>=0.9,<1" \
  "markdown2>=2.4,<3" \
  "fire>=0.5,<1" \
  "tyro>=0.8,<1" \
  "tokenizers>=0.15,<0.21" \
  "huggingface_hub>=0.23,<1" \
  "regex>=2023.0" \
  "requests>=2.31" \
  "tqdm>=4.64" \
  "pyyaml>=6" \
  "packaging>=23" \
  "filelock>=3.12" \
  "fsspec>=2023.1" \
  "pillow>=9" \
  "pyarrow>=12,<26" \
  "pandas>=1.5" \
  "dill>=0.3" \
  "multiprocess>=0.70" \
  "xxhash>=3" \
  "aiohttp>=3.8" \
  "websockets>=14,<16" \
  "msgpack>=1.0.8" \
  "msgpack-numpy>=0.4.8" \
  "aiohappyeyeballs>=2.3" \
  "appdirs>=1.4" \
  "pydantic>=1.10,<3" \
  "protobuf>=3.20,<5" \
  "jinja2>=3.1" \
  "networkx>=2.8" \
  "iopath>=0.1.9,<1" \
  "portalocker>=2.7,<5" \
  "pytorchvideo>=0.1.5,<1" \
  "shtab>=1.5.6,<2" \
  "docstring-parser>=0.15,<1" \
  "rich>=11,<16" \
  "typeguard>=4,<5" \
  "annotated-types>=0.6" \
  "antlr4-python3-runtime==4.9.*" \
  "av>=12,<19" \
  "aiosignal>=1.3" \
  "attrs>=22" \
  "certifi>=2023" \
  "charset-normalizer>=3" \
  "click>=8.1" \
  "docker-pycreds>=0.4" \
  "frozenlist>=1.1" \
  "fvcore>=0.1.5" \
  "gitpython>=3.1" \
  "gitdb>=4" \
  "smmap>=5" \
  "idna>=3" \
  "markdown-it-py>=2" \
  "markupsafe>=2" \
  "multidict>=4.5" \
  "omegaconf>=2.3,<2.4" \
  "opencv-python-headless>=4.8,<5" \
  "parameterized>=0.9" \
  "platformdirs>=3" \
  "propcache>=0.2" \
  "psutil>=5.9" \
  "python-dateutil>=2.8" \
  "pytz>=2022" \
  "pygments>=2.13" \
  "sentry-sdk>=2" \
  "setproctitle>=1.3" \
  "six>=1.16" \
  "tabulate>=0.9" \
  "termcolor>=2" \
  "typing-extensions>=4" \
  "typing-inspection>=0.4" \
  "urllib3>=1.26" \
  "yarl>=1.0" \
  "tzdata>=2023" \
  "pydantic-core>=2"

if (( install_mano )); then
  MANO_MODELS_DIR="$(find_mano_models || true)"
  if [[ -z "${MANO_MODELS_DIR}" ]]; then
    echo "MANO files are missing. Set EGOVLA_MANO_ROOT to a directory containing" >&2
    echo "MANO_LEFT.pkl and MANO_RIGHT.pkl (or copy them to ${POLICY_DIR}/mano_v1_2/models)." >&2
    exit 2
  fi

  # ``chumpy`` is used while unpickling MANO v1.2.  Keep the versions pinned
  # to the upstream EgoVLA recipe and install without dependencies so pip
  # cannot replace the image's CUDA/PyTorch stack.
  "${PYTHON_BIN}" -m pip install --upgrade --no-deps --target "${DEPS_STAGING}" \
    "chumpy==0.70" \
    "smplx==0.1.28"

  # Validate the complete legacy-pickle path now, rather than failing later
  # during a distributed launch.  The aliases mirror compat.py and are needed
  # by chumpy on modern Python/NumPy versions.
  PYTHONPATH="${DEPS_STAGING}:${POLICY_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${MANO_MODELS_DIR}" <<'PY'
import inspect
import sys
from collections import namedtuple

import numpy as np

for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(function):
        spec = inspect.getfullargspec(function)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

import smplx

models = sys.argv[1]
for side, is_right in (("LEFT", False), ("RIGHT", True)):
    model = smplx.create(
        f"{models}/MANO_{side}.pkl",
        "mano",
        use_pca=True,
        is_rhand=is_right,
        num_pca_comps=15,
    )
    assert tuple(model.hand_components.shape) == (15, 45)
print(f"[egovla] MANO verified: {models}")
PY

  echo "[egovla] MANO packages staged; models=${MANO_MODELS_DIR}"
else
  echo "[egovla] MANO remains opt-in (set EGOVLA_INSTALL_MANO=1 after installing/copying the licensed pkl files)"
fi

rm -rf "${DEPS_DIR}"
mv "${DEPS_STAGING}" "${DEPS_DIR}"
trap - EXIT
echo "[egovla] install complete (dependencies: ${DEPS_DIR}). FlashAttention remains optional; direct Inspire-12 mode is unchanged."
