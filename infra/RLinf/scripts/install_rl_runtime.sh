#!/usr/bin/env bash
# Install the RLinf policy-side runtime into the already-active interpreter.
# Activate the training environment with XPolicyLab and the policy dependencies
# installed. For Pi0.5, follow policy/Pi_05/README.md from the XPolicyLab root.
# Optional: RL_PYTHON / RL_PIP override the interpreter and install command.
# The RoboTwin env worker imports robotwin.envs.vector_env in its own interpreter.
# Provide the RoboTwin runtime and its simulator dependencies there.

set -euo pipefail

UPGRADE_TORCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upgrade-torch) UPGRADE_TORCH=1; shift ;;
    -h|--help)
      echo "usage: bash install_rl_runtime.sh [--upgrade-torch]" >&2
      echo "Activate the policy training environment first. For Pi0.5, see policy/Pi_05/README.md." >&2
      exit 0
      ;;
    *)
      echo "usage: bash install_rl_runtime.sh [--upgrade-torch]" >&2
      exit 1
      ;;
  esac
done

_XPL_RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_XPL_RLINF_ROOT="${_XPL_RL_DIR}/RLinf"
RL_PYTHON="${RL_PYTHON:-python}"

_xpl_rl_pip() {
  if [[ -n "${RL_PIP:-}" ]]; then
    # Caller supplies the install command, e.g. `uv pip install --python ... --prerelease=allow`.
    # shellcheck disable=SC2086
    ${RL_PIP} "$@"
  else
    "${RL_PYTHON}" -m pip install "$@"
  fi
}

if [[ ! -f "${_XPL_RLINF_ROOT}/pyproject.toml" ]]; then
  echo "[install_rl_runtime] RLinf checkout missing at infra/RLinf/RLinf." >&2
  echo "[install_rl_runtime] Run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! "${RL_PYTHON}" -I -c "import XPolicyLab" >/dev/null 2>&1; then
  echo "[install_rl_runtime] XPolicyLab is not importable in ${RL_PYTHON}." >&2
  echo "[install_rl_runtime] Activate the training environment and install XPolicyLab from its root with: python -m pip install -e ." >&2
  exit 1
fi

if [[ "${UPGRADE_TORCH}" -eq 1 ]]; then
  echo "[install_rl_runtime] Upgrading torch to 2.11 (cu128) in ${RL_PYTHON}"
  _xpl_rl_pip torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  py_tag="$("${RL_PYTHON}" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  torch_tag="$("${RL_PYTHON}" -c 'import torch; v=torch.__version__.split("+", 1)[0].split("."); print(f"torch{v[0]}.{v[1]}")')"
  cxx_abi="$("${RL_PYTHON}" -c 'import torch; print("cxx11abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "cxx11abiFALSE")')"
  wheel_name="flash_attn-2.8.3+cu12${torch_tag}${cxx_abi}-${py_tag}-${py_tag}-linux_x86_64.whl"
  "${RL_PYTHON}" -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
  installed=0
  for host in \
    "https://github.com/RLinf/flash-attention/releases/download/v2.8.3" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
  do
    echo "[install_rl_runtime] Installing ${wheel_name} from ${host}"
    if _xpl_rl_pip "${host}/${wheel_name}"; then
      installed=1
      break
    fi
  done
  if [[ "${installed}" -ne 1 ]]; then
    echo "[install_rl_runtime] flash-attn wheel ${wheel_name} was not available." >&2
    exit 1
  fi
fi

# hydra-core 1.4.0.dev1 / omegaconf 2.4.0.dev4 need a prerelease-capable pip.
# Pin any already-installed torch/jax so --pre cannot replace them.
_XPL_RL_CONSTRAINT="$("${RL_PYTHON}" - <<'PY'
try:
    import torch
    print("torch==" + torch.__version__.split("+", 1)[0])
except Exception:
    pass
try:
    import jax
    print("jax==" + jax.__version__)
except Exception:
    pass
PY
)"
_XPL_RL_CFILE="$(mktemp)"
printf '%s\n' "${_XPL_RL_CONSTRAINT}" > "${_XPL_RL_CFILE}"

# --no-deps is used only for the RLinf checkout so its torch/wandb pins stay unused.
if ! _xpl_rl_pip --pre -c "${_XPL_RL_CFILE}" \
  "ray[default]==2.58.0" \
  "hydra-core==1.4.0.dev1" \
  "omegaconf==2.4.0.dev4" \
  "numpy==1.26.4" \
  "sortedcontainers" \
  "accelerate" \
  "torchdata==0.11.0" \
  "tensorboard" \
  "einops"
then
  rm -f "${_XPL_RL_CFILE}"
  exit 1
fi
if ! _xpl_rl_pip --no-deps -e "${_XPL_RLINF_ROOT}"; then
  rm -f "${_XPL_RL_CFILE}"
  exit 1
fi
rm -f "${_XPL_RL_CFILE}"

"${RL_PYTHON}" -c "import ray, hydra, omegaconf, rlinf; print('[install_rl_runtime] ray', ray.__version__, 'ok in', '${RL_PYTHON}')"
