#!/bin/bash
set -euo pipefail

# Mem_0 Planning Module environment (LLaMA-Factory). Conda only; no Docker.
# Run from this policy folder: bash install_planning.sh [llama_factory_conda_env]
#
# Creates the llama_factory conda env, clones LLaMA-Factory into Mem_0/LlamaFactory
# when missing, and installs editable LLaMA-Factory + metrics/wandb deps.

planning_conda_env="${1:-llama_factory}"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${POLICY_DIR}/Mem_0"
LF_DIR="${UPSTREAM_DIR}/LlamaFactory"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${planning_conda_env}"; then
  echo "[install_planning] Conda env '${planning_conda_env}' already exists; reusing."
else
  echo "[install_planning] Creating conda env '${planning_conda_env}' (python 3.11)..."
  conda create -n "${planning_conda_env}" python=3.11 -y
fi

if [[ -d "${LF_DIR}/.git" || -f "${LF_DIR}/pyproject.toml" || -f "${LF_DIR}/setup.py" ]]; then
  echo "[install_planning] LLaMA-Factory already present at ${LF_DIR}; skipping clone."
else
  echo "[install_planning] Cloning LLaMA-Factory into ${LF_DIR}..."
  git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git "${LF_DIR}"
fi

conda activate "${planning_conda_env}"
pip install -e "${LF_DIR}"
pip install -r "${LF_DIR}/requirements/metrics.txt" wandb

echo "[install_planning] ${planning_conda_env} ready."
echo "[install_planning] Download Qwen3-VL-8B:  cd ${UPSTREAM_DIR}/checkpoints && python _download.py"
echo "[install_planning] wandb login (optional, for planning train logging)."
echo "[install_planning] Train/eval workflow: see policy/Mem_0/README.md"
