#!/bin/bash
# Policy-side environment for Auriga. The eval (RoboDojo) environment needs nothing extra:
# deploy.py uses only the standard library and numpy.
#
#   bash install.sh [conda_env_name]      # default: Auriga
#
# Steps: clone Auriga at the commit pinned in deploy.yml, create its conda env (Python 3.12,
# environment.yml + requirements.lock.txt, which pins the Codex SDK/CLI), install XPolicyLab
# editable into it, install the policy-side planner stack (PyTorch, warp, cuRobo at RoboDojo's
# pinned commit; install_planner.sh), fetch the ARX X5 meshes next to the vendored URDF, and set
# up the perception service with its weights.
set -euo pipefail

env_name=${1:-Auriga}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AURIGA_REPO_URL="https://github.com/StellarEdge-AI/Auriga.git"
AURIGA_COMMIT="$(awk '$1 == "auriga_commit:" {print $2}' "${SCRIPT_DIR}/deploy.yml")"

abs_path() {
    local value=$1
    if [[ "${value}" = /* ]]; then echo "${value}"; else echo "${SCRIPT_DIR}/${value#./}"; fi
}

if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
    cp "${SCRIPT_DIR}/config.env.example" "${SCRIPT_DIR}/config.env"
    chmod 600 "${SCRIPT_DIR}/config.env"
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_env.sh"
load_config_env "${SCRIPT_DIR}/config.env"
auriga_src="$(abs_path "${AURIGA_SRC:-third_party/Auriga}")"

mkdir -p "${SCRIPT_DIR}/third_party"
if [[ ! -d "${auriga_src}/.git" ]]; then
    git clone "${AURIGA_REPO_URL}" "${auriga_src}"
fi
git -C "${auriga_src}" fetch --depth 1 origin "${AURIGA_COMMIT}" || true
git -C "${auriga_src}" checkout --detach "${AURIGA_COMMIT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# Same as RoboDojo scripts/install.sh: conda >= 25 refuses the default channels (still listed in a
# stock Miniconda .condarc) until their Terms of Service are accepted.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    conda env create -n "${env_name}" -f "${auriga_src}/environment.yml"
fi
conda activate "${env_name}"
python -m pip install -r "${auriga_src}/requirements.lock.txt"
python -m pip install -e "${XPL_ROOT}"
# The in-process RoboDojo adapter and the vendored planner use transforms3d (the worker used to get
# it from the RoboDojo environment).
python -m pip install transforms3d

# Policy-side planner: PyTorch + warp + cuRobo d17b54ce (RoboDojo's third_party/curobo pin).
bash "${SCRIPT_DIR}/install_planner.sh" python

# ARX X5 meshes next to the vendored X5A.urdf (vendor/robodojo_planner/Assets/Robots/x5/meshes),
# pinned Hugging Face revision, SHA-256 checked. Needed before the first episode.
bash "${SCRIPT_DIR}/download_robot_meshes.sh"

# Perception service venvs (VGGT-Omega, SAM 3.1) and weights.
bash "${SCRIPT_DIR}/install_perception.sh"
bash "${SCRIPT_DIR}/download_assets.sh"
echo "[install] done: conda env ${env_name}"
