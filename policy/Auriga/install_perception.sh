#!/bin/bash
# Python environments of Auriga's perception services, built under VGGT_HOME (config.env; default
# third_party/vggt_home). Called by install.sh after the Auriga checkout exists; reuses Auriga's own
# installer services/vggt_height/setup.sh, which creates (uv-managed, no conda, nothing system-wide):
#   VGGT_HOME/venv         Python 3.11, torch 2.8.0+cu128, requirements-vggt.txt, vggt-omega @ 48b23c8 (editable)
#   VGGT_HOME/venv_sam31   Python 3.12, torch 2.8.0+cu128, sam31/requirements-sam31.txt, sam3 @ 7f473cf (editable)
# The package sets must equal Auriga's requirements-freeze files (checked below).
# Weights are a separate step: download_assets.sh. The services are started by the policy server
# (auriga_xpl/perception.py); nothing has to be started by hand.
#
#   bash install_perception.sh
#   TORCH_SOURCE=aliyun PYPI_INDEX=https://mirrors.aliyun.com/pypi/simple/ bash install_perception.sh   # mainland China
#
# Needs git, curl and python3 (only to bootstrap uv when it is not on PATH); about 12 GB under VGGT_HOME.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_env.sh"
load_config_env "${SCRIPT_DIR}/config.env"

abs_path() {
    local value=$1
    if [[ "${value}" = /* ]]; then echo "${value}"; else echo "${SCRIPT_DIR}/${value#./}"; fi
}
die() { echo "[install_perception][ERROR] $*" >&2; exit 1; }

auriga_src="$(abs_path "${AURIGA_SRC:-third_party/Auriga}")"
vggt_home="$(abs_path "${VGGT_HOME:-third_party/vggt_home}")"
service_dir="${auriga_src}/services/vggt_height"
[[ -f "${service_dir}/setup.sh" ]] || die "${service_dir}/setup.sh not found: clone Auriga first (install.sh)"
for tool in git curl python3; do
    command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"
done
if command -v nvidia-smi >/dev/null 2>&1; then
    # VGGT-Omega needs native bfloat16 (compute capability >= 8.0); see services/vggt_height/README.md FAQ.
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "${cc}" ]] && awk -v c="${cc}" 'BEGIN{exit !(c < 8.0)}'; then
        echo "[install_perception][WARN] GPU compute capability ${cc} < 8.0: VGGT-Omega falls back to emulated bf16 and may run out of memory"
    fi
fi

mkdir -p "${vggt_home}/run"
t0=$(date +%s)
echo "[install_perception] VGGT_HOME=${vggt_home} (setup log: ${vggt_home}/run/install_perception.log)"
VGGT_HOME="${vggt_home}" WITH_SAM31=1 bash "${service_dir}/setup.sh" 2>&1 | tee "${vggt_home}/run/install_perception.log"
grep -q '^DONE' "${vggt_home}/run/setup.STATUS" || die "setup.sh failed: $(cat "${vggt_home}/run/setup.STATUS" 2>/dev/null)"

# The environments must equal the reference freeze exactly (setup.sh only reports a difference).
norm() { grep -v -e '^#' -e '^-e ' -e ' @ file:' -e '^torch' "$1" | tr 'A-Z_' 'a-z-' | sort; }
diff <(norm "${service_dir}/requirements-freeze.txt") <(norm "${vggt_home}/run/freeze_vggt.txt") \
    || die "VGGT venv differs from ${service_dir}/requirements-freeze.txt"
diff <(norm "${service_dir}/sam31/requirements-freeze.txt") <(norm "${vggt_home}/run/freeze_sam31.txt") \
    || die "SAM 3.1 venv differs from ${service_dir}/sam31/requirements-freeze.txt"
for py in "${vggt_home}/venv/bin/python" "${vggt_home}/venv_sam31/bin/python"; do
    "${py}" -c "import torch; assert torch.__version__ == '2.8.0+cu128', torch.__version__" \
        || die "${py}: torch is not 2.8.0+cu128"
done

echo "[install_perception] done in $(( $(date +%s) - t0 )) s: ${vggt_home}/venv, ${vggt_home}/venv_sam31"
