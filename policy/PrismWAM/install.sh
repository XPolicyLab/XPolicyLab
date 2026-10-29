#!/usr/bin/env bash
# Create the PrismWAM policy environment: bash install.sh [env_name]  (default: prismwam)
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_xpl
require_prismwam
env_name="${1:-prismwam}"
source "$(conda info --base)/etc/profile.d/conda.sh"
run_command conda create -y -n "$env_name" --override-channels -c conda-forge python=3.13 pip
is_dry_run || conda activate "$env_name"
cd "$PRISMWAM_ROOT"
run_command python -m pip install -r environments/torch.txt
run_command python -m pip install --no-build-isolation -r requirements.txt -e .
run_command python -m pip install -r environments/xpolicylab.txt
run_command python -m pip install --no-deps -e "$XPL_ROOT"
run_command python -m pip check
