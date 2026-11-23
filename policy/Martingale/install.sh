#!/bin/bash
# Copyright © 2026 Martingale. All Rights Reserved.
set -euo pipefail

adapter=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
xpl_root=${MARTINGALE_XPOLICYLAB_ROOT:-$(cd "$adapter/../.." && pwd)}
if [[ ! -f "$xpl_root/setup_policy_server.py" ]]; then
    echo "Install from XPolicyLab/policy/Martingale or set MARTINGALE_XPOLICYLAB_ROOT to an XPolicyLab checkout." >&2
    exit 2
fi

source_dir=${MARTINGALE_SOURCE:-"$adapter/runtime"}
# Keep the existing explicit alternate-revision installation path.
if [[ -z ${MARTINGALE_SOURCE:-} && -n ${MARTINGALE_REPO_URL:-}${MARTINGALE_COMMIT:-} ]]; then
    : "${MARTINGALE_REPO_URL:?set the alternate repository URL}"
    : "${MARTINGALE_COMMIT:?set the alternate repository commit}"
    source_dir="$adapter/third_party/martingale"
    if [[ ! -d "$source_dir/.git" ]]; then
        git clone "$MARTINGALE_REPO_URL" "$source_dir"
    fi
    if ! git -C "$source_dir" cat-file -e "${MARTINGALE_COMMIT}^{commit}"; then
        git -C "$source_dir" fetch origin "$MARTINGALE_COMMIT"
    fi
    git -C "$source_dir" checkout --quiet --detach "$MARTINGALE_COMMIT"
fi
python -m pip install -c "$source_dir/constraints-tested.txt" -e "$xpl_root" -e "$source_dir"
