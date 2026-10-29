#!/usr/bin/env bash
# Shared shell utilities; source from the policy's entry points.
set -euo pipefail
POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="${XPOLICYLAB_ROOT:-$(cd -- "$POLICY_DIR/../.." && pwd)}"
BENCH_ROOT="$(cd -- "$XPL_ROOT/.." && pwd)"
# The PrismWAM source shipped with this adapter; set PRISMWAM_ROOT to use another checkout.
PRISMWAM_ROOT="${PRISMWAM_ROOT:-$POLICY_DIR/PrismWAM}"
is_dry_run() { [[ "${DRY_RUN:-0}" == 1 || "${DRY_RUN:-false}" == true ]]; }
show_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run_command() { show_command "$@"; if ! is_dry_run; then "$@"; fi; }
require_xpl() {
    [[ -f "$XPL_ROOT/setup_policy_server.py" && -f "$XPL_ROOT/utils/checkpoint_resolver.py" ]] || {
        echo 'Install this adapter into XPolicyLab/policy/PrismWAM first.' >&2; exit 2;
    }
}
require_prismwam() {
    [[ -f "$PRISMWAM_ROOT/scripts/train.py" ]] || {
        echo "No PrismWAM source at $PRISMWAM_ROOT." >&2; exit 2;
    }
    PRISMWAM_ROOT="$(cd -- "$PRISMWAM_ROOT" && pwd)"
}
activate_policy_env() {
    local target="$1"
    if [[ -f "$target/bin/activate" ]]; then
        source "$target/bin/activate"
    elif [[ "$target" == current ]]; then
        :
    else
        command -v conda >/dev/null || { echo "No venv or conda env: $target" >&2; exit 2; }
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$target"
    fi
}
