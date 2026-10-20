#!/usr/bin/env bash
# Validate the exact native RobotWin3D Clean subset; do not convert or relabel it.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  --data-root /path/to/robotwin3d [--bundle-root /path/to/training_bundle] [--download-assets]

Supported recipe: RoboTwin, arx_x5, joint. The data root is required.
The default bundle root is <resolved checkpoint root>/training.
This runs the upstream CPU precheck. --download-assets fetches the pinned public
training resources and metadata first. Native media must already be available.
EOF
}
fail() { printf '[CloudRobo_WM] %s\n' "$*" >&2; exit 2; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# < 4 )); then usage >&2; exit 2; fi
bench_name="$1"; ckpt_name="$2"; env_cfg_type="$3"; action_type="$4"; shift 4
data_root=""; bundle_root=""; download_assets=false
while (( $# )); do
    case "$1" in
        --data-root|--bundle-root)
            (( $# >= 2 )) || fail "$1 requires a value"
            case "$1" in
                --data-root) data_root="$2" ;;
                --bundle-root) bundle_root="$2" ;;
            esac
            shift 2 ;;
        --download-assets) download_assets=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done
[[ "$(printf '%s' "$bench_name" | tr '[:upper:]' '[:lower:]')" == "robotwin" ]] || fail "Only the RoboTwin Clean reproduction recipe is supported"
bench_name="RoboTwin"
[[ "$env_cfg_type" == "arx_x5" ]] || fail "This fixed recipe requires env_cfg_type=arx_x5"
[[ "$action_type" == "joint" ]] || fail "This recipe executes Joint14; action_type must be joint"
[[ -n "$ckpt_name" ]] || fail "ckpt_name must not be empty"
[[ -n "$data_root" ]] || fail "--data-root must explicitly name the native RobotWin3D directory"

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$POLICY_DIR/common.sh"
POLICY_DIR="$(crwm_policy_dir)"
POSTTRAIN_DIR="$POLICY_DIR/source_posttrain"
PYTHON="$(crwm_python posttrain)"
if [[ -z "$bundle_root" ]]; then
    checkpoint_root="$(crwm_checkpoint_root "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type" 42 true)"
    bundle_root="$checkpoint_root/training"
fi
[[ -f "$POSTTRAIN_DIR/src/crwm/precheck.py" ]] || fail "Post-training source is missing; run install.sh for the posttrain stage"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Post-training Python is unavailable: $PYTHON"
data_root="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$data_root")"
bundle_root="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$bundle_root")"
[[ -d "$data_root" ]] || fail "Data root does not exist: $data_root"
if [[ "$download_assets" == true ]]; then
    "$PYTHON" "$POLICY_DIR/download_release.py" training --output-dir "$bundle_root"
    "$PYTHON" "$POLICY_DIR/download_release.py" metadata --output-dir "$data_root"
fi
[[ -f "$bundle_root/initialization_bf16.pt" ]] || fail "Missing training initialization: $bundle_root/initialization_bf16.pt (see README.md)"
for directory in assets text_cache eef_sidecars; do
    [[ -d "$bundle_root/$directory" ]] || fail "Missing training bundle directory: $bundle_root/$directory"
done
export PYTHONPATH="$POSTTRAIN_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$POSTTRAIN_DIR"
exec "$PYTHON" -m crwm.precheck \
    --data-root "$data_root" \
    --text-cache-root "$bundle_root/text_cache" \
    --sidecar-root "$bundle_root/eef_sidecars" \
    --assets-root "$bundle_root/assets" \
    --initial-weights "$bundle_root/initialization_bf16.pt"
