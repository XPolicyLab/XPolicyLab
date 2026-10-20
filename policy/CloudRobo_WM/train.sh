#!/usr/bin/env bash
# XPolicyLab positional interface around the fixed upstream training recipe.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <device_ids|all> \
  --data-root /path/to/robotwin3d [--bundle-root /path/to/training_bundle] \
  [--output-dir /path/to/new_run] [--stop-after-steps N] [--resume /path/to/state] [--download-assets] \
  -- <torch.distributed.run arguments...>

Supported recipe: RoboTwin, arx_x5, joint, seed 42.
Use all to inherit device visibility from the launch environment.
For a numeric list, CRWM_DEVICE_FILTER must name the runtime visibility variable.
Full training requires 68 ranks. Pass the rendezvous and process-count
arguments explicitly. A finite smoke run may omit the separator/launcher args;
with --stop-after-steps it defaults to --standalone --nproc-per-node=1.
EOF
}
fail() { printf '[CloudRobo_WM] %s\n' "$*" >&2; exit 2; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# < 6 )); then usage >&2; exit 2; fi
bench_name="$1"; ckpt_name="$2"; env_cfg_type="$3"; action_type="$4"; seed="$5"; device_ids="$6"; shift 6
data_root=""; bundle_root=""; output_dir=""; stop_after_steps=""; resume=""
download_assets=false
torchrun_args=()
while (( $# )); do
    case "$1" in
        --data-root|--bundle-root|--output-dir|--stop-after-steps|--resume)
            (( $# >= 2 )) || fail "$1 requires a value"
            case "$1" in
                --data-root) data_root="$2" ;;
                --bundle-root) bundle_root="$2" ;;
                --output-dir) output_dir="$2" ;;
                --stop-after-steps) stop_after_steps="$2" ;;
                --resume) resume="$2" ;;
            esac
            shift 2 ;;
        --download-assets) download_assets=true; shift ;;
        --) shift; torchrun_args=("$@"); break ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown argument: $1; put torchrun arguments after --" ;;
    esac
done
[[ "$(printf '%s' "$bench_name" | tr '[:upper:]' '[:lower:]')" == "robotwin" ]] || fail "Only the RoboTwin Clean reproduction recipe is supported"
bench_name="RoboTwin"
[[ "$env_cfg_type" == "arx_x5" ]] || fail "This fixed recipe requires env_cfg_type=arx_x5"
[[ "$action_type" == "joint" ]] || fail "This recipe executes Joint14; action_type must be joint"
[[ "$seed" == "42" ]] || fail "The fixed reproduction recipe requires seed=42"
if [[ "$device_ids" != "all" ]]; then
    [[ "$device_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "device_ids must be all or a comma-separated list of local device indices"
    [[ "${CRWM_DEVICE_FILTER:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail "Set CRWM_DEVICE_FILTER to the runtime visibility variable, or pass all"
    export "${CRWM_DEVICE_FILTER}=$device_ids"
fi
[[ -n "$ckpt_name" ]] || fail "ckpt_name must not be empty"
[[ -n "$data_root" ]] || fail "--data-root must explicitly name the native RobotWin3D directory"
if [[ -n "$stop_after_steps" ]]; then
    [[ "$stop_after_steps" =~ ^[1-9][0-9]*$ ]] || fail "--stop-after-steps must be a positive integer"
    if (( ${#stop_after_steps} > 4 )) || (( stop_after_steps >= 5504 )); then
        fail "Finite checks require --stop-after-steps below 5504; omit it for the 68-rank full training run"
    fi
fi
if (( ${#torchrun_args[@]} == 0 )); then
    [[ -n "$stop_after_steps" ]] || fail "Full training requires explicit torchrun arguments after --; see README.md"
    torchrun_args=(--standalone --nproc-per-node=1)
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$POLICY_DIR/common.sh"
POLICY_DIR="$(crwm_policy_dir)"
XPL_DIR="$(crwm_xpl_dir)"
POSTTRAIN_DIR="$POLICY_DIR/source_posttrain"
PYTHON="$(crwm_python posttrain)"
checkpoint_root="$(crwm_checkpoint_root "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed" false)"
bundle_root="${bundle_root:-$checkpoint_root/training}"
output_dir="${output_dir:-$checkpoint_root/posttrain_outputs}"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Post-training Python is unavailable: $PYTHON"
data_root="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$data_root")"
bundle_root="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$bundle_root")"
output_dir="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_dir")"
if [[ -n "$resume" ]]; then
    resume="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$resume")"
fi
[[ -f "$XPL_DIR/utils/get_action_dim.sh" ]] || fail "Official XPolicyLab action-dimension helper is missing"
action_dim="$(bash "$XPL_DIR/utils/get_action_dim.sh" "$(dirname "$XPL_DIR")" "$env_cfg_type")"
[[ "$action_dim" == "14" ]] || fail "Official robot dimensions must total Joint14; got $action_dim"

# Resolve symlinks before checking output isolation. Never write into the bundle,
# and never silently append a fresh run to an existing nonempty output directory.
"$PYTHON" - "$bundle_root" "$output_dir" "$resume" <<'PY'
from pathlib import Path
import sys
bundle, output = (Path(value).expanduser().resolve() for value in sys.argv[1:3])
resume = sys.argv[3]
if output == bundle or output.is_relative_to(bundle):
    raise SystemExit('Training output must be outside the training bundle directory.')
if output.exists() and not output.is_dir():
    raise SystemExit('Training output exists and is not a directory.')
if output.is_dir() and any(output.iterdir()) and not resume:
    raise SystemExit('Fresh training requires a new or empty output directory; use --resume for an existing run.')
if resume and not (Path(resume).expanduser() / 'trainer_state.json').is_file():
    raise SystemExit('--resume must name a full state directory containing trainer_state.json.')
PY

# This is the upstream CPU data/asset check, before any distributed model launch.
precheck_args=(--data-root "$data_root" --bundle-root "$bundle_root")
if [[ "$download_assets" == true ]]; then precheck_args+=(--download-assets); fi
bash "$POLICY_DIR/process_data.sh" "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type" "${precheck_args[@]}"

training_args=(
    --data-root "$data_root"
    --text-cache-root "$bundle_root/text_cache"
    --sidecar-root "$bundle_root/eef_sidecars"
    --assets-root "$bundle_root/assets"
    --initial-weights "$bundle_root/initialization_bf16.pt"
    --output-dir "$output_dir"
)
if [[ -n "$stop_after_steps" ]]; then training_args+=(--stop-after-steps "$stop_after_steps"); fi
if [[ -n "$resume" ]]; then training_args+=(--resume "$resume"); fi
export PYTHONPATH="$POSTTRAIN_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$POSTTRAIN_DIR"
printf '[CloudRobo_WM] Training output: %s\n' "$output_dir"
exec "$PYTHON" -m torch.distributed.run "${torchrun_args[@]}" --module crwm.train "${training_args[@]}"
