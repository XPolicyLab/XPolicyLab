#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
if (($# < 8)); then
    echo 'Usage: train.sh BENCH CHECKPOINT ROBOT joint SEED GPU_IDS RECIPE stage1_depth|stage2_joint [KEY=VALUE...]' >&2
    exit 2
fi
require_xpl
require_prismwam
bench="$1"; checkpoint="$2"; robot="$3"; action="$4"; seed="$5"; gpu="$6"; recipe="$7"; stage="$8"
shift 8
[[ "$action" == joint ]] || { echo 'Use native PrismWAM training for non-joint recipes.' >&2; exit 2; }
[[ "$stage" == stage1_depth || "$stage" == stage2_joint ]] || { echo 'Stage must be stage1_depth or stage2_joint.' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Seed must be a nonnegative integer.' >&2; exit 2; }
python_bin="${PYTHON:-python}"
# The official training helper reads the standalone robot registry.
robot_dim="$(bash "$XPL_ROOT/utils/get_action_dim.sh" "$BENCH_ROOT" "$robot")"
mapfile -t resolved < <("$python_bin" - "$PRISMWAM_ROOT" "$POLICY_DIR" "$bench" "$checkpoint" "$robot" "$action" "$seed" "$recipe" "$stage" "$robot_dim" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

root, policy, bench, checkpoint, robot, action, seed, recipe, stage, robot_dim = sys.argv[1:]
sys.path.insert(0, str(Path(root) / "src"))
from prismwam.config import load_config

if recipe not in {"robodojo", "robotwin"} or recipe != bench.lower():
    if recipe == "realworld":
        raise SystemExit("The mixed 7D/14D real-world recipe has task-specific robot contracts. "
                         "Use PrismWAM/scripts/realworld/*.sh directly; this wrapper does not "
                         "choose one robot registry entry for all six data sources.")
    raise SystemExit("RECIPE must be robodojo or robotwin and match BENCH.")
config = Path(root) / "configs" / recipe / f"{stage}.yaml"
dimension = OmegaConf.select(load_config(config), "data.train.action_dim")
if dimension != int(robot_dim):
    raise SystemExit("Robot registry action dimension differs from the selected recipe.")
run = build_run_dir_name(dict(bench_name=bench, ckpt_name=checkpoint,
    env_cfg_type=robot, action_type=action, seed=seed))
if not run or Path(run).name != run:
    raise SystemExit("Training CHECKPOINT must be a short run name, not a filesystem path.")
print(config)
print(Path(policy) / "outputs" / run)
PY
)
[[ ${#resolved[@]} == 2 ]] || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHON="$python_bin"
dry_run=()
if is_dry_run; then dry_run=(--dry-run); fi
# Stage outputs go to outputs/<run>/<recipe>/<stage>, so Stage 2 finds Stage 1.
exec bash "$PRISMWAM_ROOT/scripts/train.sh" "${resolved[0]}" "${dry_run[@]}" \
    "seed=$seed" "paths.output_root=${resolved[1]}" \
    "paths.robodojo_root=${PRISMWAM_ROBODOJO_ROOT:-$BENCH_ROOT}" \
    "paths.robotwin_root=${PRISMWAM_ROBOTWIN_ROOT:-$BENCH_ROOT}" "$@"
