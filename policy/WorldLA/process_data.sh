#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { echo "Usage: $0 bench_name ckpt_name env_cfg_type action_type" >&2; exit 2; }
[[ $1 == RoboTwin && $3 == arx_x5 && $4 == joint ]] || {
    echo "Released dataset: RoboTwin arx_x5 joint only" >&2; exit 2;
}
: "${WORLDLA_REPRO_ROOT:?unpack the released 20K training bundle first}"
: "${WORLDLA_RAW_ROOT:?point to the official extracted demo_clean dataset}"
: "${WORLDLA_LEROBOT_ROOT:?parent of robotwin_official50_demo_clean_v21}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
mkdir -p "$WORLDLA_LEROBOT_ROOT"
inventory="$WORLDLA_LEROBOT_ROOT/official_inventory.jsonl"
python "$WORLDLA_REPRO_ROOT/tools/audit_official_data.py" \
    --data-root "$WORLDLA_RAW_ROOT" \
    --task-list "$WORLDLA_REPRO_ROOT/dataset_alignment/task_order.yaml" --output "$inventory"
python "$WORLDLA_REPRO_ROOT/tools/convert_official_lerobot.py" \
    --inventory "$inventory" \
    --dataset-root "$WORLDLA_LEROBOT_ROOT/robotwin_official50_demo_clean_v21" \
    --repo-id robotwin_official50_demo_clean_v21 --xpolicylab-parent "$BENCH_ROOT" --seed 20261004
JAX_PLATFORMS=cpu python "$WORLDLA_REPRO_ROOT/tools/verify_bundle.py" \
    --root "$WORLDLA_REPRO_ROOT" \
    --dataset-root "$WORLDLA_LEROBOT_ROOT/robotwin_official50_demo_clean_v21" --worldla-loader
