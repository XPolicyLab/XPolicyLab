#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 6 && $# -le 7 ]] || { echo "Usage: $0 bench ckpt env_cfg action seed gpu_ids [all|router|action]" >&2; exit 2; }
bench=$1 ckpt=$2 env_cfg=$3 action=$4 seed=$5 gpu_ids=$6 stage=${7:-all}
[[ $bench == RoboTwin && $env_cfg == arx_x5 && $action == joint ]] || {
    echo "Released training recipe: RoboTwin arx_x5 joint only" >&2; exit 2;
}
case "$stage" in all|router|action) ;; *) echo "Invalid stage" >&2; exit 2 ;; esac
: "${WORLDLA_REPRO_ROOT:?unpacked 20K bundle}"
: "${WORLDLA_LEROBOT_ROOT:?matching converted LeRobot dataset parent}"
: "${WORLDLA_INIT_PARAMS:?pretrained pi0.5 base params, not fine-tuned WorldLA}"
: "${WORLDLA_TOKENIZER:?paligemma_tokenizer.model}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$(dirname "$XPL_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
run_name="$(python - "$bench" "$ckpt" "$env_cfg" "$action" "$seed" <<'PY'
import sys
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
cfg = dict(zip(('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed'), sys.argv[1:]))
assert all('/' not in value and value not in ('.', '..') for value in cfg.values())
print(build_run_dir_name(cfg))
PY
)"
action_dim="$(bash "$XPL_ROOT/utils/get_action_dim.sh" "$(dirname "$XPL_ROOT")" "$env_cfg")"
[[ $action_dim == 14 ]] || { echo "Incompatible action dimension" >&2; exit 2; }
run_dir="$SCRIPT_DIR/checkpoints/$run_name"
export WORLDLA_OUTPUT_ROOT="${WORLDLA_OUTPUT_ROOT:-$run_dir/training}"
export CUDA_VISIBLE_DEVICES="$gpu_ids" PYTHONUNBUFFERED=1
mkdir -p "$WORLDLA_OUTPUT_ROOT/logs"
if [[ $stage == all || $stage == router ]]; then
    python -m worldla train --config "$WORLDLA_REPRO_ROOT/configs/router_60k.yaml" \
        --seed "$seed" --fsdp-devices "${WORLDLA_FSDP_DEVICES:-8}" \
        2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/router.log"
fi
if [[ $stage == all || $stage == action ]]; then
    python - "$WORLDLA_OUTPUT_ROOT/router_60k/experiment.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))['completed'] is True, 'Router training is not complete'
PY
    python -m worldla train --config "$WORLDLA_REPRO_ROOT/configs/action_100k.yaml" \
        --seed "$seed" --fsdp-devices "${WORLDLA_FSDP_DEVICES:-8}" \
        2>&1 | tee "$WORLDLA_OUTPUT_ROOT/logs/action.log"
    python -m worldla export --model-dir "$WORLDLA_OUTPUT_ROOT/action_100k" \
        --variant original --tokenizer "$WORLDLA_TOKENIZER" --output-dir "$run_dir/original"
fi
