#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 RoboTwin clean env_cfg_type joint seed gpu_ids [config overrides...]" >&2
    exit 2
fi
bench=$1; checkpoint_name=$2; robot=$3; action_type=$4; seed=$5; gpu_ids=$6
shift 6
[[ "$bench" == RoboTwin && "$action_type" == joint ]] || { echo "Only RoboTwin joint training is supported." >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
model_root="${DF_ROOT:-${SCRIPT_DIR}/source_discrete_forcing}"
run_id="${bench}-${checkpoint_name}-${robot}-${action_type}-${seed}"
data_root="${DF_DATA_ROOT:-${SCRIPT_DIR}/data/${bench}-${checkpoint_name}-${robot}-${action_type}}"
[[ -d "$data_root/Clean" ]] || { echo "Missing prepared Clean/ data at $data_root; run process_data.sh." >&2; exit 1; }
action_dim=$(bash "$XPL_ROOT/utils/get_action_dim.sh" "${XPL_ROOT}/.." "$robot")
[[ "$action_dim" == 14 ]] || { echo "This release requires 14D robot actions." >&2; exit 2; }
[[ ! -e "$SCRIPT_DIR/checkpoints/$run_id" ]] || { echo "Run already exists: $run_id; choose a new ckpt_name or seed." >&2; exit 1; }
num_processes=$(awk -F, '{print NF}' <<< "$gpu_ids")
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export PYTHONPATH="$model_root:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$model_root"
exec accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "$num_processes" --main_process_port "${MAIN_PROCESS_PORT:-0}" \
    starVLA/training/train_starvla.py \
    --config_yaml examples/Robotwin/train_files/robotwin_clean_method.yaml \
    run_root_dir="$SCRIPT_DIR/checkpoints" run_id="$run_id" seed="$seed" \
    datasets.vla_data.data_root_dir="$data_root" "$@"
