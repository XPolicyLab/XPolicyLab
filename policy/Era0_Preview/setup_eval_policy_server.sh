#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 9 ]] || { echo 'Expected standard 9 server arguments' >&2; exit 2; }
resolve_python "$8"
export CUDA_VISIBLE_DEVICES="$7"
checkpoint_args=()
if [[ -n "${ERA0_CHECKPOINT_PATH:-}" ]]; then
    checkpoint_args+=("model_path=${ERA0_CHECKPOINT_PATH}")
fi
exec "${PYTHON_BIN}" -P -u "${XPL_ROOT}/setup_policy_server.py" \
 --config_path "${POLICY_DIR}/deploy.yml" --overrides \
 bench_name="$1" task_name="$2" ckpt_name="$3" env_cfg_type="$4" action_type="$5" \
 seed="$6" port="$9" host="${10:-localhost}" policy_name=Era0_Preview "${checkpoint_args[@]}"
