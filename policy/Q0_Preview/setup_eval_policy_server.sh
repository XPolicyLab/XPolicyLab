#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 9 ]] || { echo 'Expected the standard XPolicyLab server arguments' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
[[ "$1" == RoboDojo && "$4" == arx_x5 && "$5" == joint ]] || { echo 'Supported: RoboDojo / arx_x5 / joint' >&2; exit 2; }
source "${SCRIPT_DIR}/tools/activate_env.sh" "$8"
export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$7"
export PYTHONDONTWRITEBYTECODE=1
source "${XPL_ROOT}/utils/resolve_eval_env_type.sh"
mode="$(resolve_eval_env_type)"
if [[ "$mode" == debug ]]; then
    exec python "${XPL_ROOT}/setup_policy_server.py" --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
        "port=$9" "host=${10:-localhost}" "bench_name=$1" "task_name=$2" "ckpt_name=$3" \
        "env_cfg_type=$4" "action_type=$5" "seed=$6" "policy_name=$(basename "$SCRIPT_DIR")"
fi
# The supervisor resolves output paths against XPL_ROOT and reserves a unique run.
args=(--port "$9" --host "${10:-localhost}" --task "$2" --seed "$6" --gpu "$7"
      --python-prefix "${Q0_PREVIEW_AGENT_PREFIX:-${Q0_PREVIEW_ACTIVE_PREFIX}}")
if [[ -n "${Q0_PREVIEW_AGENT_CONFIG:-}" ]]; then args+=(--config "$Q0_PREVIEW_AGENT_CONFIG"); fi
exec python -u "${SCRIPT_DIR}/tools/policy_service.py" "${args[@]}"
