#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 9 && $# -le 10 ]] || { echo 'Usage: server bench task checkpoint robot action seed gpu policy_env port [bind_host]' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime/env.sh"
activate_policy_env "$8"
diagnostic=False
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then diagnostic=True; fi
exec python "$XPL_ROOT/setup_policy_server.py" --config_path "$SCRIPT_DIR/deploy.yml" --overrides \
    "bench_name=$1" "task_name=$2" "ckpt_name=$3" "env_cfg_type=$4" "action_type=$5" \
    "seed=$6" "gpu_id=$7" "port=$9" "host=${10:-127.0.0.1}" "diagnostic=$diagnostic" \
    "agent_image=${EMBODIEDRSI_AGENT_IMAGE:-embodiedrsi-agent:xpolicy}"
