#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 10 && $# -le 11 ]] || { echo 'Usage: client bench task checkpoint robot action seed gpu eval_env additional_info port [server_host]' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
source "$SCRIPT_DIR/runtime/env.sh"
activate_policy_env "$8"
source "$XPL_ROOT/utils/resolve_eval_env_type.sh"
mode="$(resolve_eval_env_type)"
eval_batch="$(python -c 'import sys,yaml; print(str(yaml.safe_load(open(sys.argv[1]))["eval_batch"]).lower())' "$SCRIPT_DIR/deploy.yml")"
[[ "$1" == RoboDojo ]] || { echo 'Only RoboDojo is supported' >&2; exit 2; }
if [[ "$mode" == debug ]]; then
    exec python "$XPL_ROOT/utils/debug_env_client.py" --bench_name "$1" --task_name "$2" \
        --env_cfg_type "$4" --policy_name EmbodiedRSI --protocol ws --host "${11:-127.0.0.1}" \
        --port "${10}" --eval_batch "$eval_batch" --action_case_id embodiedrsi-debug \
        --eval_episode_num "${EMBODIEDRSI_DEBUG_EPISODES:-2}"
elif [[ "$mode" == sim ]]; then
    [[ -f "$BENCH_ROOT/scripts/eval_policy.sh" ]] || {
        echo 'Install the official RoboDojo workspace around this XPolicyLab checkout (scripts/, env_cfg/, ...). See README.md.' >&2; exit 2;
    }
    exec bash "$BENCH_ROOT/scripts/eval_policy.sh" --bench_name "$1" --task_name "$2" \
        --env_cfg_type "$4" --policy_name EmbodiedRSI --protocol ws --host "${11:-127.0.0.1}" \
        --port "${10}" --eval_batch "$eval_batch" --root_dir "$BENCH_ROOT" --device_id "$7" \
        --additional_info "$9" --seed "$6"
else
    echo 'This release supports official RoboDojo simulation and interface debug only.' >&2
    exit 2
fi
