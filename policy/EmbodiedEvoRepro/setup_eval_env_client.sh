#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 10 ]] || { echo 'Usage: setup_eval_env_client.sh bench task ckpt robot action seed gpu python_env additional_info port [host]' >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/shell_common.sh"
validate_mode
[[ $1 == RoboDojo ]] || { echo "[ERROR] This adapter supports RoboDojo only" >&2; exit 2; }
activate_python "$8"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    exec python "${XPL_ROOT}/utils/debug_env_client.py" \
        --bench_name "$1" --task_name "$2" --env_cfg_type "$4" \
        --policy_name EmbodiedEvoRepro --protocol ws --host "${11:-localhost}" \
        --port "${10}" --eval_batch false --eval_episode_num "${EMBODIEDEVO_DEBUG_EPISODES:-1}"
fi
[[ -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]] || { echo '[ERROR] Install XPolicyLab inside the RoboDojo workspace for simulation' >&2; exit 1; }
exec bash "${BENCH_ROOT}/scripts/eval_policy.sh" \
    --task_name "$2" --env_cfg_type "$4" --policy_name EmbodiedEvoRepro \
    --host "${11:-localhost}" --port "${10}" --protocol ws --eval_batch false \
    --root_dir "${BENCH_ROOT}" --device_id "$7" --additional_info "$9" --seed "$6"
