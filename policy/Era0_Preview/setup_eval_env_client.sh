#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 10 ]] || { echo 'Expected standard 10 client arguments' >&2; exit 2; }
resolve_python "$8"
export CUDA_VISIBLE_DEVICES="$7"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    exec "${PYTHON_BIN}" "${XPL_ROOT}/utils/debug_env_client.py" \
      --bench_name "$1" --task_name "$2" --env_cfg_type "$4" \
      --policy_name Era0_Preview --protocol ws --host "${11:-localhost}" \
      --port "${10}" --eval_batch true
fi
[[ "$1" == RoboDojo && "$4" == arx_x5 && "$5" == ee ]] || { echo 'Only RoboDojo/arx_x5/ee supported' >&2; exit 2; }
# Explicit provenance; no simulator/physics files are modified.
# SDK rendering and current observations are the deployment defaults.
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_NUM="${EVAL_NUM:-native}"
export ROBODOJO_NUM_ENVS="${ROBODOJO_NUM_ENVS:-10}"
# SDK owns all result/video paths; provenance is printed into the launch log.
"${PYTHON_BIN}" "${POLICY_DIR}/simulator_entry.py" --describe
cd "${ROBODOJO_ROOT}"
exec "${PYTHON_BIN}" -P -u "${POLICY_DIR}/simulator_entry.py" \
 --task_name "$2" --env_cfg_type "$4" --num_envs "${ROBODOJO_NUM_ENVS}" \
 --enable_cameras --kit_args '--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera' \
 --device_id 0 --policy_name Era0_Preview --port "${10}" --protocol ws \
 --policy_server_url "ws://${11:-localhost}:${10}" --host "${11:-localhost}" \
 --additional_info "$9" --seed "$6" --headless
