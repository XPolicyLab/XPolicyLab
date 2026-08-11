#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
shift 4 || true

if [[ "${bench_name,,}" != "robodojo" ]]; then
  echo "CSU-AI-05 process_data.sh supports bench_name=RoboDojo, got ${bench_name}" >&2
  exit 2
fi
if [[ "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
  echo "CSU-AI-05 expects env_cfg_type=arx_x5 and action_type=joint." >&2
  exit 2
fi

DATA_ROOT=${ROBODOJO_LEROBOT_V30_ROOT:?Set ROBODOJO_LEROBOT_V30_ROOT to the prepared RoboDojo LeRobot v3.0 dataset}
for required in meta data videos; do
  if [[ ! -e "${DATA_ROOT}/${required}" ]]; then
    echo "Invalid ROBODOJO_LEROBOT_V30_ROOT: missing ${required}/ under ${DATA_ROOT}" >&2
    exit 3
  fi
done

echo "[CSU-AI-05] RoboDojo LeRobot v3.0 dataset is ready: ${DATA_ROOT}"
echo "[CSU-AI-05] No policy-specific conversion is required for ckpt_name=${ckpt_name}."
if (( $# > 0 )); then
  echo "[CSU-AI-05] Ignored extra processing arguments: $*"
fi
