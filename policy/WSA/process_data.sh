#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [max_episode]" >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
max_episode=${5:-200}

if [[ "${action_type}" != "joint" ]]; then
    echo "[ERROR] WSA RoboTwin training supports only XPolicyLab action_type=joint." >&2
    exit 1
fi
if [[ "${env_cfg_type}" != "aloha_agilex" ]]; then
    echo "[ERROR] WSA RoboTwin training supports only env_cfg_type=aloha_agilex." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WSA_DIR="${WSA_ROOT:-${POLICY_DIR}/WSA}"

dataset_id="${WSA_DATASET_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
data_pattern="${WSA_DATA_PATTERN:-${bench_name}.*.${env_cfg_type}}"
export HF_LEROBOT_HOME="${WSA_LEROBOT_HOME:-${POLICY_DIR}/data/lerobot}"
action_mode="${WSA_ACTION_MODE:-delta}"
source_data_type="${WSA_SOURCE_DATA_TYPE:-RoboDojo}"
stats_root="${WSA_STATS_ROOT:-${POLICY_DIR}/data/norm_stats}"
stats_path="${stats_root}/aloha/${action_mode}/stats.json"
repo_dir="${HF_LEROBOT_HOME}/${dataset_id}"
repo_id_file="${POLICY_DIR}/data/${dataset_id}.txt"

mkdir -p "${HF_LEROBOT_HOME}" "$(dirname "${stats_path}")" "${POLICY_DIR}/data"

python "${XPL_ROOT}/scripts/transform_lerobot_v30_format.py" \
    "${data_pattern}" \
    --repo_id "${dataset_id}" \
    --data_type "${source_data_type}" \
    --max_episode "${max_episode}"

# The official converter deliberately labels mixed-robot exports as
# `unified_robot`. This adapter supports only the ALOHA-shaped RoboTwin data,
# whose keys already match WSA's built-in lowercase `aloha` mapping. Retag the
# generated single-embodiment metadata so WSA's dataset/statistics loaders can
# select that mapping without changing any tensor or image fields.
python - "${repo_dir}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
if not info_path.is_file():
    raise FileNotFoundError(f"Converted LeRobot metadata not found: {info_path}")
info = json.loads(info_path.read_text(encoding="utf-8"))
info["robot_type"] = "aloha"
info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
PY

printf '%s\n' "${repo_dir}" > "${repo_id_file}"

if [[ "${WSA_SKIP_STATS:-false}" != "true" ]]; then
    chunk_size="${WSA_STATS_CHUNK_SIZE:-50}"
    python "${WSA_DIR}/tools/compute_norm_stats_multi.py" \
        --repo_id_file "${repo_id_file}" \
        --action_mode "${action_mode}" \
        --chunk_size "${chunk_size}" \
        --num_workers "${WSA_STATS_NUM_WORKERS:-8}" \
        --output_path "${stats_path}"
fi

echo "[WSA] LeRobot v3.0 dataset: ${repo_dir}"
echo "[WSA] normalization stats: ${stats_path}"
