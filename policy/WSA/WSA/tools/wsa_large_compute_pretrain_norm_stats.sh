#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

INTERNDATA_ROOT="${INTERNDATA_ROOT:-/path/to/InternData-A1-v30}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/RoboTwin-LeRobot-v30}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-/path/to/Robochallengev3.0_eef}"
AGIBOT_ROOT="${AGIBOT_ROOT:-/path/to/Agibotv3.0}"
EGODEX_LEROBOT_ROOT="${EGODEX_LEROBOT_ROOT:-/path/to/Egodex_v_taskrepos_v30}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_STATS_ROOT="${OUTPUT_STATS_ROOT:-norm_stats_chunk${CHUNK_SIZE}}"
GROUP_FILE_DIR="${GROUP_FILE_DIR:-outputs/WSA_Large/_stats_repo_id_files/chunk${CHUNK_SIZE}}"
OVERWRITE_STATS="${OVERWRITE_STATS:-false}"
SAMPLED_ROBOT_TYPES="${SAMPLED_ROBOT_TYPES:-}"
SAMPLE_MAX_CHUNKS_PER_EPISODE="${SAMPLE_MAX_CHUNKS_PER_EPISODE:-}"
SAMPLE_MAX_CHUNKS_PER_REPO="${SAMPLE_MAX_CHUNKS_PER_REPO:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SKIP_ACTION_ROBOT_TYPES="${SKIP_ACTION_ROBOT_TYPES:-}"
ZERO_STATS_ROBOT_TYPES="${ZERO_STATS_ROBOT_TYPES:-}"
EXCLUDE_ROBOT_TYPES="${EXCLUDE_ROBOT_TYPES:-}"

case "${ACTION_TYPE}" in
  abs|delta)
    ;;
  *)
    echo "Unsupported ACTION_TYPE=${ACTION_TYPE}. Expected abs or delta."
    exit 1
    ;;
esac

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
          echo "${ds_dir}"
        fi
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(
  {
    discover_dataset_dirs "${INTERNDATA_ROOT}"
    discover_dataset_dirs "${ROBOTWIN_ROOT}"
    discover_dataset_dirs "${ROBOCHALLENGE_ROOT}"
    discover_dataset_dirs "${AGIBOT_ROOT}"
    discover_dataset_dirs "${EGODEX_LEROBOT_ROOT}"
  } | sort -u
)

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid LeRobot datasets found."
  echo "Please set one or more of: INTERNDATA_ROOT ROBOTWIN_ROOT ROBOCHALLENGE_ROOT AGIBOT_ROOT EGODEX_LEROBOT_ROOT"
  exit 1
fi

mkdir -p "${GROUP_FILE_DIR}" "${OUTPUT_STATS_ROOT}"
ALL_DATASETS_FILE="${GROUP_FILE_DIR}/all_datasets.txt"
GROUP_MANIFEST="${GROUP_FILE_DIR}/manifest.tsv"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${ALL_DATASETS_FILE}"

echo "Discovered ${#DATASET_REPO_IDS[@]} datasets"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "OUTPUT_STATS_ROOT=${OUTPUT_STATS_ROOT}"
echo "GROUP_FILE_DIR=${GROUP_FILE_DIR}"
echo "SAMPLED_ROBOT_TYPES=${SAMPLED_ROBOT_TYPES:-<none>}"
echo "SAMPLE_MAX_CHUNKS_PER_EPISODE=${SAMPLE_MAX_CHUNKS_PER_EPISODE:-<disabled>}"
echo "SAMPLE_MAX_CHUNKS_PER_REPO=${SAMPLE_MAX_CHUNKS_PER_REPO:-<disabled>}"
echo "SKIP_ACTION_ROBOT_TYPES=${SKIP_ACTION_ROBOT_TYPES:-<none>}"
if [[ -n "${SKIP_ACTION_ROBOT_TYPES}" ]]; then
  echo "  note: skipped action robot types still get zero action stats written"
fi
echo "ZERO_STATS_ROBOT_TYPES=${ZERO_STATS_ROBOT_TYPES:-<none>}"
if [[ -n "${ZERO_STATS_ROBOT_TYPES}" ]]; then
  echo "  note: zero-stats robot types are handled from metadata only"
fi
echo "EXCLUDE_ROBOT_TYPES=${EXCLUDE_ROBOT_TYPES:-<none>}"

python - "${ALL_DATASETS_FILE}" "${GROUP_FILE_DIR}" "${GROUP_MANIFEST}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from lerobot.transforms.constants import (
    get_feature_mapping,
    get_image_mapping,
    get_mask_mapping,
    infer_embodiment_variant,
)


all_datasets_file = Path(sys.argv[1])
group_file_dir = Path(sys.argv[2])
manifest = Path(sys.argv[3])


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "robot_type"


groups: dict[str, list[str]] = defaultdict(list)
for dataset_dir in all_datasets_file.read_text(encoding="utf-8").splitlines():
    dataset_dir = dataset_dir.strip()
    if not dataset_dir:
        continue
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    robot_type = info["robot_type"]
    features = info.get("features", {})
    resolved_robot_type = infer_embodiment_variant(robot_type, features)
    get_feature_mapping(robot_type, features)
    get_image_mapping(robot_type, features)
    get_mask_mapping(robot_type, features)
    groups[resolved_robot_type].append(dataset_dir)

lines = []
for robot_type in sorted(groups):
    group_path = group_file_dir / f"{sanitize(robot_type)}.txt"
    group_path.write_text("\n".join(groups[robot_type]) + "\n", encoding="utf-8")
    lines.append(f"{robot_type}\t{group_path}\t{len(groups[robot_type])}")

manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} robot-type groups to {manifest}")
for line in lines:
    robot_type, group_path, count = line.split("\t")
    print(f"  {robot_type}: {count} datasets -> {group_path}")
PY

trim_csv_item() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

csv_contains() {
  local needle="$1"
  local csv="$2"
  local item
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    item="$(trim_csv_item "${item}")"
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

build_skip_action_args() {
  local csv="$1"
  local item
  SKIP_ACTION_ARGS=()
  if [[ -z "${csv}" ]]; then
    return 0
  fi
  SKIP_ACTION_ARGS+=(--skip_action_robot_types)
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    item="$(trim_csv_item "${item}")"
    if [[ -n "${item}" ]]; then
      SKIP_ACTION_ARGS+=("${item}")
    fi
  done
}

build_skip_action_args "${SKIP_ACTION_ROBOT_TYPES}"

build_zero_stats_args() {
  local csv="$1"
  local item
  ZERO_STATS_ARGS=()
  if [[ -z "${csv}" ]]; then
    return 0
  fi
  ZERO_STATS_ARGS+=(--zero_stats_robot_types)
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    item="$(trim_csv_item "${item}")"
    if [[ -n "${item}" ]]; then
      ZERO_STATS_ARGS+=("${item}")
    fi
  done
}

build_zero_stats_args "${ZERO_STATS_ROBOT_TYPES}"

while IFS=$'\t' read -r robot_type repo_id_file dataset_count; do
  if [[ -z "${robot_type}" ]]; then
    continue
  fi

  if csv_contains "${robot_type}" "${EXCLUDE_ROBOT_TYPES}"; then
    echo "Skip excluded robot_type=${robot_type}, datasets=${dataset_count}"
    continue
  fi

  stats_path="${OUTPUT_STATS_ROOT}/${robot_type}/${ACTION_TYPE}/stats.json"
  if [[ -f "${stats_path}" && "${OVERWRITE_STATS}" != "true" ]]; then
    echo "Skip existing stats: ${stats_path}"
    continue
  fi

  echo "Computing stats for robot_type=${robot_type}, datasets=${dataset_count}"
  EXTRA_ARGS=("${SKIP_ACTION_ARGS[@]}" "${ZERO_STATS_ARGS[@]}")
  if csv_contains "${robot_type}" "${SAMPLED_ROBOT_TYPES}"; then
    echo "  sampled stats enabled for ${robot_type}"
    if [[ -n "${SAMPLE_MAX_CHUNKS_PER_EPISODE}" ]]; then
      EXTRA_ARGS+=(--max_chunks_per_episode "${SAMPLE_MAX_CHUNKS_PER_EPISODE}")
    fi
    if [[ -n "${SAMPLE_MAX_CHUNKS_PER_REPO}" ]]; then
      EXTRA_ARGS+=(--max_chunks_per_repo "${SAMPLE_MAX_CHUNKS_PER_REPO}")
    fi
    EXTRA_ARGS+=(--sample_seed "${SAMPLE_SEED}")
  fi

  python tools/compute_norm_stats_multi.py \
    --repo_id_file "${repo_id_file}" \
    --action_mode "${ACTION_TYPE}" \
    --chunk_size "${CHUNK_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --output_path "${stats_path}" \
    "${EXTRA_ARGS[@]}"
done < "${GROUP_MANIFEST}"

echo "Done. Use DATASET_EXTERNAL_STATS_ROOT=${OUTPUT_STATS_ROOT} for WSA_Large pretraining."
