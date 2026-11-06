#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_WORLD_ARENA2_ROOT="/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/WorldArena2"
WORLD_ARENA2_ROOT="${1:-${WORLD_ARENA2_ROOT:-${DEFAULT_WORLD_ARENA2_ROOT}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAMPLE_LOAD="${SAMPLE_LOAD:-32}"
CHECK_VIDEOS="${CHECK_VIDEOS:-true}"
COUNT_VIDEO_FRAMES="${COUNT_VIDEO_FRAMES:-false}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-false}"
RUN_ID="${RUN_ID:-$(date +'%Y_%m_%d_%H_%M_%S')}"
REPORT_ROOT="${REPORT_ROOT:-${PROJ_ROOT}/outputs/integrity/worldarena2/${RUN_ID}}"

TASKS=(
  clean_table
  clean_table_instruction_follow
  fold_box
  fold_shirt
  insert
  peel_cucumber
  pick_potato_chip
  pour_over_coffee
  pour_water
  wipe_table
)

case "${CHECK_VIDEOS}" in
  true|false) ;;
  *) echo "CHECK_VIDEOS must be true or false, got ${CHECK_VIDEOS}"; exit 2 ;;
esac
case "${COUNT_VIDEO_FRAMES}" in
  true|false) ;;
  *) echo "COUNT_VIDEO_FRAMES must be true or false, got ${COUNT_VIDEO_FRAMES}"; exit 2 ;;
esac
case "${ALLOW_PARTIAL}" in
  true|false) ;;
  *) echo "ALLOW_PARTIAL must be true or false, got ${ALLOW_PARTIAL}"; exit 2 ;;
esac
if ! [[ "${SAMPLE_LOAD}" =~ ^[0-9]+$ ]]; then
  echo "SAMPLE_LOAD must be a non-negative integer, got ${SAMPLE_LOAD}"
  exit 2
fi
if [[ ! -d "${WORLD_ARENA2_ROOT}" ]]; then
  echo "WorldArena2 root does not exist: ${WORLD_ARENA2_ROOT}"
  exit 2
fi

mkdir -p "${REPORT_ROOT}"

echo "WorldArena2 integrity check"
echo "ROOT=${WORLD_ARENA2_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "SAMPLE_LOAD=${SAMPLE_LOAD}"
echo "CHECK_VIDEOS=${CHECK_VIDEOS}"
echo "COUNT_VIDEO_FRAMES=${COUNT_VIDEO_FRAMES}"
echo "ALLOW_PARTIAL=${ALLOW_PARTIAL}"

FAILED_TASKS=()
PASSED=0

for task in "${TASKS[@]}"; do
  dataset_dir="${WORLD_ARENA2_ROOT}/${task}"
  report_path="${REPORT_ROOT}/${task}.json"

  echo
  echo "================================================================"
  echo "Checking ${task}: ${dataset_dir}"
  echo "================================================================"

  if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
    echo "ERROR: missing ${dataset_dir}/meta/info.json"
    FAILED_TASKS+=("${task}")
    continue
  fi

  ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/check_lerobot_v3_integrity.py"
    --dataset "${dataset_dir}"
    --deep
    --sample-load "${SAMPLE_LOAD}"
    --json-output "${report_path}"
  )
  if [[ "${CHECK_VIDEOS}" == "true" ]]; then
    ARGS+=(--check-videos)
  fi
  if [[ "${COUNT_VIDEO_FRAMES}" == "true" ]]; then
    ARGS+=(--count-video-frames)
  fi
  if [[ "${ALLOW_PARTIAL}" == "true" ]]; then
    ARGS+=(--allow-partial)
  fi

  if "${ARGS[@]}"; then
    PASSED=$((PASSED + 1))
  else
    FAILED_TASKS+=("${task}")
  fi
done

echo
echo "================================================================"
echo "WorldArena2 integrity summary"
echo "passed=${PASSED}/${#TASKS[@]}"
echo "reports=${REPORT_ROOT}"

if (( ${#FAILED_TASKS[@]} > 0 )); then
  echo "failed=${#FAILED_TASKS[@]}"
  printf '  - %s\n' "${FAILED_TASKS[@]}"
  exit 1
fi

echo "All ${#TASKS[@]} datasets passed."
