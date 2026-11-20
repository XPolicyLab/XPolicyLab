#!/usr/bin/env bash
set -euo pipefail

# Read one absolute local LeRobot v3 dataset path per line and compute an
# independent normalization JSON for every dataset using the existing tool.
#
# Usage:
#   REPO_ID_FILE=/path/to/repo_ids.txt \
#   OUTPUT_PATH=/path/to/output_directory \
#   ACTION_MODE=delta \
#   CHUNK_SIZE=50 \
#   NUM_WORKERS=1 \
#   bash tools/compute_norm_stats_per_dataset.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REPO_ID_FILE="${REPO_ID_FILE:?Set REPO_ID_FILE to the dataset-list text file}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH to the output directory}"
ACTION_MODE="${ACTION_MODE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${REPO_ID_FILE}" ]]; then
    echo "ERROR: repo list does not exist: ${REPO_ID_FILE}" >&2
    exit 1
fi
if [[ "${ACTION_MODE}" != "abs" && "${ACTION_MODE}" != "delta" ]]; then
    echo "ERROR: ACTION_MODE must be abs or delta, got: ${ACTION_MODE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

declare -A OUTPUT_OWNERS=()
dataset_count=0
line_number=0

while IFS= read -r dataset_path || [[ -n "${dataset_path}" ]]; do
    line_number=$((line_number + 1))
    dataset_path="${dataset_path%$'\r'}"
    dataset_path="${dataset_path#"${dataset_path%%[![:space:]]*}"}"
    dataset_path="${dataset_path%"${dataset_path##*[![:space:]]}"}"
    [[ -z "${dataset_path}" ]] && continue

    if [[ "${dataset_path}" != /* ]]; then
        echo "ERROR: line ${line_number} is not an absolute path: ${dataset_path}" >&2
        exit 1
    fi
    if [[ ! -d "${dataset_path}" ]]; then
        echo "ERROR: dataset directory does not exist (line ${line_number}): ${dataset_path}" >&2
        exit 1
    fi
    if [[ ! -f "${dataset_path}/meta/info.json" ]]; then
        echo "ERROR: not a LeRobot v3 dataset; missing meta/info.json: ${dataset_path}" >&2
        exit 1
    fi

    dataset_name="$(basename -- "${dataset_path%/}")"
    output_file="${OUTPUT_PATH%/}/${dataset_name}.json"
    if [[ -n "${OUTPUT_OWNERS[${output_file}]+x}" ]]; then
        echo "ERROR: duplicate output name '${dataset_name}.json' for:" >&2
        echo "  ${OUTPUT_OWNERS[${output_file}]}" >&2
        echo "  ${dataset_path}" >&2
        exit 1
    fi
    OUTPUT_OWNERS["${output_file}"]="${dataset_path}"

    dataset_count=$((dataset_count + 1))
    echo "========== dataset ${dataset_count}: ${dataset_path} =========="
    echo "output: ${output_file}"
    "${PYTHON_BIN}" tools/compute_norm_stats_multi.py \
        --repo_ids "${dataset_path}" \
        --action_mode "${ACTION_MODE}" \
        --chunk_size "${CHUNK_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --output_path "${output_file}"
done < "${REPO_ID_FILE}"

if (( dataset_count == 0 )); then
    echo "ERROR: repo list contains no dataset paths: ${REPO_ID_FILE}" >&2
    exit 1
fi

echo "========== completed =========="
echo "datasets: ${dataset_count}"
echo "output directory: ${OUTPUT_PATH}"
