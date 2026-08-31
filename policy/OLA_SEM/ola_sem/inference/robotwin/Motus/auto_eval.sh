#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$POLICY_DIR/paths_config.yml}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: configuration not found: $CONFIG_FILE" >&2
    echo "Copy paths_config.example.yml to paths_config.yml and edit it." >&2
    exit 1
fi
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

yaml_value() {
    local key="$1"
    awk -v key="$key" '
        $1 == key ":" {
            value = substr($0, index($0, ":") + 1)
            sub(/[[:space:]]+#.*/, "", value)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") {
                value = substr(value, 2, length(value) - 2)
            }
            print value
            exit
        }
    ' "$CONFIG_FILE"
}

TASKS_FILE="${TASKS_FILE:-$(yaml_value tasks_file)}"
TASKS_FILE="${TASKS_FILE:-tasks_all.txt}"
if [[ "$TASKS_FILE" != /* ]]; then
    TASKS_FILE="$POLICY_DIR/$TASKS_FILE"
fi
if [[ ! -f "$TASKS_FILE" ]]; then
    echo "Error: task list not found: $TASKS_FILE" >&2
    exit 1
fi

mapfile -t TASKS < <(sed 's/\r$//' "$TASKS_FILE" | awk 'NF && !seen[$0]++')
if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "Error: no tasks found in $TASKS_FILE" >&2
    exit 1
fi

GPU_IDS_VALUE="${GPU_IDS:-$(yaml_value gpu_ids)}"
GPU_IDS_VALUE="${GPU_IDS_VALUE//[/}"
GPU_IDS_VALUE="${GPU_IDS_VALUE//]/}"
GPU_IDS_VALUE="${GPU_IDS_VALUE//,/ }"
read -r -a GPU_LIST <<< "$GPU_IDS_VALUE"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    else
        GPU_LIST=(0)
    fi
fi

LOG_DIR="${LOG_DIR:-$POLICY_DIR/logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
failures=0

wait_batch() {
    local index
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            echo "Completed: ${BATCH_TASKS[$index]}"
        else
            echo "Failed: ${BATCH_TASKS[$index]} (see $LOG_DIR/${BATCH_TASKS[$index]}.log)" >&2
            failures=$((failures + 1))
        fi
    done
    PIDS=()
    BATCH_TASKS=()
}

PIDS=()
BATCH_TASKS=()
for index in "${!TASKS[@]}"; do
    task="${TASKS[$index]}"
    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    CONFIG_FILE="$CONFIG_FILE" \
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_DIR/${task}.log" \
    GPU_ID="$gpu" \
    bash "$POLICY_DIR/eval.sh" "$task" &
    PIDS+=("$!")
    BATCH_TASKS+=("$task")

    if [[ "${#PIDS[@]}" -eq "${#GPU_LIST[@]}" ]]; then
        wait_batch
    fi
done
if [[ "${#PIDS[@]}" -gt 0 ]]; then
    wait_batch
fi

echo "Evaluation complete: ${#TASKS[@]} tasks, $failures failures. Logs: $LOG_DIR"
if [[ "$failures" -gt 0 ]]; then
    exit 1
fi
