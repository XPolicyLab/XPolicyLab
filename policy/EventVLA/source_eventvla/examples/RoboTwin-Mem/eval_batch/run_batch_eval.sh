#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_MEM_EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_FILES_DIR="$ROBOTWIN_MEM_EXAMPLE_DIR/eval_files"
EVENTVLA_ROOT="$(cd "$ROBOTWIN_MEM_EXAMPLE_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$EVENTVLA_ROOT/.." && pwd)"
DEFAULT_ROBOTWIN_MEM_ROOT="$REPO_ROOT/RoboTwin-Mem"

CONFIG_PATH="${1:-$SCRIPT_DIR/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH"
  echo "Usage: bash $0 <weights_config.sh>"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"

ROBOTWIN_MEM_ROOT="${ROBOTWIN_MEM_ROOT:-$DEFAULT_ROBOTWIN_MEM_ROOT}"

if ! declare -p TASKS >/dev/null 2>&1; then
  TASKS=(
    "cover_blocks_hard"
    "pick_the_unhidden_block"
    "find_seal_and_seal_stamp"
    "pick_objects_in_order"
    "put_back_block_hard"
    "press_button_keyframe"
    "rearrange_blocks_hard"
    "reproduce_route"
  )
fi

normalize_task_name() {
  local task="$1"
  case "$task" in
    observe_and_pickup_hard) printf '%s' "pick_the_unhidden_block" ;;
    observe_and_pickup_object) printf '%s' "pick_objects_in_order" ;;
    reproduct_route) printf '%s' "reproduce_route" ;;
    *) printf '%s' "$task" ;;
  esac
}

for idx in "${!TASKS[@]}"; do
  TASKS[$idx]="$(normalize_task_name "${TASKS[$idx]}")"
done

NUM_TASKS="${#TASKS[@]}"

if ! declare -p GPU_SLOTS >/dev/null 2>&1; then
  GPU_SLOTS=(0 0 1 1 2 2 3 3)
fi

if ! declare -p PORT_SLOTS >/dev/null 2>&1; then
  PORT_SLOTS=(5902 5903 5904 5905 5906 5907 5908 5909)
fi

TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-0}"
HOST="${HOST:-127.0.0.1}"
UNNORM_KEY="${UNNORM_KEY:-robotwin_mem}"
ACTION_MODE="${ACTION_MODE:-abs}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_START_MAX_RETRIES="${SERVER_START_MAX_RETRIES:-2}"
SERVER_RETRY_SLEEP="${SERVER_RETRY_SLEEP:-20}"
SERVER_START_STAGGER_SECONDS="${SERVER_START_STAGGER_SECONDS:-8}"
SERVER_STARTUP_STALL_TIMEOUT="${SERVER_STARTUP_STALL_TIMEOUT:-300}"
EVAL_LOG_SNAPSHOT_INTERVAL="${EVAL_LOG_SNAPSHOT_INTERVAL:-0}"
AUTO_KILL_PORT_OCCUPY="${AUTO_KILL_PORT_OCCUPY:-1}"
POLICY_NAME="${POLICY_NAME:-model2robotwin_mem_interface}"
USE_BF16="${USE_BF16:-1}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_EVENTVLA_ENV_PY="/shared/smartbot/yangganlin/anaconda3/envs/starVLA/bin/python"
DEFAULT_ROBOTWIN_MEM_ENV_PY="/shared/smartbot/yangganlin/anaconda3/envs/RMBench/bin/python"

if [[ -n "${EVENTVLA_PYTHON:-}" ]]; then
  SERVER_PYTHON="$EVENTVLA_PYTHON"
elif [[ -n "${STAR_VLA_PYTHON:-}" ]]; then
  SERVER_PYTHON="$STAR_VLA_PYTHON"
elif [[ -x "$DEFAULT_EVENTVLA_ENV_PY" ]]; then
  SERVER_PYTHON="$DEFAULT_EVENTVLA_ENV_PY"
else
  SERVER_PYTHON="python"
fi

if [[ -n "${ROBOTWIN_MEM_PYTHON:-}" ]]; then
  EVAL_PYTHON_BIN="$ROBOTWIN_MEM_PYTHON"
elif [[ -n "${RMBENCH_PYTHON:-}" ]]; then
  EVAL_PYTHON_BIN="$RMBENCH_PYTHON"
elif [[ -n "${EVAL_PYTHON:-}" ]]; then
  EVAL_PYTHON_BIN="$EVAL_PYTHON"
elif [[ -x "$DEFAULT_ROBOTWIN_MEM_ENV_PY" ]]; then
  EVAL_PYTHON_BIN="$DEFAULT_ROBOTWIN_MEM_ENV_PY"
else
  EVAL_PYTHON_BIN="python"
fi

get_python_bin_dir() {
  local python_path="$1"
  if [[ "$python_path" == */* ]]; then
    dirname "$python_path"
  fi
}

SERVER_PYTHON_BIN_DIR="$(get_python_bin_dir "$SERVER_PYTHON")"
EVAL_PYTHON_BIN_DIR="$(get_python_bin_dir "$EVAL_PYTHON_BIN")"
SOCKET_CHECK_PYTHON="${SOCKET_CHECK_PYTHON:-$EVAL_PYTHON_BIN}"

DEPLOY_POLICY_PATH="${DEPLOY_POLICY_PATH:-$EVAL_FILES_DIR/deploy_policy.yml}"
SERVER_SCRIPT_PATH="$EVENTVLA_ROOT/deployment/model_server/server_policy.py"
EVAL_SCRIPT_PATH="$ROBOTWIN_MEM_ROOT/script/eval_policy.py"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/batch_eval/$RUN_TAG}"
ROUND_ID="${ROUND_ID:-1}"
ROUND_DIR="$LOG_ROOT/round_${ROUND_ID}"
RUNTIME_LOG_ROOT="${RUNTIME_LOG_ROOT:-/tmp/eventvla_robotwin_batch_eval/$RUN_TAG}"
RUNTIME_ROUND_DIR="$RUNTIME_LOG_ROOT/round_${ROUND_ID}"

mkdir -p "$ROUND_DIR" "$RUNTIME_ROUND_DIR"
SUMMARY_CSV="$LOG_ROOT/summary.csv"
printf 'round,task_index,task,ckpt_var,ckpt_path,gpu,port,seed,server_ready,eval_exit,final_exit,success_rate,failed_examples,result_file,server_log,eval_log,status_file\n' > "$SUMMARY_CSV"

cleanup_children() {
  local pid
  while read -r pid; do
    if declare -F kill_pid_tree >/dev/null 2>&1; then
      kill_pid_tree TERM "$pid"
    else
      kill "$pid" 2>/dev/null || true
    fi
  done < <(jobs -pr)
}
trap cleanup_children INT TERM

is_port_open() {
  local host="$1"
  local port="$2"
  "$SOCKET_CHECK_PYTHON" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.4)
ok = sock.connect_ex((host, port)) == 0
sock.close()
raise SystemExit(0 if ok else 1)
PY
}

get_listen_pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"$port" -sTCP:LISTEN -n -P 2>/dev/null | sort -u
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u
    return 0
  fi

  return 0
}

clear_port_listeners() {
  local port="$1"
  local pids=()
  local pid

  while read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(get_listen_pids_on_port "$port")

  if (( ${#pids[@]} == 0 )); then
    return 0
  fi

  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done

  sleep 0.3

  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done

  return 0
}

file_mtime() {
  local file="$1"
  if [[ ! -e "$file" ]]; then
    printf '0'
    return 0
  fi

  stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null || printf '0'
}

copy_runtime_log() {
  local src="$1"
  local dst="$2"
  local dst_dir
  local dst_base
  local tmp

  dst_dir="$(dirname "$dst")"
  dst_base="$(basename "$dst")"
  tmp="$dst_dir/.${dst_base}.tmp.$$"

  if [[ ! -f "$src" ]]; then
    return 0
  fi

  if cp "$src" "$tmp" 2>/dev/null; then
    mv -f "$tmp" "$dst"
    return 0
  fi

  rm -f "$tmp" 2>/dev/null || true
  return 1
}

start_log_snapshotter() {
  local src="$1"
  local dst="$2"
  local interval="$3"

  (
    while true; do
      copy_runtime_log "$src" "$dst" || true
      sleep "$interval" || exit 0
    done
  ) &
  LOG_SNAPSHOTTER_PID="$!"
}

stop_log_snapshotter() {
  local pid="$1"
  local src="$2"
  local dst="$3"

  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  copy_runtime_log "$src" "$dst" || true
}

wait_server_ready() {
  local host="$1"
  local port="$2"
  local pid="$3"
  local timeout="$4"
  local log_file="${5:-}"
  local stall_timeout="${6:-0}"
  local elapsed=0
  local last_log_mtime=0
  local stall_elapsed=0

  if [[ -n "$log_file" ]]; then
    last_log_mtime="$(file_mtime "$log_file")"
  fi

  while (( elapsed < timeout )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 2
    fi
    if is_port_open "$host" "$port"; then
      return 0
    fi
    sleep 1
    ((elapsed++))

    if (( stall_timeout > 0 )) && [[ -n "$log_file" ]]; then
      local current_log_mtime
      current_log_mtime="$(file_mtime "$log_file")"
      if [[ "$current_log_mtime" != "$last_log_mtime" ]]; then
        last_log_mtime="$current_log_mtime"
        stall_elapsed=0
      else
        ((stall_elapsed++))
        if (( stall_elapsed >= stall_timeout )); then
          return 3
        fi
      fi
    fi
  done

  return 1
}

kill_pid_tree() {
  local signal="$1"
  local pid="$2"

  if command -v pgrep >/dev/null 2>&1; then
    local child
    while read -r child; do
      [[ -n "$child" ]] && kill_pid_tree "$signal" "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi

  kill "-$signal" "$pid" 2>/dev/null || true
}

kill_server_pid() {
  local pid="$1"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  kill_pid_tree TERM "$pid"
  for _ in {1..10}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done
  kill_pid_tree KILL "$pid"
}

get_ckpt_var_name() {
  local task_idx_1based="$1"
  printf 'WEIGHT_TASK%s' "$task_idx_1based"
}

get_ckpt_setting_var_name() {
  local task_idx_1based="$1"
  printf 'CKPT_SETTING_TASK%s' "$task_idx_1based"
}

get_keyframe_cluster_window_var_name() {
  local task_idx_1based="$1"
  printf 'KEYFRAME_CLUSTER_TIMESTEP_WINDOW_TASK%s' "$task_idx_1based"
}

infer_ckpt_setting_from_path() {
  local ckpt_path="$1"
  local run_name
  run_name="$(basename "$(dirname "$(dirname "$ckpt_path")")")"

  if [[ "$run_name" =~ qwen3OFT_([^/]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$run_name" =~ qwen2_5OFT_([^/]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$run_name" =~ OFT_([^/]+)$ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi

  printf ''
}

sanitize_ckpt_tag() {
  local value="$1"
  value="$(printf '%s' "$value" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[._-]+//; s/[._-]+$//')"
  printf '%s' "${value:-unknown_ckpt}"
}

resolve_ckpt_setting() {
  local task_idx_1based="$1"
  local ckpt_path="$2"
  local ckpt_setting_var
  ckpt_setting_var="$(get_ckpt_setting_var_name "$task_idx_1based")"
  local ckpt_setting="${!ckpt_setting_var:-}"

  if [[ -z "$ckpt_setting" ]]; then
    ckpt_setting="$(infer_ckpt_setting_from_path "$ckpt_path")"
  fi
  if [[ -z "$ckpt_setting" ]]; then
    ckpt_setting="weight_task${task_idx_1based}"
  fi

  printf '%s' "$ckpt_setting"
}

append_optional_override() {
  local var_name="$1"
  local override_name="$2"
  local value="${!var_name-}"
  if [[ -n "$value" ]]; then
    eval_override_flags+=(--"$override_name" "$value")
  fi
}

validate_common_paths() {
  local missing=0

  if [[ ! -d "$ROBOTWIN_MEM_ROOT" ]]; then
    echo "[ERROR] RoboTwin-Mem root not found: $ROBOTWIN_MEM_ROOT"
    missing=1
  fi
  if [[ ! -f "$DEPLOY_POLICY_PATH" ]]; then
    echo "[ERROR] deploy_policy.yml not found: $DEPLOY_POLICY_PATH"
    missing=1
  fi
  if [[ ! -f "$SERVER_SCRIPT_PATH" ]]; then
    echo "[ERROR] EventVLA server script not found: $SERVER_SCRIPT_PATH"
    missing=1
  fi
  if [[ ! -f "$EVAL_SCRIPT_PATH" ]]; then
    echo "[ERROR] RoboTwin-Mem eval_policy.py not found: $EVAL_SCRIPT_PATH"
    missing=1
  fi

  return "$missing"
}

validate_task_list() {
  local missing=0
  local task

  for task in "${TASKS[@]}"; do
    if [[ ! -f "$ROBOTWIN_MEM_ROOT/envs/${task}.py" ]]; then
      echo "[ERROR] RoboTwin-Mem task env not found for task '$task': $ROBOTWIN_MEM_ROOT/envs/${task}.py"
      missing=1
    fi
  done

  return "$missing"
}

validate_ckpt_list() {
  local missing=0
  local idx ckpt_var ckpt_path

  for ((idx = 1; idx <= NUM_TASKS; idx++)); do
    ckpt_var="$(get_ckpt_var_name "$idx")"
    ckpt_path="${!ckpt_var:-}"
    if [[ -z "$ckpt_path" ]]; then
      echo "[ERROR] Missing variable: $ckpt_var"
      missing=1
    elif [[ ! -f "$ckpt_path" ]]; then
      echo "[ERROR] Checkpoint not found for $ckpt_var: $ckpt_path"
      missing=1
    fi
  done

  return "$missing"
}

validate_slot_layout() {
  if (( ${#GPU_SLOTS[@]} != NUM_TASKS )); then
    echo "[ERROR] GPU_SLOTS size mismatch: expected $NUM_TASKS, got ${#GPU_SLOTS[@]}"
    return 1
  fi

  if (( ${#PORT_SLOTS[@]} != NUM_TASKS )); then
    echo "[ERROR] PORT_SLOTS size mismatch: expected $NUM_TASKS, got ${#PORT_SLOTS[@]}"
    return 1
  fi

  return 0
}

collect_eval_metrics() {
  local task="$1"
  local ckpt_tag="$2"

  local success_rate="NA"
  local failed_examples="NA"
  local result_file="NA"

  local base_dir="$ROBOTWIN_MEM_ROOT/eval_result/$task/$POLICY_NAME/$TASK_CONFIG/$ckpt_tag"

  if [[ -d "$base_dir" ]]; then
    local result_candidates=()
    local f
    while IFS= read -r -d '' f; do
      result_candidates+=("$f")
    done < <(find "$base_dir" -maxdepth 2 -type f -name '_result.txt' -print0 2>/dev/null)

    if (( ${#result_candidates[@]} > 0 )); then
      result_file="$(ls -1t "${result_candidates[@]}" 2>/dev/null | head -n 1)"

      local sr
      sr="$(awk -F': ' '/^Success Rate:/{print $2; exit}' "$result_file" 2>/dev/null || true)"
      [[ -n "$sr" ]] && success_rate="$sr"

      local eval_detail_file
      eval_detail_file="$(dirname "$result_file")/eval_log.txt"
      if [[ -f "$eval_detail_file" ]]; then
        failed_examples="$(awk -F'[=, ]+' '/result=Fail/{printf "%s%s", (n++?"|":""), "ep"$2"(seed"$4")"} END{if(n==0) printf "none"}' "$eval_detail_file" 2>/dev/null || true)"
        [[ -z "$failed_examples" ]] && failed_examples="NA"
      fi
    fi
  fi

  printf '%s\t%s\t%s\n' "$success_rate" "$failed_examples" "$result_file"
}

run_one_job() {
  local slot_idx="$1"
  local row_file="$2"

  local task_idx_1based=$((slot_idx + 1))
  local task="${TASKS[$slot_idx]}"
  local gpu="${GPU_SLOTS[$slot_idx]}"
  local port="${PORT_SLOTS[$slot_idx]}"

  local ckpt_var
  ckpt_var="$(get_ckpt_var_name "$task_idx_1based")"
  local ckpt_path="${!ckpt_var}"
  local ckpt_setting
  ckpt_setting="$(resolve_ckpt_setting "$task_idx_1based" "$ckpt_path")"
  local ckpt_tag
  ckpt_tag="$(sanitize_ckpt_tag "$ckpt_setting")"
  local keyframe_cluster_window_var
  keyframe_cluster_window_var="$(get_keyframe_cluster_window_var_name "$task_idx_1based")"
  local keyframe_cluster_window="${!keyframe_cluster_window_var:-${KEYFRAME_CLUSTER_TIMESTEP_WINDOW:-}}"

  local server_log="$ROUND_DIR/server_${task}_${ckpt_tag}_g${gpu}_p${port}.log"
  local server_runtime_log="$RUNTIME_ROUND_DIR/server_${task}_${ckpt_tag}_g${gpu}_p${port}.log"
  local eval_log="$ROUND_DIR/eval_${task}_${ckpt_tag}_g${gpu}_p${port}.log"
  local eval_runtime_log="$RUNTIME_ROUND_DIR/eval_${task}_${ckpt_tag}_g${gpu}_p${port}.log"
  local status_file="$ROUND_DIR/status_${task}_${ckpt_tag}.txt"

  local server_ready="0"
  local eval_exit="999"
  local final_exit="0"
  local success_rate="NA"
  local failed_examples="NA"
  local result_file="NA"
  local server_pid=""
  local eval_snapshot_pid=""
  local LOG_SNAPSHOTTER_PID=""

  local bf16_flags=()
  if [[ "$USE_BF16" == "1" ]]; then
    bf16_flags=(--use_bf16)
  fi

  local eval_override_flags=(
    --task_name "$task"
    --task_config "$TASK_CONFIG"
    --ckpt_setting "$ckpt_setting"
    --seed "$SEED"
    --policy_name "$POLICY_NAME"
    --instruction_type "$INSTRUCTION_TYPE"
    --host "$HOST"
    --port "$port"
    --policy_ckpt_path "$ckpt_path"
    --unnorm_key "$UNNORM_KEY"
    --action_mode "\"$ACTION_MODE\""
  )

  if [[ -n "${keyframe_cluster_window:-}" ]]; then
    eval_override_flags+=(--keyframe_cluster_timestep_window "$keyframe_cluster_window")
  fi
  append_optional_override "KEYFRAME_COMMIT_CONFIDENCE_THRESHOLD" "keyframe_commit_confidence_threshold"
  append_optional_override "FIRST_CHUNK_RANDOM_REPLAN" "first_chunk_random_replan"
  append_optional_override "FIRST_CHUNK_RANDOM_REPLAN_SEED" "first_chunk_random_replan_seed"
  append_optional_override "FIRST_CHUNK_FIXED_REPLAN_STEP" "first_chunk_fixed_replan_step"
  append_optional_override "SAMPLING_INTERVAL" "sampling_interval"
  append_optional_override "REPLAN_AFTER_KEYFRAME_COMMIT" "replan_after_keyframe_commit"
  if declare -p EVAL_EXTRA_OVERRIDES >/dev/null 2>&1; then
    eval_override_flags+=("${EVAL_EXTRA_OVERRIDES[@]}")
  fi

  echo "[$(date '+%F %T')] round=$ROUND_ID task=$task gpu=$gpu port=$port ckpt=$ckpt_path" > "$status_file"
  echo "[INFO] Runtime server log while running: $server_runtime_log" >> "$status_file"
  echo "[INFO] Runtime eval log while running: $eval_runtime_log" >> "$status_file"
  if [[ -n "${keyframe_cluster_window:-}" ]]; then
    echo "[INFO] keyframe_cluster_timestep_window=$keyframe_cluster_window" >> "$status_file"
  fi

  : > "$server_runtime_log"
  : > "$eval_runtime_log"
  {
    echo "[INFO] Server stdout/stderr are captured on local scratch while running."
    echo "[INFO] Runtime log: $server_runtime_log"
    echo "[INFO] Final log will be copied here after server shutdown."
  } > "$server_log"
  {
    echo "[INFO] Eval stdout/stderr are captured on local scratch while running."
    echo "[INFO] Runtime log: $eval_runtime_log"
    echo "[INFO] Final log will be copied here after eval finishes."
    if [[ "$EVAL_LOG_SNAPSHOT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
      echo "[INFO] Optional readable snapshots are copied here every ${EVAL_LOG_SNAPSHOT_INTERVAL}s while eval runs."
    fi
  } > "$eval_log"

  if is_port_open "$HOST" "$port"; then
    if [[ "$AUTO_KILL_PORT_OCCUPY" == "1" ]]; then
      echo "[WARN] Port already in use, trying to clear listeners: $HOST:$port" >> "$status_file"
      clear_port_listeners "$port"
    fi
  fi

  if is_port_open "$HOST" "$port"; then
    echo "[ERROR] Port already in use before server start: $HOST:$port" >> "$status_file"
    final_exit=31
    eval_exit=31
  else
    if (( SERVER_START_STAGGER_SECONDS > 0 && slot_idx > 0 )); then
      local stagger_sleep=$((slot_idx * SERVER_START_STAGGER_SECONDS))
      echo "[INFO] Staggering server start by ${stagger_sleep}s" >> "$status_file"
      sleep "$stagger_sleep"
    fi

    local attempt=1
    local ready_rc=1
    while (( attempt <= SERVER_START_MAX_RETRIES )); do
      echo "[INFO] Starting EventVLA server attempt $attempt/$SERVER_START_MAX_RETRIES at $(date '+%F %T')" >> "$status_file"
      echo "[INFO] ===== EventVLA server attempt $attempt/$SERVER_START_MAX_RETRIES at $(date '+%F %T') =====" >> "$server_runtime_log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [[ -n "$SERVER_PYTHON_BIN_DIR" ]]; then
          export PATH="$SERVER_PYTHON_BIN_DIR:${PATH:-}"
        fi
        export PYTHONPATH="$EVENTVLA_ROOT:${PYTHONPATH:-}"
        cd "$EVENTVLA_ROOT" || exit 1
        "$SERVER_PYTHON" "$SERVER_SCRIPT_PATH" \
          --ckpt_path "$ckpt_path" \
          --port "$port" \
          "${bf16_flags[@]}"
      ) >> "$server_runtime_log" 2>&1 &
      server_pid=$!

      if wait_server_ready "$HOST" "$port" "$server_pid" "$SERVER_READY_TIMEOUT" "$server_runtime_log" "$SERVER_STARTUP_STALL_TIMEOUT"; then
        server_ready="1"
        ready_rc=0
        echo "[INFO] Server ready on $HOST:$port at $(date '+%F %T')" >> "$status_file"
        break
      fi

      ready_rc=$?
      echo "[WARN] Server attempt $attempt/$SERVER_START_MAX_RETRIES not ready. rc=$ready_rc" >> "$status_file"
      kill_server_pid "$server_pid"
      wait "$server_pid" 2>/dev/null || true
      copy_runtime_log "$server_runtime_log" "$server_log" || true
      server_pid=""

      if (( attempt < SERVER_START_MAX_RETRIES )); then
        echo "[INFO] Sleeping ${SERVER_RETRY_SLEEP}s before retry" >> "$status_file"
        sleep "$SERVER_RETRY_SLEEP"
      fi
      ((attempt++))
    done

    if [[ "$server_ready" == "1" ]]; then
      if [[ "$EVAL_LOG_SNAPSHOT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
        start_log_snapshotter "$eval_runtime_log" "$eval_log" "$EVAL_LOG_SNAPSHOT_INTERVAL"
        eval_snapshot_pid="$LOG_SNAPSHOTTER_PID"
      fi
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [[ -n "$EVAL_PYTHON_BIN_DIR" ]]; then
          export PATH="$EVAL_PYTHON_BIN_DIR:${PATH:-}"
        fi
        export PYTHONPATH="$ROBOTWIN_MEM_ROOT:$EVENTVLA_ROOT:$EVAL_FILES_DIR:${PYTHONPATH:-}"
        cd "$ROBOTWIN_MEM_ROOT" || exit 1

        PYTHONWARNINGS=ignore::UserWarning \
        "$EVAL_PYTHON_BIN" script/eval_policy.py \
          --config "$DEPLOY_POLICY_PATH" \
          --overrides \
          "${eval_override_flags[@]}"
      ) > "$eval_runtime_log" 2>&1
      eval_exit=$?
      if [[ -n "$eval_snapshot_pid" ]]; then
        stop_log_snapshotter "$eval_snapshot_pid" "$eval_runtime_log" "$eval_log"
        eval_snapshot_pid=""
      else
        copy_runtime_log "$eval_runtime_log" "$eval_log" || true
      fi
      final_exit=$eval_exit
    else
      server_ready="0"
      eval_exit=$((40 + ready_rc))
      final_exit=$eval_exit
      echo "[ERROR] Server not ready after $SERVER_START_MAX_RETRIES attempt(s). rc=$ready_rc" >> "$status_file"
    fi

    if [[ -n "$server_pid" ]]; then
      kill_server_pid "$server_pid"
      wait "$server_pid" 2>/dev/null || true
      copy_runtime_log "$server_runtime_log" "$server_log" || true
    fi
  fi

  local metric_line
  metric_line="$(collect_eval_metrics "$task" "$ckpt_setting")"
  IFS=$'\t' read -r success_rate failed_examples result_file <<< "$metric_line"

  {
    echo "server_ready=$server_ready"
    echo "eval_exit=$eval_exit"
    echo "final_exit=$final_exit"
    echo "success_rate=$success_rate"
    echo "failed_examples=$failed_examples"
    echo "result_file=$result_file"
    echo "server_log=$server_log"
    echo "server_runtime_log=$server_runtime_log"
    echo "eval_log=$eval_log"
    echo "eval_runtime_log=$eval_runtime_log"
    echo "status_file=$status_file"
  } >> "$status_file"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$ROUND_ID" "$task_idx_1based" "$task" "$ckpt_var" "$ckpt_path" "$gpu" "$port" "$SEED" \
    "$server_ready" "$eval_exit" "$final_exit" "$success_rate" "$failed_examples" "$result_file" \
    "$server_log" "$eval_log" "$status_file" > "$row_file"

  return "$final_exit"
}

if ! validate_common_paths; then
  exit 1
fi

if ! validate_task_list; then
  exit 1
fi

if ! validate_ckpt_list; then
  echo "[ERROR] Weight list invalid. Fix config: $CONFIG_PATH"
  exit 1
fi

if ! validate_slot_layout; then
  exit 1
fi

echo "[INFO] Batch eval started"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Logs: $LOG_ROOT"
echo "[INFO] EVENTVLA_ROOT: $EVENTVLA_ROOT"
echo "[INFO] ROBOTWIN_MEM_ROOT: $ROBOTWIN_MEM_ROOT"
echo "[INFO] SERVER_PYTHON(EventVLA): $SERVER_PYTHON"
echo "[INFO] EVAL_PYTHON(RoboTwin-Mem): $EVAL_PYTHON_BIN"
echo "[INFO] Tasks: ${TASKS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] Validation passed. Planned jobs:"
  for slot_idx in "${!TASKS[@]}"; do
    task_idx_1based=$((slot_idx + 1))
    ckpt_var="$(get_ckpt_var_name "$task_idx_1based")"
    ckpt_path="${!ckpt_var}"
    ckpt_setting="$(resolve_ckpt_setting "$task_idx_1based" "$ckpt_path")"
    printf '[DRY_RUN] slot=%s task=%s gpu=%s port=%s ckpt_setting=%s ckpt=%s\n' \
      "$slot_idx" "${TASKS[$slot_idx]}" "${GPU_SLOTS[$slot_idx]}" "${PORT_SLOTS[$slot_idx]}" "$ckpt_setting" "$ckpt_path"
  done
  echo "[DRY_RUN] No servers or simulations were started."
  exit 0
fi

pids=()
row_files=()
for slot_idx in "${!TASKS[@]}"; do
  row_file="$ROUND_DIR/row_slot${slot_idx}.csv"
  row_files+=("$row_file")
  run_one_job "$slot_idx" "$row_file" &
  pids+=("$!")
done

overall_fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    overall_fail=1
  fi
done

for row_file in "${row_files[@]}"; do
  [[ -f "$row_file" ]] && cat "$row_file" >> "$SUMMARY_CSV"
done

if (( overall_fail == 1 )); then
  echo "[DONE] Batch eval finished with failures. See: $SUMMARY_CSV"
  exit 1
fi

echo "[DONE] Batch eval finished successfully. Summary: $SUMMARY_CSV"
exit 0
