#!/usr/bin/env bash

set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-4360}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export LD_LIBRARY_PATH="${CUDA_HOME:+${CUDA_HOME}/lib64:}${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-zaleni/WSA-Large-RoboTwin}"
if (( $# > 1 )); then
    echo "Usage:"
    echo "  PRETRAINED_CKPT=/path/to/checkpoints/200000/pretrained_model bash evaluation/RoboTwin/${SCRIPT_NAME}"
    echo "  bash evaluation/RoboTwin/${SCRIPT_NAME} /path/to/checkpoints/200000/pretrained_model"
    exit 1
fi
if (( $# == 1 )); then
    PRETRAINED_CKPT="$1"
fi
if [[ -z "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT is empty."
    echo "Pass the WSA_Large pretrained_model dir, or the checkpoint step dir containing pretrained_model."
    exit 1
fi
if [[ -e "${PRETRAINED_CKPT}" && ! -d "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT exists but is not a directory: ${PRETRAINED_CKPT}"
    exit 1
fi

POLICY_TYPE="${POLICY_TYPE:-WSA_Large}"
WSA_LARGE_LOAD_TEXT_ENCODER="${WSA_LARGE_LOAD_TEXT_ENCODER:-true}"
WSA_LARGE_REDIRECT_COMMON_FILES="${WSA_LARGE_REDIRECT_COMMON_FILES:-true}"
WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN="${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN:-true}"
export WSA_LARGE_LOAD_TEXT_ENCODER WSA_LARGE_REDIRECT_COMMON_FILES WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN

# DiffSynth/Wan component loader defaults to ./checkpoints relative to the
# current working directory. Eval launches from third_party/RoboTwin, so point it
# back to this repo's checkpoints dir and avoid accidental ModelScope downloads.
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJ_ROOT}/checkpoints}"
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD

WAN_MODEL_ID="${WAN_MODEL_ID:-}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-}"
FUTURE_3D_PRETRAINED_PATH="${FUTURE_3D_PRETRAINED_PATH:-}"
WSA_LARGE_STATS_PATH="${WSA_LARGE_STATS_PATH:-}"
export WAN_MODEL_ID WAN_TOKENIZER_MODEL_ID ACTION_DIT_PRETRAINED_PATH FUTURE_3D_PRETRAINED_PATH WSA_LARGE_STATS_PATH

DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DISABLE_DA3_TEACHER_FOR_EVAL="${DISABLE_DA3_TEACHER_FOR_EVAL:-true}"

BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output_wsa_large_6b_delta}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
START_TASK_IDX="${START_TASK_IDX:-0}"
TASK_COUNT="${TASK_COUNT:-50}"
MAX_TASKS=50

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-35}"

ACTION_MODE="${ACTION_MODE:-delta}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-false}"
SKIP_GET_OBS_WITHIN_REPLAN="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-0}"
STATS_KEY="${STATS_KEY:-aloha}"
DTYPE="${DTYPE:-bfloat16}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
INFER_HORIZON="${INFER_HORIZON:-16}"
ACTION_HORIZON_SIZE="${ACTION_HORIZON_SIZE:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
DECODE_IMAGE_FLAG="${DECODE_IMAGE_FLAG:-false}"

VIDEO_HEIGHT="${VIDEO_HEIGHT:-384}"
VIDEO_WIDTH="${VIDEO_WIDTH:-320}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-robotwin}"
STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
export STANDARDIZE_VIDEO_SIZE_BY_CAMERAS
export BINARIZE_GRIPPER SKIP_GET_OBS_WITHIN_REPLAN

CKPT_TAG="${CKPT_TAG:-6B-pretrained-120k-${ACTION_MODE}-s${SEED}h${INFER_HORIZON}}"
RUN_NAME="${RUN_NAME:-${CKPT_TAG}-$(date +%Y_%m_%d_%H_%M_%S)}"
RUN_OUTPUT_PATH="${BASE_OUTPUT_PATH}/${RUN_NAME}"

if (( START_TASK_IDX < 0 )); then
    echo "START_TASK_IDX must be >= 0, got ${START_TASK_IDX}"
    exit 1
fi
if (( TASK_COUNT <= 0 )); then
    echo "TASK_COUNT must be > 0, got ${TASK_COUNT}"
    exit 1
fi
if (( START_TASK_IDX + TASK_COUNT > MAX_TASKS )); then
    echo "Requested task range exceeds RoboTwin randomized task count (${MAX_TASKS})."
    echo "Got START_TASK_IDX=${START_TASK_IDX}, TASK_COUNT=${TASK_COUNT}"
    exit 1
fi

parse_gpu_ids() {
    local source_string=""
    if [[ -n "${GPU_IDS}" ]]; then
        source_string="${GPU_IDS// /}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        source_string="${CUDA_VISIBLE_DEVICES// /}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_ID_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader)
        return
    else
        GPU_ID_ARRAY=("0")
        return
    fi
    IFS=',' read -r -a GPU_ID_ARRAY <<< "${source_string}"
}

parse_gpu_ids
if (( ${#GPU_ID_ARRAY[@]} == 0 )); then
    echo "No GPU ids resolved. Set GPU_IDS explicitly, for example GPU_IDS=0,1."
    exit 1
fi
if (( MAX_JOBS_PER_GPU <= 0 )); then
    echo "MAX_JOBS_PER_GPU must be > 0, got ${MAX_JOBS_PER_GPU}"
    exit 1
fi

mkdir -p "${RUN_OUTPUT_PATH}/tasks"
TASK_END_IDX=$((START_TASK_IDX + TASK_COUNT - 1))

declare -a SLOT_GPU_IDS=()
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
    for ((slot_repeat = 0; slot_repeat < MAX_JOBS_PER_GPU; slot_repeat++)); do
        SLOT_GPU_IDS+=("${gpu_id}")
    done
done
TOTAL_SLOTS=${#SLOT_GPU_IDS[@]}

declare -a SLOT_PIDS
declare -a SLOT_TASKS
declare -a SLOT_OUTPUT_DIRS
declare -a FAILED_TASKS=()
for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
    SLOT_PIDS[slot_idx]=""
    SLOT_TASKS[slot_idx]=""
    SLOT_OUTPUT_DIRS[slot_idx]=""
done

{
    echo "script: ${SCRIPT_DIR}/${SCRIPT_NAME}"
    echo "pretrained_ckpt: ${PRETRAINED_CKPT}"
    echo "run_output_path: ${RUN_OUTPUT_PATH}"
    echo "task_config: ${TASK_CONFIG}"
    echo "task_range: ${START_TASK_IDX}-${TASK_END_IDX}"
    echo "task_count: ${TASK_COUNT}"
    echo "gpu_ids: ${GPU_ID_ARRAY[*]}"
    echo "max_jobs_per_gpu: ${MAX_JOBS_PER_GPU}"
    echo "total_parallel_jobs: ${TOTAL_SLOTS}"
    echo "policy_type: ${POLICY_TYPE}"
    echo "action_mode: ${ACTION_MODE}"
    echo "binarize_gripper: ${BINARIZE_GRIPPER}"
    echo "skip_get_obs_within_replan: ${SKIP_GET_OBS_WITHIN_REPLAN}"
    echo "test_num: ${TEST_NUM}"
    echo "seed: ${SEED}"
    echo "dtype: ${DTYPE}"
    echo "infer_horizon: ${INFER_HORIZON}"
    echo "num_inference_steps: ${NUM_INFERENCE_STEPS}"
    echo "wsa_large_load_text_encoder: ${WSA_LARGE_LOAD_TEXT_ENCODER}"
    echo "wsa_large_redirect_common_files: ${WSA_LARGE_REDIRECT_COMMON_FILES}"
    echo "wsa_large_skip_dit_load_from_pretrain: ${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN}"
    echo "diffsynth_model_base_path: ${DIFFSYNTH_MODEL_BASE_PATH}"
    echo "diffsynth_skip_download: ${DIFFSYNTH_SKIP_DOWNLOAD}"
    echo "wan_model_id: ${WAN_MODEL_ID:-<checkpoint_config>}"
    echo "wan_tokenizer_model_id: ${WAN_TOKENIZER_MODEL_ID:-<checkpoint_config>}"
    echo "action_dit_pretrained_path: ${ACTION_DIT_PRETRAINED_PATH:-<skipped>}"
    echo "future_3d_pretrained_path: ${FUTURE_3D_PRETRAINED_PATH:-<skipped>}"
    echo "da3_model_path_or_name: ${DA3_MODEL_PATH_OR_NAME:-<checkpoint_config>}"
    echo "da3_code_root: ${DA3_CODE_ROOT:-<checkpoint_config>}"
    echo "disable_da3_teacher_for_eval: ${DISABLE_DA3_TEACHER_FOR_EVAL}"
    echo "video_size: [${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    echo "concat_multi_camera: ${CONCAT_MULTI_CAMERA}"
} > "${RUN_OUTPUT_PATH}/launch_info.txt"

printf "task_idx\tgpu_id\texit_code\toutput_dir\n" > "${RUN_OUTPUT_PATH}/job_status.tsv"

append_optional_wsa_large_args() {
    if [[ -n "${WAN_MODEL_ID}" ]]; then
        CMD+=(--args.wsa_large_model_id "${WAN_MODEL_ID}")
    fi
    if [[ -n "${WAN_TOKENIZER_MODEL_ID}" ]]; then
        CMD+=(--args.wsa_large_tokenizer_model_id "${WAN_TOKENIZER_MODEL_ID}")
    fi
    if [[ -n "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
        CMD+=(--args.wsa_large_action_dit_pretrained_path "${ACTION_DIT_PRETRAINED_PATH}")
    fi
    if [[ -n "${FUTURE_3D_PRETRAINED_PATH}" ]]; then
        CMD+=(--args.wsa_large_future_3d_pretrained_path "${FUTURE_3D_PRETRAINED_PATH}")
    fi
    if [[ -n "${WSA_LARGE_STATS_PATH}" ]]; then
        CMD+=(--args.wsa_large_stats_path "${WSA_LARGE_STATS_PATH}")
    fi
    if [[ -n "${DA3_MODEL_PATH_OR_NAME}" ]]; then
        CMD+=(--args.da3_model_path_or_name "${DA3_MODEL_PATH_OR_NAME}")
    fi
    if [[ -n "${DA3_CODE_ROOT}" ]]; then
        CMD+=(--args.da3_code_root "${DA3_CODE_ROOT}")
    fi
    if [[ "${DISABLE_DA3_TEACHER_FOR_EVAL}" == "true" ]]; then
        CMD+=(--args.disable_3d_teacher_for_eval)
    fi
    if [[ "${DECODE_IMAGE_FLAG}" == "true" ]]; then
        CMD+=(--args.decode_image_flag)
    fi
}

build_cmd() {
    local task_idx="$1"
    local task_output_dir="$2"
    CMD=(
        python ../../evaluation/RoboTwin/inference.py
        --args.ckpt_path "${PRETRAINED_CKPT}"
        --args.video_dir "${task_output_dir}"
        --args.task_config "${TASK_CONFIG}"
        --args.task_idx "${task_idx}"
        --args.action_mode "${ACTION_MODE}"
        --args.test_num "${TEST_NUM}"
        --args.seed "${SEED}"
        --args.stats_key "${STATS_KEY}"
        --args.dtype "${DTYPE}"
        --args.image_history_interval "${IMAGE_HISTORY_INTERVAL}"
        --args.infer_horizon "${INFER_HORIZON}"
        --args.action_horizon_size "${ACTION_HORIZON_SIZE}"
        --args.num_inference_steps "${NUM_INFERENCE_STEPS}"
        --args.instruction_type "${INSTRUCTION_TYPE}"
        --args.log_level "${LOG_LEVEL}"
        --args.policy_type "${POLICY_TYPE}"
        --args.wsa_large_video_height "${VIDEO_HEIGHT}"
        --args.wsa_large_video_width "${VIDEO_WIDTH}"
        --args.wsa_large_concat_multi_camera "${CONCAT_MULTI_CAMERA}"
    )
    append_optional_wsa_large_args
}

write_task_command_file() {
    local gpu_id="$1"
    local task_idx="$2"
    local task_output_dir="$3"
    build_cmd "${task_idx}" "${task_output_dir}"
    {
        printf 'DIFFSYNTH_MODEL_BASE_PATH=%q ' "${DIFFSYNTH_MODEL_BASE_PATH}"
        printf 'DIFFSYNTH_SKIP_DOWNLOAD=%q ' "${DIFFSYNTH_SKIP_DOWNLOAD}"
        printf 'WSA_LARGE_LOAD_TEXT_ENCODER=%q ' "${WSA_LARGE_LOAD_TEXT_ENCODER}"
        printf 'WSA_LARGE_REDIRECT_COMMON_FILES=%q ' "${WSA_LARGE_REDIRECT_COMMON_FILES}"
        printf 'WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN=%q ' "${WSA_LARGE_SKIP_DIT_LOAD_FROM_PRETRAIN}"
        printf 'BINARIZE_GRIPPER=%q ' "${BINARIZE_GRIPPER}"
        printf 'SKIP_GET_OBS_WITHIN_REPLAN=%q ' "${SKIP_GET_OBS_WITHIN_REPLAN}"
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    } > "${task_output_dir}/command.txt"
}

cleanup() {
    for pid in "${SLOT_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

launch_task() {
    local slot_idx="$1"
    local task_idx="$2"
    local gpu_id="${SLOT_GPU_IDS[slot_idx]}"
    local task_output_dir="${RUN_OUTPUT_PATH}/tasks/task_$(printf '%02d' "${task_idx}")"
    local task_log_path="${task_output_dir}/run.log"

    mkdir -p "${task_output_dir}"
    write_task_command_file "${gpu_id}" "${task_idx}" "${task_output_dir}"

    (
        set +e
        cd "${PROJ_ROOT}/third_party/RoboTwin"
        build_cmd "${task_idx}" "${task_output_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}" > "${task_log_path}" 2>&1
        exit_code=$?
        printf "%s\n" "${exit_code}" > "${task_output_dir}/exit_code.txt"
        exit "${exit_code}"
    ) &

    local pid=$!
    SLOT_PIDS[slot_idx]="${pid}"
    SLOT_TASKS[slot_idx]="${task_idx}"
    SLOT_OUTPUT_DIRS[slot_idx]="${task_output_dir}"
    echo "[launch] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} pid=${pid}"
}

refresh_run_summary() {
    local strict_mode="${1:-progress}"
    python - "${PROJ_ROOT}" "${RUN_OUTPUT_PATH}" "${START_TASK_IDX}" "${TASK_COUNT}" "${strict_mode}" "${TEST_NUM}" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path

proj_root = Path(sys.argv[1])
run_output_path = Path(sys.argv[2])
start_task_idx = int(sys.argv[3])
task_count = int(sys.argv[4])
strict = sys.argv[5].strip().lower() == "strict"
expected_test_num = int(sys.argv[6])
expected_task_indices = list(range(start_task_idx, start_task_idx + task_count))

module = ast.parse((proj_root / "evaluation" / "RoboTwin" / "inference.py").read_text(encoding="utf-8"))
task_names = None
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TASK_NAMES":
                task_names = [str(item) for item in ast.literal_eval(node.value)]
                break
if task_names is None:
    raise RuntimeError("Failed to read TASK_NAMES from inference.py")

job_status = {}
job_status_path = run_output_path / "job_status.tsv"
if job_status_path.exists():
    for line in job_status_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("task_idx\t"):
            continue
        task_idx, gpu_id, exit_code, output_dir = line.split("\t", 3)
        job_status[int(task_idx)] = {
            "gpu_id": gpu_id,
            "exit_code": int(exit_code),
            "output_dir": output_dir,
        }

def normalize(item, task_idx, task_name):
    item = dict(item)
    item["task_idx"] = int(item.get("task_idx", task_idx))
    item["task_name"] = str(item.get("task_name", task_name))
    item["success_count"] = int(item.get("success_count", 0))
    item["test_num"] = int(item.get("test_num", 0))
    item["success_rate"] = round(item["success_count"] / item["test_num"] * 100, 2) if item["test_num"] else 0.0
    return item

def parse_log(task_output_dir, task_idx, task_name):
    run_log = task_output_dir / "run.log"
    if not run_log.exists():
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", run_log.read_text(encoding="utf-8", errors="ignore"))
    matches = re.findall(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not matches:
        return None
    success_count, test_num, _ = matches[-1]
    if int(test_num) != expected_test_num:
        return None
    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "success_count": int(success_count),
        "test_num": int(test_num),
        "success_rate": round(int(success_count) / int(test_num) * 100, 2),
        "source": "run_log_fallback",
    }

summaries = []
summary_by_task = {}
nonzero_exit_tasks = []
for task_idx in expected_task_indices:
    task_output_dir = run_output_path / "tasks" / f"task_{task_idx:02d}"
    summary_path = task_output_dir / "summary.json"
    task_name = task_names[task_idx]
    item = None
    if summary_path.exists():
        item = normalize(json.loads(summary_path.read_text(encoding="utf-8")), task_idx, task_name)
    else:
        item = parse_log(task_output_dir, task_idx, task_name)
        if item is not None:
            summary_path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if item is not None:
        summaries.append(item)
        summary_by_task[task_idx] = item
        continue
    status = job_status.get(task_idx)
    if status and status["exit_code"] != 0:
        nonzero_exit_tasks.append(task_idx)

finished_task_ids = set(job_status)
pending_tasks = sorted(task_idx for task_idx in expected_task_indices if task_idx not in finished_task_ids and task_idx not in summary_by_task)
missing_summary_tasks = sorted(task_idx for task_idx in finished_task_ids if task_idx not in summary_by_task and task_idx not in nonzero_exit_tasks)

rows = []
for task_idx in expected_task_indices:
    task_name = task_names[task_idx]
    if task_idx in summary_by_task:
        row = {"status": "completed", **summary_by_task[task_idx]}
    elif task_idx in nonzero_exit_tasks:
        row = {"task_idx": task_idx, "task_name": task_name, "status": "failed", "exit_code": job_status[task_idx]["exit_code"]}
    elif task_idx in missing_summary_tasks:
        row = {"task_idx": task_idx, "task_name": task_name, "status": "missing_summary"}
    else:
        row = {"task_idx": task_idx, "task_name": task_name, "status": "pending"}
    rows.append(row)

completed = len(summaries)
total_success = sum(item["success_count"] for item in summaries)
total_tests = sum(item["test_num"] for item in summaries)
avg_task_success_rate = round(sum(item["success_rate"] for item in summaries) / completed, 2) if completed else 0.0
overall_episode_success_rate = round(total_success / total_tests * 100, 2) if total_tests else 0.0

aggregate = {
    "run_output_path": str(run_output_path),
    "completed_tasks": completed,
    "expected_tasks": task_count,
    "avg_task_success_rate": avg_task_success_rate,
    "overall_episode_success_rate": overall_episode_success_rate,
    "total_success": total_success,
    "total_tests": total_tests,
    "pending_tasks": pending_tasks,
    "missing_summary_tasks": missing_summary_tasks,
    "nonzero_exit_tasks": nonzero_exit_tasks,
    "tasks": sorted(summaries, key=lambda item: item["task_idx"]),
    "ordered_tasks": rows,
}
run_output_path.joinpath("summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    f"run_output_path: {run_output_path}",
    f"completed_tasks: {completed}/{task_count}",
    f"avg_task_success_rate: {avg_task_success_rate:.2f}%",
    f"overall_episode_success_rate: {overall_episode_success_rate:.2f}%",
    f"total_success: {total_success}",
    f"total_tests: {total_tests}",
    f"pending_tasks: {pending_tasks if pending_tasks else 'none'}",
    f"missing_summary_tasks: {missing_summary_tasks if missing_summary_tasks else 'none'}",
    f"nonzero_exit_tasks: {nonzero_exit_tasks if nonzero_exit_tasks else 'none'}",
    "",
    "per_task:",
]
for row in rows:
    if row["status"] == "completed":
        lines.append(f"{row['task_idx']:02d} {row['task_name']}: {row['success_rate']:.2f}% ({row['success_count']}/{row['test_num']})")
    elif row["status"] == "failed":
        lines.append(f"{row['task_idx']:02d} {row['task_name']}: FAILED (exit_code={row['exit_code']})")
    else:
        lines.append(f"{row['task_idx']:02d} {row['task_name']}: {row['status'].upper()}")
run_output_path.joinpath("summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

raise SystemExit(1 if strict and (pending_tasks or missing_summary_tasks or nonzero_exit_tasks) else 0)
PY
}

reap_finished_slots() {
    local updated=0
    for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
        local pid="${SLOT_PIDS[slot_idx]}"
        if [[ -z "${pid}" ]]; then
            continue
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            local exit_code=0
            if wait "${pid}"; then
                exit_code=0
            else
                exit_code=$?
            fi
            local task_idx="${SLOT_TASKS[slot_idx]}"
            local gpu_id="${SLOT_GPU_IDS[slot_idx]}"
            local task_output_dir="${SLOT_OUTPUT_DIRS[slot_idx]}"
            local summary_path="${task_output_dir}/summary.json"
            if [[ "${exit_code}" -ne 0 && -f "${summary_path}" ]]; then
                echo "[warn] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} wrote summary despite exit_code=${exit_code}; treating as completed" >&2
                exit_code=0
                printf "%s\n" "${exit_code}" > "${task_output_dir}/exit_code.txt"
            fi
            printf "%s\t%s\t%s\t%s\n" "${task_idx}" "${gpu_id}" "${exit_code}" "${task_output_dir}" >> "${RUN_OUTPUT_PATH}/job_status.tsv"
            if [[ "${exit_code}" -ne 0 ]]; then
                FAILED_TASKS+=("${task_idx}")
                echo "[error] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} exit_code=${exit_code}" >&2
                echo "        log: ${task_output_dir}/run.log" >&2
            else
                echo "[done] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx}" >&2
            fi
            SLOT_PIDS[slot_idx]=""
            SLOT_TASKS[slot_idx]=""
            SLOT_OUTPUT_DIRS[slot_idx]=""
            updated=1
        fi
    done
    if (( updated == 1 )); then
        refresh_run_summary "progress"
    fi
}

find_free_slot() {
    while true; do
        reap_finished_slots || true
        for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
            if [[ -z "${SLOT_PIDS[slot_idx]}" ]]; then
                echo "${slot_idx}"
                return
            fi
        done
        sleep "${POLL_INTERVAL_SECONDS}"
    done
}

wait_for_all_slots() {
    while true; do
        local has_running=0
        reap_finished_slots || true
        for pid in "${SLOT_PIDS[@]}"; do
            if [[ -n "${pid}" ]]; then
                has_running=1
                break
            fi
        done
        if [[ "${has_running}" -eq 0 ]]; then
            return
        fi
        sleep "${POLL_INTERVAL_SECONDS}"
    done
}

task_is_completed() {
    local task_idx="$1"
    local task_output_dir="${RUN_OUTPUT_PATH}/tasks/task_$(printf '%02d' "${task_idx}")"
    [[ -f "${task_output_dir}/summary.json" ]]
}

refresh_run_summary "progress"

echo "Launching WSA_Large RoboTwin randomized evaluation:"
echo "  tasks         : ${START_TASK_IDX}-${TASK_END_IDX}"
echo "  output        : ${RUN_OUTPUT_PATH}"
echo "  gpus          : ${GPU_ID_ARRAY[*]}"
echo "  jobs per gpu  : ${MAX_JOBS_PER_GPU}"
echo "  parallel jobs : ${TOTAL_SLOTS}"
echo "  action mode   : ${ACTION_MODE}"

for ((task_idx = START_TASK_IDX; task_idx <= TASK_END_IDX; task_idx++)); do
    if task_is_completed "${task_idx}"; then
        echo "[skip] task_idx=${task_idx} already completed"
        continue
    fi
    free_slot="$(find_free_slot)"
    launch_task "${free_slot}" "${task_idx}"
done

wait_for_all_slots

aggregate_exit_code=0
refresh_run_summary "strict" || aggregate_exit_code=$?
if (( aggregate_exit_code != 0 )); then
    echo "Aggregation detected incomplete or failed tasks."
    echo "See ${RUN_OUTPUT_PATH}/summary.txt and ${RUN_OUTPUT_PATH}/job_status.tsv for details."
    exit 1
fi
if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "Finished with failed task launches: ${FAILED_TASKS[*]}"
    echo "See ${RUN_OUTPUT_PATH}/job_status.tsv and per-task run.log for details."
    exit 1
fi

echo "Finished all ${TASK_COUNT} randomized tasks."
echo "Summary:"
echo "  ${RUN_OUTPUT_PATH}/summary.txt"
echo "  ${RUN_OUTPUT_PATH}/summary.json"
