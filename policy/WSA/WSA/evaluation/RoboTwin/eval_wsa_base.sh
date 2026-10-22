#!/usr/bin/env bash

set -euo pipefail

###############################################################################
################################# ENV config ##################################

# export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# WANDB_TOKEN="${WANDB_TOKEN:-}"
# CONDA_ROOT="${_CONDA_ROOT:-${CONDA_ROOT:-}}"
# CONDA_ENV="${CONDA_ENV:-internvla_a1}"

# if [[ -n "${CONDA_ROOT}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
#     source "${CONDA_ROOT}/etc/profile.d/conda.sh"
#     conda activate "${CONDA_ENV}"
# fi

###############################################################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-4545}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export LD_LIBRARY_PATH="${CUDA_HOME:+${CUDA_HOME}/lib64:}${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd "${PROJ_ROOT}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-zaleni/WSA-Base-RoboTwin}"

POLICY_TYPE="${POLICY_TYPE:-}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-nvidia/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DISABLE_DA3_TEACHER_FOR_EVAL="${DISABLE_DA3_TEACHER_FOR_EVAL:-true}"

BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output_wsa_base_50}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
START_TASK_IDX="${START_TASK_IDX:-0}"
TASK_COUNT="${TASK_COUNT:-50}"
MAX_TASKS=50

GPU_IDS="${GPU_IDS:-0,1}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-35}"

RESIZE_SIZE="${RESIZE_SIZE:-224}"
ACTION_MODE="${ACTION_MODE:-delta}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-false}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-42}"
STATS_KEY="${STATS_KEY:-aloha}"
DTYPE="${DTYPE:-bfloat16}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
INFER_HORIZON="${INFER_HORIZON:-16}"
ACTION_HORIZON_SIZE="${ACTION_HORIZON_SIZE:-50}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
DECODE_IMAGE_FLAG="${DECODE_IMAGE_FLAG:-false}"
ROBOTWIN_EVAL_CONFIG="${ROBOTWIN_EVAL_CONFIG-${PROJ_ROOT}/configs/robotwin_eval_config.yaml}"

export BINARIZE_GRIPPER

if (( $# > 1 )); then
    echo "Usage:"
    echo "  bash evaluation/RoboTwin/${SCRIPT_NAME} [ckpt_dir_or_hf_repo_id]"
    echo "  PRETRAINED_CKPT=zaleni/WSA-Base-RoboTwin bash evaluation/RoboTwin/${SCRIPT_NAME}"
    exit 1
fi

if (( $# == 1 )); then
    PRETRAINED_CKPT="$1"
fi

if [[ -z "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT is empty."
    echo "Usage:"
    echo "  PRETRAINED_CKPT=/path/to/pretrained_model bash evaluation/RoboTwin/${SCRIPT_NAME}"
    echo "  PRETRAINED_CKPT=zaleni/WSA-Base-RoboTwin bash evaluation/RoboTwin/${SCRIPT_NAME}"
    echo "  bash evaluation/RoboTwin/${SCRIPT_NAME} /path/to/pretrained_model"
    exit 1
fi

CKPT_TAG="wsa_base-delta"
DEFAULT_RUN_NAME="${CKPT_TAG}-robotwin-$(date +%Y_%m_%d_%H_%M_%S)"
RUN_NAME="${DEFAULT_RUN_NAME}"
# RUN_NAME="wsa_base-3d-delta-multidata_pretrained300k-finetune-200k-s42h32-robotwin-2026_04_16_09_56_34"
RUN_OUTPUT_PATH="${BASE_OUTPUT_PATH}/${RUN_NAME}"

if [[ -e "${PRETRAINED_CKPT}" && ! -d "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT exists but is not a directory: ${PRETRAINED_CKPT}"
    exit 1
fi

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

if [[ -n "${ROBOTWIN_EVAL_CONFIG}" && "${ROBOTWIN_EVAL_CONFIG}" != /* ]]; then
    ROBOTWIN_EVAL_CONFIG="${PROJ_ROOT}/${ROBOTWIN_EVAL_CONFIG}"
fi

if [[ -n "${ROBOTWIN_EVAL_CONFIG}" && ! -f "${ROBOTWIN_EVAL_CONFIG}" ]]; then
    echo "ROBOTWIN_EVAL_CONFIG is set but the file was not found: ${ROBOTWIN_EVAL_CONFIG}"
    echo "Set ROBOTWIN_EVAL_CONFIG= to disable per-task eval settings."
    exit 1
fi

declare -a TASK_NAMES_BY_IDX=()
declare -a TASK_INFER_HORIZONS=()
declare -a TASK_BINARIZE_GRIPPERS=()

load_per_task_eval_config() {
    local settings_output=""

    settings_output="$(python "${SCRIPT_DIR}/eval_config.py" \
        --project-root "${PROJ_ROOT}" \
        --eval-config "${ROBOTWIN_EVAL_CONFIG}" \
        --default-infer-horizon "${INFER_HORIZON}" \
        --default-binarize-gripper "${BINARIZE_GRIPPER}")"

    while IFS=$'\t' read -r task_idx task_name infer_horizon binarize_gripper; do
        [[ -z "${task_idx}" ]] && continue
        TASK_NAMES_BY_IDX[task_idx]="${task_name}"
        TASK_INFER_HORIZONS[task_idx]="${infer_horizon}"
        TASK_BINARIZE_GRIPPERS[task_idx]="${binarize_gripper}"
    done <<< "${settings_output}"
}

load_per_task_eval_config
if (( ${#TASK_NAMES_BY_IDX[@]} < MAX_TASKS )); then
    echo "Resolved only ${#TASK_NAMES_BY_IDX[@]} RoboTwin task settings, expected ${MAX_TASKS}."
    exit 1
fi

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
    if [[ -d "${PRETRAINED_CKPT}" ]]; then
        echo "checkpoint_source: local_dir"
    else
        echo "checkpoint_source: hf_repo_id"
    fi
    echo "run_output_path: ${RUN_OUTPUT_PATH}"
    echo "task_config: ${TASK_CONFIG}"
    echo "robotwin_eval_config: ${ROBOTWIN_EVAL_CONFIG:-<disabled>}"
    echo "task_range: ${START_TASK_IDX}-${TASK_END_IDX}"
    echo "task_count: ${TASK_COUNT}"
    echo "gpu_ids: ${GPU_ID_ARRAY[*]}"
    echo "max_jobs_per_gpu: ${MAX_JOBS_PER_GPU}"
    echo "total_parallel_jobs: ${TOTAL_SLOTS}"
    echo "action_mode: ${ACTION_MODE}"
    echo "default_binarize_gripper: ${BINARIZE_GRIPPER}"
    echo "test_num: ${TEST_NUM}"
    echo "seed: ${SEED}"
    echo "resize_size: ${RESIZE_SIZE}"
    echo "stats_key: ${STATS_KEY}"
    echo "dtype: ${DTYPE}"
    echo "instruction_type: ${INSTRUCTION_TYPE}"
    echo "default_infer_horizon: ${INFER_HORIZON}"
    echo "policy_type: ${POLICY_TYPE:-auto_from_checkpoint}"
    echo "qwen3_vl_pretrained_path: ${QWEN3_VL_PRETRAINED_PATH:-<checkpoint_config>}"
    echo "qwen3_vl_processor_path: ${QWEN3_VL_PROCESSOR_PATH:-<checkpoint_config>}"
    echo "cosmos_tokenizer_path_or_name: ${COSMOS_TOKENIZER_PATH_OR_NAME:-<checkpoint_config>}"
    echo "da3_model_path_or_name: ${DA3_MODEL_PATH_OR_NAME:-<checkpoint_config>}"
    echo "da3_code_root: ${DA3_CODE_ROOT:-<checkpoint_config>}"
    echo "disable_da3_teacher_for_eval: ${DISABLE_DA3_TEACHER_FOR_EVAL}"
    echo "poll_interval_seconds: ${POLL_INTERVAL_SECONDS}"
} > "${RUN_OUTPUT_PATH}/launch_info.txt"

{
    printf 'PRETRAINED_CKPT=%q ' "${PRETRAINED_CKPT}"
    printf 'BASE_OUTPUT_PATH=%q ' "${BASE_OUTPUT_PATH}"
    printf 'RUN_NAME=%q ' "${RUN_NAME}"
    printf 'TASK_CONFIG=%q ' "${TASK_CONFIG}"
    printf 'START_TASK_IDX=%q ' "${START_TASK_IDX}"
    printf 'TASK_COUNT=%q ' "${TASK_COUNT}"
    printf 'GPU_IDS=%q ' "$(IFS=,; echo "${GPU_ID_ARRAY[*]}")"
    printf 'MAX_JOBS_PER_GPU=%q ' "${MAX_JOBS_PER_GPU}"
    printf 'ROBOTWIN_EVAL_CONFIG=%q ' "${ROBOTWIN_EVAL_CONFIG}"
    printf 'ACTION_MODE=%q ' "${ACTION_MODE}"
    printf 'BINARIZE_GRIPPER=%q ' "${BINARIZE_GRIPPER}"
    printf 'TEST_NUM=%q ' "${TEST_NUM}"
    printf 'SEED=%q ' "${SEED}"
    printf 'RESIZE_SIZE=%q ' "${RESIZE_SIZE}"
    printf 'POLICY_TYPE=%q ' "${POLICY_TYPE}"
    printf 'QWEN3_VL_PRETRAINED_PATH=%q ' "${QWEN3_VL_PRETRAINED_PATH}"
    printf 'QWEN3_VL_PROCESSOR_PATH=%q ' "${QWEN3_VL_PROCESSOR_PATH}"
    printf 'COSMOS_TOKENIZER_PATH_OR_NAME=%q ' "${COSMOS_TOKENIZER_PATH_OR_NAME}"
    printf 'DA3_MODEL_PATH_OR_NAME=%q ' "${DA3_MODEL_PATH_OR_NAME}"
    printf 'DA3_CODE_ROOT=%q ' "${DA3_CODE_ROOT}"
    printf 'DISABLE_DA3_TEACHER_FOR_EVAL=%q ' "${DISABLE_DA3_TEACHER_FOR_EVAL}"
    printf 'bash %q\n' "${SCRIPT_DIR}/${SCRIPT_NAME}"
} > "${RUN_OUTPUT_PATH}/launch_command.txt"

printf "task_idx\tgpu_id\texit_code\toutput_dir\n" > "${RUN_OUTPUT_PATH}/job_status.tsv"

cleanup() {
    for pid in "${SLOT_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup INT TERM

write_task_command_file() {
    local gpu_id="$1"
    local task_idx="$2"
    local task_output_dir="$3"
    local task_infer_horizon="${TASK_INFER_HORIZONS[task_idx]:-${INFER_HORIZON}}"
    local task_binarize_gripper="${TASK_BINARIZE_GRIPPERS[task_idx]:-${BINARIZE_GRIPPER}}"

    local -a cmd=(
        python ../../evaluation/RoboTwin/inference.py
        --args.ckpt_path "${PRETRAINED_CKPT}"
        --args.video_dir "${task_output_dir}"
        --args.task_config "${TASK_CONFIG}"
        --args.task_idx "${task_idx}"
        --args.resize_size "${RESIZE_SIZE}"
        --args.action_mode "${ACTION_MODE}"
        --args.test_num "${TEST_NUM}"
        --args.seed "${SEED}"
        --args.stats_key "${STATS_KEY}"
        --args.dtype "${DTYPE}"
        --args.image_history_interval "${IMAGE_HISTORY_INTERVAL}"
        --args.infer_horizon "${task_infer_horizon}"
        --args.action_horizon_size "${ACTION_HORIZON_SIZE}"
        --args.instruction_type "${INSTRUCTION_TYPE}"
        --args.log_level "${LOG_LEVEL}"
    )

    if [[ "${DECODE_IMAGE_FLAG}" == "true" ]]; then
        cmd+=(--args.decode_image_flag)
    fi

    if [[ -n "${POLICY_TYPE}" ]]; then
        cmd+=(--args.policy_type "${POLICY_TYPE}")
    fi

    if [[ -n "${QWEN3_VL_PRETRAINED_PATH}" ]]; then
        cmd+=(--args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}")
    fi

    if [[ -n "${QWEN3_VL_PROCESSOR_PATH}" ]]; then
        cmd+=(--args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}")
    fi

    if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME}" ]]; then
        cmd+=(--args.cosmos_tokenizer_path_or_name "${COSMOS_TOKENIZER_PATH_OR_NAME}")
    fi

    if [[ -n "${DA3_MODEL_PATH_OR_NAME}" ]]; then
        cmd+=(--args.da3_model_path_or_name "${DA3_MODEL_PATH_OR_NAME}")
    fi

    if [[ -n "${DA3_CODE_ROOT}" ]]; then
        cmd+=(--args.da3_code_root "${DA3_CODE_ROOT}")
    fi

    if [[ "${DISABLE_DA3_TEACHER_FOR_EVAL}" == "true" ]]; then
        cmd+=(--args.disable_3d_teacher_for_eval)
    fi

    {
        printf 'BINARIZE_GRIPPER=%q ' "${task_binarize_gripper}"
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "${task_output_dir}/command.txt"
}

launch_task() {
    local slot_idx="$1"
    local task_idx="$2"
    local gpu_id="${SLOT_GPU_IDS[slot_idx]}"
    local task_output_dir="${RUN_OUTPUT_PATH}/tasks/task_$(printf '%02d' "${task_idx}")"
    local task_log_path="${task_output_dir}/run.log"
    local task_name="${TASK_NAMES_BY_IDX[task_idx]:-task_${task_idx}}"
    local task_infer_horizon="${TASK_INFER_HORIZONS[task_idx]:-${INFER_HORIZON}}"
    local task_binarize_gripper="${TASK_BINARIZE_GRIPPERS[task_idx]:-${BINARIZE_GRIPPER}}"

    mkdir -p "${task_output_dir}"
    write_task_command_file "${gpu_id}" "${task_idx}" "${task_output_dir}"

    (
        set +e
        cd "${PROJ_ROOT}/third_party/RoboTwin"

        CMD=(
            python ../../evaluation/RoboTwin/inference.py
            --args.ckpt_path "${PRETRAINED_CKPT}"
            --args.video_dir "${task_output_dir}"
            --args.task_config "${TASK_CONFIG}"
            --args.task_idx "${task_idx}"
            --args.resize_size "${RESIZE_SIZE}"
            --args.action_mode "${ACTION_MODE}"
            --args.test_num "${TEST_NUM}"
            --args.seed "${SEED}"
            --args.stats_key "${STATS_KEY}"
            --args.dtype "${DTYPE}"
            --args.image_history_interval "${IMAGE_HISTORY_INTERVAL}"
            --args.infer_horizon "${task_infer_horizon}"
            --args.action_horizon_size "${ACTION_HORIZON_SIZE}"
            --args.instruction_type "${INSTRUCTION_TYPE}"
            --args.log_level "${LOG_LEVEL}"
        )

        if [[ "${DECODE_IMAGE_FLAG}" == "true" ]]; then
            CMD+=(--args.decode_image_flag)
        fi

        if [[ -n "${POLICY_TYPE}" ]]; then
            CMD+=(--args.policy_type "${POLICY_TYPE}")
        fi

        if [[ -n "${QWEN3_VL_PRETRAINED_PATH}" ]]; then
            CMD+=(--args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}")
        fi

        if [[ -n "${QWEN3_VL_PROCESSOR_PATH}" ]]; then
            CMD+=(--args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}")
        fi

        if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME}" ]]; then
            CMD+=(--args.cosmos_tokenizer_path_or_name "${COSMOS_TOKENIZER_PATH_OR_NAME}")
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

        BINARIZE_GRIPPER="${task_binarize_gripper}" CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}" > "${task_log_path}" 2>&1
        exit_code=$?
        printf "%s\n" "${exit_code}" > "${task_output_dir}/exit_code.txt"
        exit "${exit_code}"
    ) &

    local pid=$!
    SLOT_PIDS[slot_idx]="${pid}"
    SLOT_TASKS[slot_idx]="${task_idx}"
    SLOT_OUTPUT_DIRS[slot_idx]="${task_output_dir}"

    echo "[launch] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} task=${task_name} pid=${pid}"
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
    local summary_path="${task_output_dir}/summary.json"

    if [[ ! -f "${summary_path}" ]]; then
        return 1
    fi

    return 0
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
strict_mode = sys.argv[5].strip().lower()
expected_test_num = int(sys.argv[6])

expected_task_indices = list(range(start_task_idx, start_task_idx + task_count))


def load_task_names(project_root: Path) -> list[str]:
    inference_path = project_root / "evaluation" / "RoboTwin" / "inference.py"
    module = ast.parse(inference_path.read_text(encoding="utf-8"), filename=str(inference_path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TASK_NAMES":
                task_names = ast.literal_eval(node.value)
                if not isinstance(task_names, list):
                    raise TypeError("TASK_NAMES is not a list")
                return [str(item) for item in task_names]
    raise RuntimeError(f"Failed to find TASK_NAMES in {inference_path}")


def normalize_summary(item: dict, task_idx: int, task_name: str) -> dict:
    item = dict(item)
    item["task_idx"] = int(item.get("task_idx", task_idx))
    item["task_name"] = str(item.get("task_name", task_name))
    item["success_count"] = int(item.get("success_count", 0))
    item["test_num"] = int(item.get("test_num", 0))
    item["success_rate"] = round(
        (item["success_count"] / item["test_num"]) * 100, 2
    ) if item["test_num"] else float(item.get("success_rate", 0.0))
    return item


def parse_summary_from_run_log(task_output_dir: Path, task_idx: int, task_name: str) -> dict | None:
    run_log_path = task_output_dir / "run.log"
    if not run_log_path.exists():
        return None

    text = run_log_path.read_text(encoding="utf-8", errors="ignore")
    clean_text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    matches = re.findall(
        r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9]+(?:\.[0-9]+)?)%",
        clean_text,
    )
    if not matches:
        return None

    success_count_str, test_num_str, _ = matches[-1]
    test_num = int(test_num_str)
    if test_num != expected_test_num:
        return None

    success_count = int(success_count_str)
    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "success_count": success_count,
        "test_num": test_num,
        "success_rate": round((success_count / test_num) * 100, 2) if test_num else 0.0,
        "source": "run_log_fallback",
    }


task_names = load_task_names(proj_root)
expected_tasks = [
    {
        "task_idx": task_idx,
        "task_name": task_names[task_idx],
    }
    for task_idx in expected_task_indices
]
expected_tasks.sort(key=lambda item: (item["task_name"], item["task_idx"]))

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

task_summaries = []
summary_by_task_idx = {}
nonzero_exit_tasks = []

for task_idx in expected_task_indices:
    task_output_dir = run_output_path / "tasks" / f"task_{task_idx:02d}"
    summary_path = task_output_dir / "summary.json"
    task_name = task_names[task_idx]

    if summary_path.exists():
        item = normalize_summary(
            json.loads(summary_path.read_text(encoding="utf-8")),
            task_idx,
            task_name,
        )
        task_summaries.append(item)
        summary_by_task_idx[task_idx] = item
        continue

    fallback_item = parse_summary_from_run_log(task_output_dir, task_idx, task_name)
    if fallback_item is not None:
        summary_path.write_text(
            json.dumps(fallback_item, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        task_summaries.append(fallback_item)
        summary_by_task_idx[task_idx] = fallback_item
        continue

    status = job_status.get(task_idx)
    if status and status["exit_code"] != 0:
        nonzero_exit_tasks.append(task_idx)

finished_task_ids = set(job_status.keys())
missing_summary_tasks = sorted(
    task_idx
    for task_idx in finished_task_ids
    if task_idx not in summary_by_task_idx and task_idx not in nonzero_exit_tasks
)
pending_tasks = sorted(
    task_idx
    for task_idx in expected_task_indices
    if task_idx not in finished_task_ids and task_idx not in summary_by_task_idx
)

task_summaries.sort(key=lambda item: (item["task_name"], item["task_idx"]))

ordered_task_rows = []
for task in expected_tasks:
    task_idx = task["task_idx"]
    summary = summary_by_task_idx.get(task_idx)
    status = job_status.get(task_idx)

    row = {
        "task_idx": task_idx,
        "task_name": task["task_name"],
        "status": "pending",
    }

    if summary is not None:
        row.update(
            {
                "status": "completed",
                "success_count": summary["success_count"],
                "test_num": summary["test_num"],
                "success_rate": summary["success_rate"],
            }
        )
    elif status and status["exit_code"] != 0:
        row.update(
            {
                "status": "failed",
                "exit_code": status["exit_code"],
            }
        )
    elif task_idx in missing_summary_tasks:
        row["status"] = "missing_summary"

    ordered_task_rows.append(row)

completed_tasks = len(task_summaries)
total_success = sum(item["success_count"] for item in task_summaries)
total_tests = sum(item["test_num"] for item in task_summaries)
avg_task_success_rate = round(
    sum(item["success_rate"] for item in task_summaries) / completed_tasks, 2
) if completed_tasks else 0.0
overall_episode_success_rate = round((total_success / total_tests) * 100, 2) if total_tests else 0.0

aggregate_summary = {
    "run_output_path": str(run_output_path),
    "completed_tasks": completed_tasks,
    "expected_tasks": task_count,
    "avg_task_success_rate": avg_task_success_rate,
    "overall_episode_success_rate": overall_episode_success_rate,
    "total_success": total_success,
    "total_tests": total_tests,
    "pending_tasks": pending_tasks,
    "missing_summary_tasks": missing_summary_tasks,
    "nonzero_exit_tasks": nonzero_exit_tasks,
    "tasks": task_summaries,
    "ordered_tasks": ordered_task_rows,
}

(run_output_path / "summary.json").write_text(
    json.dumps(aggregate_summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

lines = [
    f"run_output_path: {run_output_path}",
    f"completed_tasks: {completed_tasks}/{task_count}",
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

for item in ordered_task_rows:
    if item["status"] == "completed":
        lines.append(
            f"{item['task_idx']:02d} {item['task_name']}: "
            f"{item['success_rate']:.2f}% ({item['success_count']}/{item['test_num']})"
        )
    elif item["status"] == "failed":
        lines.append(
            f"{item['task_idx']:02d} {item['task_name']}: FAILED (exit_code={item['exit_code']})"
        )
    elif item["status"] == "missing_summary":
        lines.append(f"{item['task_idx']:02d} {item['task_name']}: MISSING_SUMMARY")
    else:
        lines.append(f"{item['task_idx']:02d} {item['task_name']}: PENDING")

(run_output_path / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

strict = strict_mode == "strict"
raise SystemExit(1 if strict and (pending_tasks or missing_summary_tasks or nonzero_exit_tasks) else 0)
PY
}

refresh_run_summary "progress"

echo "Launching RoboTwin randomized evaluation:"
echo "  tasks         : ${START_TASK_IDX}-${TASK_END_IDX}"
echo "  output        : ${RUN_OUTPUT_PATH}"
echo "  eval config   : ${ROBOTWIN_EVAL_CONFIG:-disabled}"
echo "  gpus          : ${GPU_ID_ARRAY[*]}"
echo "  jobs per gpu  : ${MAX_JOBS_PER_GPU}"
echo "  parallel jobs : ${TOTAL_SLOTS}"

for ((task_idx = START_TASK_IDX; task_idx <= TASK_END_IDX; task_idx++)); do
    if task_is_completed "${task_idx}"; then
        echo "[skip] task_idx=${task_idx} already completed at ${RUN_OUTPUT_PATH}/tasks/task_$(printf '%02d' "${task_idx}")"
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
