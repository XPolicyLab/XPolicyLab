# ============================================================
# Environment Configuration
# ============================================================

starVLA_PYTHON=~/miniconda3/envs/LDA_opensource/bin/python
ROBOCASA_PYTHON=~/miniconda3/envs/robocasa/bin/python
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ============================================================
# Default Arguments
# ============================================================
CKPT_DEFAULT="~/checkpoints/lda/post-train/robocasa/checkpoints"
N_ENVS_DEFAULT=10
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=12

# ============================================================
# GPU / Port Configuration
# ============================================================
# 这里只填你想用的物理 GPU 编号
GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}
BASE_PORT=6398

# ============================================================
# Parse command-line arguments
# ============================================================
CKPT_INPUT=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}
MIN_CKPT_STEP_GAP=5000

echo "=== Evaluation Configuration ==="
echo "Checkpoint Input     : ${CKPT_INPUT}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "Min CKPT Step Gap    : ${MIN_CKPT_STEP_GAP}"
echo "GPU List (physical)  : ${GPU_LIST[*]}"
echo "Num GPUs             : ${NUM_GPUS}"
echo "Base Port            : ${BASE_PORT}"
echo "================================"

# ============================================================
# Environment List
# ============================================================
ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

# ============================================================
# Utility: build checkpoint evaluation order
# ============================================================
CKPT_QUEUE=()
CKPT_QUEUE_STEPS=()

step_is_far_from_pass() {
    local STEP=$1
    local MIN_GAP=$2
    shift 2

    local SELECTED_STEP
    local DIFF
    for SELECTED_STEP in "$@"; do
        if (( STEP > SELECTED_STEP )); then
            DIFF=$((STEP - SELECTED_STEP))
        else
            DIFF=$((SELECTED_STEP - STEP))
        fi

        if (( DIFF < MIN_GAP )); then
            return 1
        fi
    done

    return 0
}

order_sorted_ckpts_with_gap() {
    local MIN_GAP=$1
    shift

    local CURRENT_STEPS=()
    local CURRENT_PATHS=()
    local LINE
    local STEP
    local CKPT_PATH

    for LINE in "$@"; do
        IFS=$'\t' read -r STEP CKPT_PATH <<< "${LINE}"
        CURRENT_STEPS+=("${STEP}")
        CURRENT_PATHS+=("${CKPT_PATH}")
    done

    CKPT_QUEUE=()
    CKPT_QUEUE_STEPS=()

    while (( ${#CURRENT_PATHS[@]} > 0 )); do
        local PASS_STEPS=()
        local DEFERRED_STEPS=()
        local DEFERRED_PATHS=()
        local IDX

        for IDX in "${!CURRENT_PATHS[@]}"; do
            STEP=${CURRENT_STEPS[$IDX]}
            CKPT_PATH=${CURRENT_PATHS[$IDX]}

            if step_is_far_from_pass "${STEP}" "${MIN_GAP}" "${PASS_STEPS[@]}"; then
                CKPT_QUEUE+=("${CKPT_PATH}")
                CKPT_QUEUE_STEPS+=("${STEP}")
                PASS_STEPS+=("${STEP}")
            else
                DEFERRED_STEPS+=("${STEP}")
                DEFERRED_PATHS+=("${CKPT_PATH}")
            fi
        done

        CURRENT_STEPS=("${DEFERRED_STEPS[@]}")
        CURRENT_PATHS=("${DEFERRED_PATHS[@]}")
    done
}

build_ckpt_queue() {
    local CKPT_INPUT=$1

    if [[ -f "${CKPT_INPUT}" ]]; then
        CKPT_QUEUE=("${CKPT_INPUT}")
        if [[ $(basename "${CKPT_INPUT}") =~ ^steps_([0-9]+).*\.pt$ ]]; then
            CKPT_QUEUE_STEPS=("$((10#${BASH_REMATCH[1]}))")
        else
            CKPT_QUEUE_STEPS=("unknown")
        fi
        return 0
    fi

    if [[ ! -d "${CKPT_INPUT}" ]]; then
        echo "Checkpoint input is neither a file nor a directory: ${CKPT_INPUT}" >&2
        return 1
    fi

    local ENTRIES=()
    local CKPT_PATH
    local CKPT_NAME
    local STEP

    while IFS= read -r -d '' CKPT_PATH; do
        CKPT_NAME=$(basename "${CKPT_PATH}")
        if [[ "${CKPT_NAME}" =~ ^steps_([0-9]+).*\.pt$ ]]; then
            STEP=$((10#${BASH_REMATCH[1]}))
            ENTRIES+=("${STEP}"$'\t'"${CKPT_PATH}")
        fi
    done < <(find "${CKPT_INPUT}" -maxdepth 1 -type f -name 'steps_*.pt' -print0)

    if (( ${#ENTRIES[@]} == 0 )); then
        echo "No checkpoints matching steps_*.pt found in: ${CKPT_INPUT}" >&2
        return 1
    fi

    local SORTED_ENTRIES=()
    while IFS= read -r LINE; do
        SORTED_ENTRIES+=("${LINE}")
    done < <(printf "%s\n" "${ENTRIES[@]}" | sort -t $'\t' -k1,1nr -k2,2)

    order_sorted_ckpts_with_gap "${MIN_CKPT_STEP_GAP}" "${SORTED_ENTRIES[@]}"
}

if ! build_ckpt_queue "${CKPT_INPUT}"; then
    exit 1
fi

echo "=== Checkpoint Evaluation Queue ==="
for IDX in "${!CKPT_QUEUE[@]}"; do
    printf "%3d. step=%s | %s\n" "$((IDX + 1))" "${CKPT_QUEUE_STEPS[$IDX]}" "${CKPT_QUEUE[$IDX]}"
done
echo "===================================="

# ============================================================
# Utility: print GPU info inside Python process
# ============================================================
GPU_DEBUG_SNIPPET='import os, torch; \
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES")); \
print("torch.cuda.is_available =", torch.cuda.is_available()); \
print("torch.cuda.device_count =", torch.cuda.device_count()); \
print("torch version =", torch.__version__); \
print("current pid =", os.getpid()); \
print("hostname =", os.uname().nodename if hasattr(os, "uname") else "unknown"); \
print("-----")'

# ============================================================
# Evaluation Function
# ============================================================
EvalEnv() {
    local GPU_ID=$1          # physical GPU id, e.g. 2 / 3
    local PORT=$2            # logical port, e.g. 6398 / 6399
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9

    local SAVE_ROOT
    SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local CKPT_NAME
    CKPT_NAME=$(basename "$CKPT_PATH" .pt)
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos/${CKPT_NAME}/8_denoise_steps_n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_$(echo "${ENV_NAME}" | tr '/' '_')"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | physical GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${ROBOCASA_PYTHON}" -c "${GPU_DEBUG_SNIPPET}" \
        > "${LOG_DIR}/gpu_debug_eval_$(echo "${ENV_NAME}" | tr '/' '_')_gpu${GPU_ID}.log" 2>&1

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${ROBOCASA_PYTHON}" ~/code/LDA_opensource/examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        > "${LOG_DIR}/eval_env_$(echo "${ENV_NAME}" | tr '/' '_')_gpu${GPU_ID}.log" 2>&1
}

# ============================================================
# Utility: summarize success rates from evaluation logs
# ============================================================
summarize_success_rates() {
    local expected_count="${1:-${#ENV_NAMES[@]}}"
    local count=0
    local total="0"
    local missing_count=0

    echo "=== Success Rate Summary ==="
    for TASK_IDX in "${!ENV_NAMES[@]}"; do
        local ENV_NAME="${ENV_NAMES[$TASK_IDX]}"
        local IDX=$((TASK_IDX % NUM_GPUS))
        local GPU_ID=${GPU_LIST[$IDX]}
        local LOG_NAME
        LOG_NAME=$(echo "${ENV_NAME}" | tr '/' '_')
        local EVAL_LOG="${LOG_DIR}/eval_env_${LOG_NAME}_gpu${GPU_ID}.log"

        if [[ ! -e "${EVAL_LOG}" ]]; then
            echo "Missing evaluation log | env ${ENV_NAME} | log ${EVAL_LOG}" >&2
            missing_count=$((missing_count + 1))
            continue
        fi

        local RATE
        RATE="$(sed -nE 's/.*Success rate:[[:space:]]*([-+]?[0-9]+([.][0-9]+)?).*/\1/p' "${EVAL_LOG}" | tail -n 1)"
        if [[ -z "${RATE}" ]]; then
            echo "Missing success rate | env ${ENV_NAME} | log ${EVAL_LOG}" >&2
            missing_count=$((missing_count + 1))
            continue
        fi

        printf "%s: %.2f\n" "${ENV_NAME}" "${RATE}"
        total="$(awk -v total="${total}" -v rate="${RATE}" 'BEGIN { printf "%.6f", total + rate }')"
        count=$((count + 1))
    done

    if (( count == 0 )); then
        echo "No success rates found in ${LOG_DIR}" >&2
        echo "============================"
        return 1
    fi

    awk -v total="${total}" -v count="${count}" 'BEGIN { printf "Average task success rate: %.2f (%d tasks)\n", total / count, count }'
    if (( count != expected_count )); then
        echo "Incomplete success summary: parsed ${count}/${expected_count} task(s), missing ${missing_count}." >&2
        echo "============================"
        return 1
    fi

    echo "============================"
}

# ============================================================
# Cleanup Function
# ============================================================
SERVER_PIDS=()

cleanup() {
    if (( ${#SERVER_PIDS[@]} == 0 )); then
        return 0
    fi

    echo ""
    echo "Shutting down policy servers..."
    for PID in "${SERVER_PIDS[@]:-}"; do
        if [[ -n "${PID}" ]]; then
            kill "${PID}" 2>/dev/null && echo "Killed server PID ${PID}"
        fi
    done
    SERVER_PIDS=()
}
trap cleanup EXIT INT TERM

# ============================================================
# Run one checkpoint
# ============================================================
RunSingleCkpt() {
    local CKPT_PATH=$1
    local LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
    local RUN_STATUS=0

    mkdir -p "${LOG_DIR}"

    echo "=== Launching Multi-GPU Evaluation ==="
    echo "Checkpoint Path  : ${CKPT_PATH}"
    echo "GPUs (physical)  : ${GPU_LIST[*]}"
    echo "Num Environments : ${#ENV_NAMES[@]}"
    echo "Log Directory    : ${LOG_DIR}"

    SERVER_PIDS=()

    # Step 1: Launch Policy Servers
    for IDX in $(seq 0 $((NUM_GPUS - 1))); do
        local GPU_ID=${GPU_LIST[$IDX]}          # physical GPU id
        local PORT=$((BASE_PORT + IDX))         # logical slot -> unique port

        echo "Starting policy server | logical slot ${IDX} | physical GPU ${GPU_ID} | Port ${PORT}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${starVLA_PYTHON}" -c "${GPU_DEBUG_SNIPPET}" \
            > "${LOG_DIR}/gpu_debug_server_slot${IDX}_gpu${GPU_ID}_port${PORT}.log" 2>&1

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${starVLA_PYTHON}" ~/code/LDA_opensource/deployment/model_server/server_policy.py \
            --ckpt_path "${CKPT_PATH}" \
            --port "${PORT}" \
            --use_bf16 \
            > "${LOG_DIR}/server_slot${IDX}_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

        SERVER_PIDS[$IDX]=$!
        sleep 10
    done

    sleep 30

    # Step 2: Dispatch Environments to GPUs
    local COUNT=0
    local ENV_NAME
    for ENV_NAME in "${ENV_NAMES[@]}"; do
        local IDX=$((COUNT % NUM_GPUS))         # logical slot
        local GPU_ID=${GPU_LIST[$IDX]}          # physical GPU id
        local PORT=$((BASE_PORT + IDX))         # bind to corresponding server

        if (( (COUNT + 1) % NUM_GPUS == 0 )); then
            EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                    "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}"
        else
            EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                    "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" &
        fi

        COUNT=$((COUNT + 1))
        sleep 2
    done

    # Step 3: Wait for Evaluations
    while pgrep -f "examples/Robocasa_tabletop/eval_files/simulation_env.py" > /dev/null; do
        echo "Waiting for all evaluation environments to finish..."
        sleep 30
    done

    if ! summarize_success_rates "${#ENV_NAMES[@]}"; then
        echo "Success rate summary is incomplete. Check logs in ${LOG_DIR}."
        RUN_STATUS=1
    fi

    cleanup

    echo "=== Evaluation Finished | Checkpoint: ${CKPT_PATH} ==="
    return "${RUN_STATUS}"
}

# ============================================================
# Run all checkpoints
# ============================================================
FAILED=0
for IDX in "${!CKPT_QUEUE[@]}"; do
    echo ""
    echo "=== Checkpoint $((IDX + 1))/${#CKPT_QUEUE[@]} | step=${CKPT_QUEUE_STEPS[$IDX]} ==="
    if ! RunSingleCkpt "${CKPT_QUEUE[$IDX]}"; then
        FAILED=1
        echo "Checkpoint evaluation failed or summary incomplete: ${CKPT_QUEUE[$IDX]}" >&2
    fi
done

if (( FAILED != 0 )); then
    echo "One or more checkpoint evaluations finished with incomplete summaries." >&2
    exit 1
fi

echo "=== All Evaluations Finished ==="
