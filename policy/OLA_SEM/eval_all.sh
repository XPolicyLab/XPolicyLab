#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash eval_all.sh <checkpoint_root> <policy_conda_env> <robotwin_conda_env> [episodes] [gpu_ids] [seed] [conditions]

Runs all 50 RoboTwin tasks on local GPUs. Each GPU gets an independent
OLA-SEM policy server and a share of the task-condition evaluations.
Defaults: episodes=100, gpu_ids=all, seed=42, conditions=clean,randomized.

Required environment variables:
  OLA_SEM_WAN_PATH       Path to Wan2.2-TI2V-5B
  OLA_SEM_VLM_PATH       Path to Qwen3-VL-2B-Instruct
  OLA_SEM_ROBOTWIN_ROOT  RoboTwin checkout (defaults to beside XPolicyLab)

Optional environment variables:
  OLA_SEM_TASKS_FILE     Task list (default: eval_tasks_50.txt)
  OLA_SEM_EVAL_RUN_ROOT  Parent output directory (default: eval_runs)
  OLA_SEM_EVAL_VIDEO_LOG Enable videos (default: false)
  OLA_SEM_GPU_IDS        GPU list used when gpu_ids=all (for example: 0,1,3)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -lt 3 ]]; then
    usage >&2
    exit 2
fi
checkpoint_root=$1
policy_conda_env=$2
robotwin_conda_env=$3
episodes=${4:-100}
gpu_ids_arg=${5:-all}
seed=${6:-42}
conditions_csv=${7:-clean,randomized}

if ! [[ "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] episodes must be a positive integer; got ${episodes}" >&2
    exit 2
fi
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed must be a non-negative integer; got ${seed}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
tasks_file=${OLA_SEM_TASKS_FILE:-${SCRIPT_DIR}/eval_tasks_50.txt}
run_id=${OLA_SEM_RUN_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}
run_parent=${OLA_SEM_EVAL_RUN_ROOT:-${SCRIPT_DIR}/eval_runs}
run_root=${run_parent}/${run_id}
mkdir -p "${run_root}/elements" "${run_root}/logs" "${run_root}/status"

[[ -f "${checkpoint_root}/config.json" ]] || {
    echo "[ERROR] Missing checkpoint config: ${checkpoint_root}/config.json" >&2
    exit 1
}
[[ -f "${checkpoint_root}/pytorch_model/mp_rank_00_model_states.pt" ]] || {
    echo "[ERROR] Missing checkpoint weights below ${checkpoint_root}/pytorch_model" >&2
    exit 1
}
[[ -f "${tasks_file}" ]] || { echo "[ERROR] Missing task list: ${tasks_file}" >&2; exit 1; }
: "${OLA_SEM_WAN_PATH:?Set OLA_SEM_WAN_PATH}"
: "${OLA_SEM_VLM_PATH:?Set OLA_SEM_VLM_PATH}"

mapfile -t tasks < <(sed '/^[[:space:]]*$/d' "${tasks_file}")
if [[ "${#tasks[@]}" -ne 50 ]]; then
    echo "[ERROR] Expected exactly 50 tasks in ${tasks_file}; got ${#tasks[@]}" >&2
    exit 1
fi
if [[ "$(printf '%s\n' "${tasks[@]}" | sort -u | wc -l)" -ne 50 ]]; then
    echo "[ERROR] Task list contains duplicates: ${tasks_file}" >&2
    exit 1
fi

IFS=',' read -r -a conditions <<< "${conditions_csv}"
if [[ "${#conditions[@]}" -eq 0 ]] || \
   [[ "$(printf '%s\n' "${conditions[@]}" | sort -u | wc -l)" -ne "${#conditions[@]}" ]]; then
    echo "[ERROR] conditions must be a non-empty list without duplicates: ${conditions_csv}" >&2
    exit 2
fi
for condition in "${conditions[@]}"; do
    if [[ "${condition}" != "clean" && "${condition}" != "randomized" ]]; then
        echo "[ERROR] conditions must contain only clean and/or randomized; got ${condition}" >&2
        exit 2
    fi
done

if [[ "${gpu_ids_arg}" == "all" ]]; then
    if [[ -n "${OLA_SEM_GPU_IDS:-}" ]]; then
        gpu_ids_csv=${OLA_SEM_GPU_IDS}
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        gpu_ids_csv=${CUDA_VISIBLE_DEVICES}
    else
        command -v nvidia-smi >/dev/null 2>&1 || {
            echo "[ERROR] Cannot discover GPUs: nvidia-smi is unavailable." >&2
            exit 1
        }
        gpu_ids_csv=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
    fi
else
    gpu_ids_csv=${gpu_ids_arg}
fi

IFS=',' read -r -a gpu_ids <<< "${gpu_ids_csv}"
for index in "${!gpu_ids[@]}"; do
    gpu_ids[index]=$(printf '%s' "${gpu_ids[index]}" | tr -d '[:space:]')
done
if [[ "${#gpu_ids[@]}" -eq 0 ]] || [[ -z "${gpu_ids[0]}" ]] || \
   [[ "$(printf '%s\n' "${gpu_ids[@]}" | sed '/^$/d' | sort -u | wc -l)" -ne "${#gpu_ids[@]}" ]]; then
    echo "[ERROR] No usable unique GPU IDs were found in: ${gpu_ids_csv:-<empty>}" >&2
    exit 1
fi
for gpu_id in "${gpu_ids[@]}"; do
    if [[ -z "${gpu_id}" ]]; then
        echo "[ERROR] Empty GPU ID in: ${gpu_ids_csv}" >&2
        exit 1
    fi
done
gpu_count=${#gpu_ids[@]}

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export EVAL_ENV_TYPE=sim

server_pids=()
worker_pids=()
cleanup() {
    local pid
    for pid in "${worker_pids[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]:-}" "${server_pids[@]:-}"; do
        [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "[OLA_SEM] run_id=${run_id} tasks=50 conditions=${conditions_csv} episodes=${episodes} gpus=${gpu_ids_csv}"
echo "[OLA_SEM] output=${run_root}"

ports=()
for worker_index in "${!gpu_ids[@]}"; do
    while true; do
        port=$(bash "${XPL_ROOT}/utils/get_free_port.sh")
        if ! printf '%s\n' "${ports[@]:-}" | grep -Fxq "${port}"; then
            break
        fi
    done
    ports+=("${port}")
    gpu_id=${gpu_ids[worker_index]}
    log_gpu=${gpu_id//[^[:alnum:]_.-]/_}
    bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
        RoboTwin "${tasks[0]}" "${checkpoint_root}" aloha_agilex joint "${seed}" \
        "${gpu_id}" "${policy_conda_env}" "${port}" 127.0.0.1 \
        >"${run_root}/logs/policy_server_gpu_${log_gpu}.out" \
        2>"${run_root}/logs/policy_server_gpu_${log_gpu}.err" &
    server_pids+=("$!")
done

for worker_index in "${!gpu_ids[@]}"; do
    bash "${XPL_ROOT}/utils/wait_for_policy_server.sh" \
        127.0.0.1 "${ports[worker_index]}" "${server_pids[worker_index]}" \
        "OLA-SEM policy server on GPU ${gpu_ids[worker_index]}" 1200
done

run_gpu_worker() {
    local worker_index=$1
    local gpu_id=$2
    local port=$3
    local server_pid=$4
    local ordinal=0
    local worker_failed=0
    local condition task task_config stem status

    for condition in "${conditions[@]}"; do
        task_config="demo_${condition}"
        for task in "${tasks[@]}"; do
            if (( ordinal % gpu_count != worker_index )); then
                ordinal=$((ordinal + 1))
                continue
            fi
            ordinal=$((ordinal + 1))
            stem="${condition}_${task}"
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                echo "[ERROR] GPU ${gpu_id} policy server exited before ${condition}/${task}" >&2
                printf '125\n' >"${run_root}/status/${stem}.exit_code"
                printf 'policy server exited\n' >"${run_root}/status/${stem}.error"
                worker_failed=1
                continue
            fi

            echo "[OLA_SEM] start gpu=${gpu_id} condition=${condition} task=${task}"
            set +e
            OLA_SEM_ROBOTWIN_TASK_CONFIG="${task_config}" \
            OLA_SEM_TEST_NUM="${episodes}" \
            OLA_SEM_EVAL_VIDEO_LOG="${OLA_SEM_EVAL_VIDEO_LOG:-false}" \
            OLA_SEM_RESULT_PATH_TAG="ola_sem_${run_id}_${condition}" \
            OLA_SEM_ELEMENT_SUMMARY_PATH="${run_root}/elements/${stem}.json" \
            bash "${SCRIPT_DIR}/setup_eval_env_client.sh" \
                RoboTwin "${task}" "${checkpoint_root}" aloha_agilex joint "${seed}" \
                "${gpu_id}" "${robotwin_conda_env}" \
                "local_full_eval=${run_id}" "${port}" 127.0.0.1 \
                >"${run_root}/logs/${stem}.out" \
                2>"${run_root}/logs/${stem}.err"
            status=$?
            set -e
            printf '%s\n' "${status}" >"${run_root}/status/${stem}.exit_code"
            if [[ "${status}" -eq 0 ]]; then
                echo "[OLA_SEM] passed gpu=${gpu_id} condition=${condition} task=${task}"
            else
                worker_failed=1
                echo "[OLA_SEM] failed gpu=${gpu_id} condition=${condition} task=${task} exit=${status}" >&2
            fi
        done
    done
    return "${worker_failed}"
}

for worker_index in "${!gpu_ids[@]}"; do
    run_gpu_worker "${worker_index}" "${gpu_ids[worker_index]}" \
        "${ports[worker_index]}" "${server_pids[worker_index]}" &
    worker_pids+=("$!")
done

worker_status=0
set +e
for pid in "${worker_pids[@]}"; do
    wait "${pid}" || worker_status=1
done
set -e
worker_pids=()

summary_status=0
python3 "${SCRIPT_DIR}/summarize_local_eval.py" \
    --run-dir "${run_root}" \
    --tasks-file "${tasks_file}" \
    --conditions "${conditions_csv}" \
    --expected-episodes "${episodes}" || summary_status=$?

echo "[OLA_SEM] local evaluation finished: worker_status=${worker_status} summary_status=${summary_status}"
echo "[OLA_SEM] summary=${run_root}/summary.md"
[[ "${worker_status}" -eq 0 && "${summary_status}" -eq 0 ]]
