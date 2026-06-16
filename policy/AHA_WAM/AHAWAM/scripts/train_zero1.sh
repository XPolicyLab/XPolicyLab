#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]

Multi-node env vars:
  NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 DEEPSPEED_HOSTFILE=hostfile bash scripts/train_zero1.sh 8 task=<task_name>
  NNODES=2 NODE_RANK=1 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 DEEPSPEED_HOSTFILE=hostfile bash scripts/train_zero1.sh 8 task=<task_name>

For single-node launch, MASTER_ADDR/MASTER_PORT can be omitted. MASTER_PORT defaults to 29500.}"
shift

EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  case "${EXTRA_ARGS[0]}" in
    causal)
      echo "Error: causal training variants are not included in this release. Use task=<task> with model=ahawam for retained chunk-local training." >&2
      exit 1
      ;;
    standard)
      EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
      ;;
  esac
fi
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-${DS_HOSTFILE:-${HOSTFILE:-}}}"

DEFAULT_TASK="robodojo_local_history_updated_kv_prior_only_16"
DEFAULT_MODEL="ahawam"
is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

if (( NUM_MACHINES < 1 )); then
  echo "Error: NNODES/NUM_MACHINES must be >= 1, got ${NUM_MACHINES}." >&2
  exit 1
fi

if (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: NODE_RANK/MACHINE_RANK must be in [0, $((NUM_MACHINES - 1))], got ${MACHINE_RANK}." >&2
  exit 1
fi

if (( NUM_MACHINES > 1 )) && [[ -z "${DEEPSPEED_HOSTFILE}" ]]; then
  echo "Error: multi-node training requires DEEPSPEED_HOSTFILE, DS_HOSTFILE, or HOSTFILE to be set." >&2
  echo "Example hostfile:" >&2
  echo "  192.168.254.114 slots=${NPROC_PER_NODE}" >&2
  echo "  192.168.254.116 slots=${NPROC_PER_NODE}" >&2
  exit 1
fi

NUM_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))
ACCELERATE_ARGS=()

if [[ -n "${DEEPSPEED_HOSTFILE}" ]]; then
  if [[ ! -f "${DEEPSPEED_HOSTFILE}" ]]; then
    echo "Error: DEEPSPEED_HOSTFILE does not exist: ${DEEPSPEED_HOSTFILE}" >&2
    exit 1
  fi
  ACCELERATE_ARGS+=(--deepspeed_hostfile "${DEEPSPEED_HOSTFILE}")
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

# Read the model specified in `- override /model: <model>` from a task yaml file.
# Returns the model name, or empty string if not found.
extract_model_from_task_yaml() {
  local task_name="$1"
  local yaml_path="./configs/task/${task_name}.yaml"
  if [[ ! -f "${yaml_path}" ]]; then
    echo ""
    return 0
  fi
  local model_name
  model_name="$(grep -m1 '^\s*-\s*override\s*/model\s*:' "${yaml_path}" \
    | sed 's/.*\/model\s*:\s*//' \
    | tr -d '[:space:]')"
  echo "${model_name}"
}

TASK_BASENAME="${DEFAULT_TASK}"
HAS_TASK_OVERRIDE=0
HAS_MODEL_OVERRIDE=0
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      HAS_TASK_OVERRIDE=1
      ;;
    model=*)
      HAS_MODEL_OVERRIDE=1
      ;;
  esac
done

if [[ "${HAS_TASK_OVERRIDE}" -eq 0 ]]; then
  EXTRA_ARGS=("task=${DEFAULT_TASK}" "${EXTRA_ARGS[@]}")
fi

if [[ "${HAS_MODEL_OVERRIDE}" -eq 0 ]]; then
  YAML_MODEL="$(extract_model_from_task_yaml "${TASK_BASENAME}")"
  if [[ -n "${YAML_MODEL}" ]]; then
    echo "[model] resolved from task yaml (${TASK_BASENAME}.yaml): model=${YAML_MODEL}"
    EXTRA_ARGS=("model=${YAML_MODEL}" "${EXTRA_ARGS[@]}")
  else
    echo "[model] task yaml not found or no /model override in it, falling back to default: model=${DEFAULT_MODEL}"
    EXTRA_ARGS=("model=${DEFAULT_MODEL}" "${EXTRA_ARGS[@]}")
  fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    echo "Error: multi-node training requires RUN_ID to be set explicitly." >&2
    echo "Use the same RUN_ID on every machine, for example:" >&2
    echo "  RUN_ID=$(date +%Y-%m-%d_%H-%M-%S) NNODES=${NUM_MACHINES} NODE_RANK=0 MASTER_ADDR=${MAIN_PROCESS_IP} MASTER_PORT=${MAIN_PROCESS_PORT} bash scripts/train_zero1.sh ${NPROC_PER_NODE} task=${TASK_BASENAME}" >&2
    echo "  RUN_ID=<same-run-id> NNODES=${NUM_MACHINES} NODE_RANK=1 MASTER_ADDR=${MAIN_PROCESS_IP} MASTER_PORT=${MAIN_PROCESS_PORT} bash scripts/train_zero1.sh ${NPROC_PER_NODE} task=${TASK_BASENAME}" >&2
    exit 1
  fi
else
  echo "[run_id] using externally provided RUN_ID=${RUN_ID}"
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_processes=${NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT} deepspeed_hostfile=${DEEPSPEED_HOSTFILE:-none} task=${TASK_BASENAME} run_id=${RUN_ID}"
echo "[env] hostname=$(hostname) SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_PROCID=${SLURM_PROCID:-unset} LOCAL_RANK=${LOCAL_RANK:-unset} RANK=${RANK:-unset} WORLD_SIZE=${WORLD_SIZE:-unset}"
echo "[env] NNODES=${NNODES:-unset} NODE_RANK=${NODE_RANK:-unset} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} HOSTFILE=${HOSTFILE:-unset} DS_HOSTFILE=${DS_HOSTFILE:-unset} DEEPSPEED_HOSTFILE=${DEEPSPEED_HOSTFILE:-unset}"
python - <<'PY'
import os
import socket

master_addr = os.environ.get("MASTER_ADDR", "")
if master_addr:
    try:
        resolved = socket.gethostbyname(master_addr)
    except OSError as exc:
        resolved = f"<resolve failed: {exc}>"
else:
    resolved = "<unset>"
print(
    "[env] resolved_master_addr="
    f"{master_addr or '<unset>'}->{resolved} local_ips={socket.gethostbyname_ex(socket.gethostname())[-1]}",
    flush=True,
)
PY

# CUDA allocator: expandable segments reduces fragmentation-induced defrag stalls.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+${PYTORCH_CUDA_ALLOC_CONF},}expandable_segments:True"

#   "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \

HYDRA_FULL_ERROR=1 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${ACCELERATE_ARGS[@]}" \
  scripts/train.py \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
