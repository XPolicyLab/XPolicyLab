#!/usr/bin/env bash
# RoboDojo evaluation driver.
#
# Two modes:
#
#   bash scripts/eval_robodojo.sh
#       Full official protocol: 42 tasks / 54 configs / 2100 episodes, spread
#       over NUM_CLIENTS Isaac processes (one per client GPU) that consume a
#       shared config queue and fan their requests out over NUM_SERVERS policy
#       services started by scripts/serve_policy.sh.
#
#   bash scripts/eval_robodojo.sh RoboDojo <task> <ckpt_name> <env_cfg> joint \
#                                 <seed> <policy_gpu> <env_gpu>
#       One benchmark config. This is the form cogwam.eval.rollout spawns.
#
# Required environment:
#   ROBODOJO_ROOT        the benchmark checkout (no default; see below)
#   COGWAM_CKPT_PATH     released artifact directory (or training checkpoint)
#   COGWAM_POLICY_HOST   address of the machine running scripts/serve_policy.sh
# Optional:
#   ROBODOJO_PYTHON      interpreter that has Isaac Sim (auto-detected otherwise)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_YAML="${COGWAM_DEPLOY_YAML:-${REPO_ROOT}/configs/robodojo_deploy.yml}"

ROBODOJO_REPOSITORY="https://github.com/RoboDojo-Benchmark/RoboDojo.git"
ROBODOJO_COMMIT="9b4cc885e8f530ed3ab14a30a312ae70242771c4"

if [[ -z "${ROBODOJO_ROOT:-}" ]]; then
  cat >&2 <<EOF
[RoboDojo][ERROR] ROBODOJO_ROOT is not set. The benchmark is a separate project
and is not vendored here. Clone it at the pinned commit and export the path:

  git clone --recurse-submodules ${ROBODOJO_REPOSITORY}
  cd RoboDojo && git checkout ${ROBODOJO_COMMIT}
  git submodule update --init --recursive
  export ROBODOJO_ROOT=\$PWD
EOF
  exit 2
fi
ROBODOJO_ROOT="$(cd "${ROBODOJO_ROOT}" && pwd -P)"
export ROBODOJO_ROOT
XPL_ROOT="${ROBODOJO_ROOT}/XPolicyLab"
UTILS_DIR="${XPL_ROOT}/utils"
OVERLAY_ROOT="${REPO_ROOT}/cogwam/eval/xpolicy_overlay"

export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

resolve_robodojo_python() {
  local candidate
  local -a candidates=()
  [[ -z "${ROBODOJO_PYTHON:-}" ]] || candidates+=("${ROBODOJO_PYTHON}")
  [[ -z "${CONDA_PREFIX:-}" ]] || candidates+=("${CONDA_PREFIX}/bin/python")
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")

  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import importlib.util

required = ("isaacsim", "torch", "torchvision", "cv2", "websockets", "yaml", "msgpack")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! ROBODOJO_PYTHON="$(resolve_robodojo_python)"; then
  echo "[RoboDojo][ERROR] cannot find a Python with the Isaac Sim dependencies." >&2
  echo "[RoboDojo][ERROR] set ROBODOJO_PYTHON=/absolute/path/to/robodojo-env/bin/python." >&2
  echo "[RoboDojo][ERROR] current PATH=${PATH}" >&2
  exit 1
fi
export ROBODOJO_PYTHON

# ---------------------------------------------------------------------------
# Full-protocol mode
# ---------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
  echo "[RoboDojo-fast] entry host=${HOSTNAME:-unknown}"

  # One Isaac process per client GPU, with vectorized environments inside that
  # process. NUM_CLIENTS=8 therefore means exactly eight Isaac processes, never
  # several Isaac processes on one GPU.
  export NUM_CLIENTS="${NUM_CLIENTS:-8}"
  [[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[RoboDojo-fast][ERROR] NUM_CLIENTS must be a positive integer" >&2
    exit 2
  }
  if [[ -z "${CLIENT_CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      export CLIENT_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    else
      client_gpu_ids=""
      for ((gpu = 0; gpu < NUM_CLIENTS; ++gpu)); do
        [[ -z "${client_gpu_ids}" ]] || client_gpu_ids+=","
        client_gpu_ids+="${gpu}"
      done
      export CLIENT_CUDA_VISIBLE_DEVICES="${client_gpu_ids}"
    fi
  fi
  IFS=',' read -r -a configured_client_gpus <<< "${CLIENT_CUDA_VISIBLE_DEVICES}"
  if (( ${#configured_client_gpus[@]} != NUM_CLIENTS )); then
    echo "[RoboDojo-fast][ERROR] NUM_CLIENTS=${NUM_CLIENTS}, but CLIENT_CUDA_VISIBLE_DEVICES exposes ${#configured_client_gpus[@]} entries" >&2
    exit 2
  fi
  export CLIENT_NUM_WORKERS="${CLIENT_NUM_WORKERS:-1}"
  export TASKS_PER_GPU=1
  export NUM_SERVERS="${NUM_SERVERS:-8}"
  export ROBODOJO_ENVS_PER_CLIENT="${ROBODOJO_ENVS_PER_CLIENT:-6}"
  export ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE="${ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE:-25}"
  export N_ACTION_STEPS="${N_ACTION_STEPS:-${ROBODOJO_REPLAN_STEPS:-10}}"
  export ROBODOJO_REPLAN_STEPS="${N_ACTION_STEPS}"
  export NUM_EPISODES="${NUM_EPISODES:-native}"
  # Do not inherit a visualization choice from a reused terminal.
  export ROBODOJO_SAVE_MODE=none
  export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
  export RESULT_DISCOVERY_TIMEOUT="${RESULT_DISCOVERY_TIMEOUT:-60}"
  export SLOT_START_STAGGER_SECONDS="${SLOT_START_STAGGER_SECONDS:-2}"
  export TASK_RETRY_BACKOFF_SECONDS="${TASK_RETRY_BACKOFF_SECONDS:-60}"

  [[ -n "${COGWAM_CKPT_PATH:-}" ]] || {
    echo "[RoboDojo-fast][ERROR] export COGWAM_CKPT_PATH=/absolute/path/to/artifact_dir" >&2
    exit 2
  }
  [[ -n "${COGWAM_POLICY_HOST:-}" ]] || {
    echo "[RoboDojo-fast][ERROR] export COGWAM_POLICY_HOST=<policy-server-IP> (see scripts/serve_policy.sh)" >&2
    exit 2
  }

  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  exec "${ROBODOJO_PYTHON}" -u -m cogwam.eval.rollout
fi

# ---------------------------------------------------------------------------
# Single-config mode
# ---------------------------------------------------------------------------
if [[ $# -ne 8 ]]; then
  echo "Usage: bash eval_robodojo.sh <bench> <task> <ckpt_name> <env_cfg> <action_type> <seed> <policy_gpu> <env_gpu>" >&2
  echo "   or: bash eval_robodojo.sh   (full 42-task / 2100-episode protocol)" >&2
  exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8

CHECKPOINT_PATH="${COGWAM_CKPT_PATH:?export COGWAM_CKPT_PATH=/absolute/path/to/artifact_dir}"
REPLAN_STEPS="${ROBODOJO_REPLAN_STEPS:-10}"
EXPECTED_ACTION_CHUNK_SIZE="${ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE:-25}"
if ! [[ "${EXPECTED_ACTION_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[RoboDojo][ERROR] ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE must be a positive integer, got ${EXPECTED_ACTION_CHUNK_SIZE}" >&2
  exit 2
fi
if ! [[ "${REPLAN_STEPS}" =~ ^[0-9]+$ ]] || (( 10#${REPLAN_STEPS} < 1 || 10#${REPLAN_STEPS} > 10#${EXPECTED_ACTION_CHUNK_SIZE} )); then
  echo "[RoboDojo][ERROR] ROBODOJO_REPLAN_STEPS must be an integer in [1,${EXPECTED_ACTION_CHUNK_SIZE}], got ${REPLAN_STEPS}" >&2
  exit 2
fi
RTC_ENABLED="${ROBODOJO_RTC_ENABLED:-false}"
case "${RTC_ENABLED,,}" in
  1|true|yes|on) RTC_ENABLED=true ;;
  0|false|no|off) RTC_ENABLED=false ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_RTC_ENABLED must be a boolean, got ${RTC_ENABLED}" >&2
    exit 2
    ;;
esac
RTC_OVERLAP=$((10#${EXPECTED_ACTION_CHUNK_SIZE} - 10#${REPLAN_STEPS}))
RTC_EXECUTION_HORIZON="${ROBODOJO_RTC_EXECUTION_HORIZON:-$((RTC_OVERLAP > 0 ? RTC_OVERLAP : 1))}"
RTC_INFERENCE_DELAY="${ROBODOJO_RTC_INFERENCE_DELAY:-1}"
RTC_MAX_GUIDANCE_WEIGHT="${ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT:-20.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE:-linear}"
RTC_DEBUG_MAX_REPLANS="${ROBODOJO_RTC_DEBUG_MAX_REPLANS:-8}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE,,}"
if ! [[ "${RTC_EXECUTION_HORIZON}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[RoboDojo][ERROR] ROBODOJO_RTC_EXECUTION_HORIZON must be a positive integer, got ${RTC_EXECUTION_HORIZON}" >&2
  exit 2
fi
if ! [[ "${RTC_INFERENCE_DELAY}" =~ ^[0-9]+$ ]]; then
  echo "[RoboDojo][ERROR] ROBODOJO_RTC_INFERENCE_DELAY must be a non-negative integer, got ${RTC_INFERENCE_DELAY}" >&2
  exit 2
fi
if ! [[ "${RTC_DEBUG_MAX_REPLANS}" =~ ^[0-9]+$ ]]; then
  echo "[RoboDojo][ERROR] ROBODOJO_RTC_DEBUG_MAX_REPLANS must be a non-negative integer, got ${RTC_DEBUG_MAX_REPLANS}" >&2
  exit 2
fi
if ! [[ "${RTC_MAX_GUIDANCE_WEIGHT}" =~ ^(([1-9][0-9]*)([.][0-9]*)?|0[.][0-9]*[1-9][0-9]*)$ ]]; then
  echo "[RoboDojo][ERROR] ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT must be positive, got ${RTC_MAX_GUIDANCE_WEIGHT}" >&2
  exit 2
fi
case "${RTC_PREFIX_ATTENTION_SCHEDULE}" in
  exp|linear|ones|zeros) ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE must be exp, linear, ones, or zeros, got ${RTC_PREFIX_ATTENTION_SCHEDULE}" >&2
    exit 2
    ;;
esac
if [[ "${RTC_ENABLED}" == "true" ]] && {
  (( RTC_OVERLAP < 1 )) \
    || (( 10#${RTC_EXECUTION_HORIZON} > RTC_OVERLAP )) \
    || (( 10#${RTC_INFERENCE_DELAY} > 10#${RTC_EXECUTION_HORIZON} ));
}; then
  echo "[RoboDojo][ERROR] RTC with H${EXPECTED_ACTION_CHUNK_SIZE}/replan${REPLAN_STEPS} requires execution_horizon in [1,${RTC_OVERLAP}] and inference_delay <= execution_horizon; got horizon=${RTC_EXECUTION_HORIZON}, delay=${RTC_INFERENCE_DELAY}" >&2
  exit 2
fi
LOG_PLANNER_TEXT="${ROBODOJO_LOG_PLANNER_TEXT:-false}"
case "${LOG_PLANNER_TEXT,,}" in
  1|true|yes|on) LOG_PLANNER_TEXT=true ;;
  0|false|no|off) LOG_PLANNER_TEXT=false ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_LOG_PLANNER_TEXT must be a boolean, got ${LOG_PLANNER_TEXT}" >&2
    exit 2
    ;;
esac
NUM_ENVS="${ROBODOJO_ENVS_PER_CLIENT:-1}"
if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ ]] || (( 10#${NUM_ENVS} < 1 )); then
  echo "[RoboDojo][ERROR] ROBODOJO_ENVS_PER_CLIENT must be a positive integer, got ${NUM_ENVS}" >&2
  exit 2
fi
if (( 10#${NUM_ENVS} > 1 )); then
  EVAL_BATCH=true
else
  EVAL_BATCH=false
fi

EVAL_MODE="${ROBODOJO_EVAL_MODE:-fast}"
EVAL_MODE="${EVAL_MODE,,}"
case "${EVAL_MODE}" in
  fast)
    requested_trials="${ROBODOJO_TRIALS:-${EVAL_NUM:-native}}"
    ;;
  visualize)
    requested_trials="${ROBODOJO_TRIALS:-${EVAL_NUM:-5}}"
    ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_EVAL_MODE must be fast or visualize, got ${EVAL_MODE}" >&2
    exit 2
    ;;
esac

# Artifact selector. The default preserves the released behavior: fast
# evaluation writes no media, while visualize writes RoboDojo's native MP4s.
SAVE_MODE="${ROBODOJO_SAVE_MODE:-}"
if [[ -z "${SAVE_MODE}" ]]; then
  [[ "${EVAL_MODE}" == "visualize" ]] && SAVE_MODE=video || SAVE_MODE=none
fi
SAVE_MODE="${SAVE_MODE,,}"
case "${SAVE_MODE}" in
  none) export ROBODOJO_DISABLE_EVAL_VIDEO=1 ;;
  video) export ROBODOJO_DISABLE_EVAL_VIDEO=0 ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_SAVE_MODE must be none or video, got ${SAVE_MODE}" >&2
    exit 2
    ;;
esac
export ROBODOJO_SAVE_MODE="${SAVE_MODE}"
if [[ "${requested_trials}" != "native" ]] && { ! [[ "${requested_trials}" =~ ^[0-9]+$ ]] || (( 10#${requested_trials} <= 0 )); }; then
  echo "[RoboDojo][ERROR] ROBODOJO_TRIALS must be native or a positive integer, got ${requested_trials}" >&2
  exit 2
fi
if [[ "${requested_trials}" == "native" ]]; then
  unset EVAL_NUM
else
  export EVAL_NUM="${requested_trials}"
fi

# Keep standalone artifacts next to the evaluated model. The full-protocol
# coordinator supplies ROBODOJO_OUTPUT_ROOT for each config.
if [[ -d "${CHECKPOINT_PATH}" ]]; then
  MODEL_ROOT="${CHECKPOINT_PATH}"
elif [[ "${CHECKPOINT_PATH}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(dirname "$(dirname "${CHECKPOINT_PATH}")")"
else
  MODEL_ROOT="$(dirname "${CHECKPOINT_PATH}")"
fi
MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd -P)"
OUTPUT_RUN_ID="${ROBODOJO_OUTPUT_RUN_ID:-${ROBODOJO_RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$}}"
trial_tag="$([[ "${requested_trials}" == "native" ]] && printf native || printf 'trials%s' "${requested_trials}")"
RUN_NAME="${ROBODOJO_RUN_NAME:-${EVAL_MODE}_${task_name}_${trial_tag}}"
EXPLICIT_OUTPUT_ROOT="${ROBODOJO_OUTPUT_ROOT:-${OUTPUT_ROOT:-}}"
requested_output_root="${ROBODOJO_OUTPUT_ROOT:-${OUTPUT_ROOT:-${MODEL_ROOT}/robodojo_eval_results/${RUN_NAME}_${OUTPUT_RUN_ID}}}"
mkdir -p "${requested_output_root}"
OUTPUT_ROOT="$(cd "${requested_output_root}" && pwd -P)"
LOG_DIR="${OUTPUT_ROOT}/logs"
NATIVE_WORKDIR="${OUTPUT_ROOT}/native"
NATIVE_RESULT_ROOT="${NATIVE_WORKDIR}/eval_result"
CLIENT_LOG="${LOG_DIR}/client.log"
CLIENT_RAW_LOG="${LOG_DIR}/client.raw.log"
mkdir -p "${LOG_DIR}" "${NATIVE_WORKDIR}"
export ROBODOJO_EVAL_RESULT_ROOT="${NATIVE_RESULT_ROOT}"

# Keep the normal client log readable in fast mode while retaining every byte
# emitted by Isaac/Kit in client.raw.log. Set ROBODOJO_CONCISE_LOGS=0 when
# debugging simulator startup.
if [[ -n "${ROBODOJO_CONCISE_LOGS:-}" ]]; then
  CONCISE_LOGS="${ROBODOJO_CONCISE_LOGS}"
elif [[ "${EVAL_MODE}" == "fast" ]]; then
  CONCISE_LOGS=1
else
  CONCISE_LOGS=0
fi
filter_robodojo_output() {
  local line cuda_enumeration_warning_seen=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${ROBODOJO_HIDE_CAMERA_APERTURE_WARNING:-1}" == "1" \
      && "${line}" == *"[Warning] [isaacsim.sensors.camera.camera]"* \
      && "${line}" == *"are inconsistent with the pixel resolution aspect ratio"* \
      && "${line}" == *"Setting 'verticalAperture'"* ]]; then
      continue
    fi
    if [[ "${CONCISE_LOGS}" == "1" ]]; then
      # Extension-by-extension startup and headless desktop warnings dominate
      # Isaac's output but do not describe rollout progress. They remain in
      # client.raw.log for post-mortem debugging.
      if [[ -z "${line}" ]] \
        || [[ "${line}" =~ ^\[[0-9]+\.[0-9]+s\]\ \[ext: ]] \
        || [[ "${line}" == \|* ]] \
        || [[ "${line}" == *"======================================================================================"* ]] \
        || [[ "${line}" == "Loading user config located at:"* ]] \
        || [[ "${line}" == "[Info] [carb] Logging to file:"* ]] \
        || [[ "${line}" == *"carb.windowing-glfw.plugin"* ]] \
        || [[ "${line}" == *"[omni.platforminfo.plugin] failed to open the default display"* ]] \
        || [[ "${line}" == *"[carb.cudainterop.plugin]"* ]] \
        || [[ "${line}" == *"IPhysxBenchmarks"* ]] \
        || [[ "${line}" == "  warnings.warn("* ]] \
        || [[ "${line}" == *"Different types of CPU governors"* ]] \
        || [[ "${line}" == *"[omni.log] Source: omni.hydra was already registered"* ]] \
        || [[ "${line}" == *"[omni.isaac.dynamic_control]"*"deprecated"* ]] \
        || [[ "${line}" == *"[carb.audio."* ]] \
        || [[ "${line}" == *"[omni.fabric.plugin] Warning: attribute viewportHandle"* ]] \
        || [[ "${line}" == *"[omni.usd] Warning"* ]] \
        || [[ "${line}" == *"[omni.graph.core.plugin]"* ]] \
        || [[ "${line}" == *"[omni.replicator.core.scripts.extension]"* ]] \
        || [[ "${line}" == *"RequestsDependencyWarning"* ]] \
        || [[ "${line}" == *"pxr.Semantics is deprecated"* ]] \
        || [[ "${line}" == XDG_RUNTIME_DIR* ]] \
        || [[ "${line}" == "sh: 1: zenity: not found" ]] \
        || [[ "${line}" == *"Simulation App Starting" ]] \
        || [[ "${line}" == *"Simulation App Startup Complete" ]] \
        || [[ "${line}" == *"] app ready" ]] \
        || [[ "${line}" == *"[omni.kvdb.plugin] Disabling key-value database because another kit process is locking it"* ]] \
        || [[ "${line}" == *"enable_external_forces_every_iteration"* ]] \
        || [[ "${line}" == *"[env.scene_manager.objects.background]"*"RTSubframes"* ]] \
        || [[ "${line}" == *"[isaacsim.core.prims.impl.cloth_prim]"*"deprecated"* ]] \
        || [[ "${line}" == "Warning: Material prim at "*" has no children." ]]; then
        continue
      fi
      if [[ "${line}" == *"Skipping NVIDIA GPU due CUDA being in bad state"* ]] \
        || [[ "${line}" == *"Please restart your system if CUDA is known to work"* ]]; then
        if (( cuda_enumeration_warning_seen == 0 )); then
          printf '%s\n' \
            "[RoboDojo][WARN] Isaac reported a CUDA device-enumeration warning; continuing with the selected active GPU. Full details: ${CLIENT_RAW_LOG}"
          cuda_enumeration_warning_seen=1
        fi
        continue
      fi
    fi
    printf '%s\n' "${line}"
  done
}
exec > >(tee -a "${CLIENT_RAW_LOG}" | filter_robodojo_output | tee -a "${CLIENT_LOG}") 2>&1

if [[ "${bench_name}" != "RoboDojo" ]]; then
  echo "[RoboDojo][ERROR] this adapter only supports bench_name=RoboDojo, got ${bench_name}" >&2
  exit 2
fi
if [[ "${action_type}" != "joint" ]]; then
  echo "[RoboDojo][ERROR] this checkpoint ABI requires action_type=joint, got ${action_type}" >&2
  exit 2
fi

for required in \
  "${XPL_ROOT}/setup_policy_server.py" \
  "${UTILS_DIR}/get_free_port.sh" \
  "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${ROBODOJO_ROOT}/src/eval_client/main.py" \
  "${ROBODOJO_ROOT}/env_cfg/${env_cfg_type}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/config/${task_name}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/tasks/${task_name}.py" \
  "${REPO_ROOT}/cogwam/eval/launch_bridge.py" \
  "${REPO_ROOT}/cogwam/eval/launch_client.py" \
  "${OVERLAY_ROOT}/XPolicyLab/policy/cogwam/model.py" \
  "${OVERLAY_ROOT}/XPolicyLab/policy/cogwam/deploy.py" \
  "${DEPLOY_YAML}"; do
  [[ -f "${required}" ]] || { echo "[RoboDojo][ERROR] missing ${required}" >&2; exit 1; }
done
[[ -e "${CHECKPOINT_PATH}" ]] || { echo "[RoboDojo][ERROR] missing ${CHECKPOINT_PATH}" >&2; exit 1; }

# RoboDojo resolves assets relative to its checkout. Validate the official
# directory names before importing Isaac Sim so an incomplete mount fails in
# seconds instead of after a long simulator startup.
for required_dir in \
  "${ROBODOJO_ROOT}/Assets/Robots" \
  "${ROBODOJO_ROOT}/Assets/Object" \
  "${ROBODOJO_ROOT}/Assets/Material" \
  "${ROBODOJO_ROOT}/Assets/Eval_Layout"; do
  [[ -d "${required_dir}" ]] || { echo "[RoboDojo][ERROR] missing asset directory ${required_dir}" >&2; exit 1; }
done

# This launcher is client-only. Requiring an explicit remote address prevents
# an accidental second model copy from being loaded in the Isaac client job.
policy_host="${COGWAM_POLICY_HOST:?export COGWAM_POLICY_HOST=<policy server IP>}"
policy_remote_port="${COGWAM_POLICY_PORT:-7777}"
if [[ -n "${ROBODOJO_XPOLICY_PORT:-}" ]]; then
  [[ "${ROBODOJO_XPOLICY_PORT}" =~ ^[0-9]+$ ]] \
    && (( ROBODOJO_XPOLICY_PORT >= 1 && ROBODOJO_XPOLICY_PORT <= 65535 )) || {
      echo "[RoboDojo][ERROR] ROBODOJO_XPOLICY_PORT must be an integer in [1, 65535]" >&2
      exit 2
    }
  bridge_port="${ROBODOJO_XPOLICY_PORT}"
else
  bridge_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
fi

BRIDGE_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z "${BRIDGE_PID}" ]] || kill "${BRIDGE_PID}" 2>/dev/null || true
  [[ -z "${BRIDGE_PID}" ]] || wait "${BRIDGE_PID}" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT INT TERM

# The overlay must stay before RoboDojo's own XPolicyLab package in both the
# bridge and the simulator process. This makes model.py *and* deploy.py come
# from this repository while all transport/simulator modules are resolved from
# the external RoboDojo checkout.
export PYTHONPATH="${OVERLAY_ROOT}:${REPO_ROOT}:${ROBODOJO_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

# RoboDojo includes both ``additional_info`` and ``ROBODOJO_RUN_ID`` in its
# native directory tree. The outer OUTPUT_ROOT is already unique and carries a
# timestamp, so keep these inner components stable and short.
export ROBODOJO_RUN_ID="${ROBODOJO_NATIVE_RUN_ID:-run}"

"${ROBODOJO_PYTHON}" - \
  "${OUTPUT_ROOT}/run_metadata.json" \
  "${CHECKPOINT_PATH}" \
  "${ckpt_name}" \
  "${seed}" \
  "${task_name}" \
  "${EVAL_MODE}" \
  "${requested_trials}" \
  "${EXPECTED_ACTION_CHUNK_SIZE}" \
  "${REPLAN_STEPS}" \
  "${RTC_ENABLED}" \
  "${RTC_EXECUTION_HORIZON}" \
  "${RTC_INFERENCE_DELAY}" \
  "${RTC_MAX_GUIDANCE_WEIGHT}" \
  "${RTC_PREFIX_ATTENTION_SCHEDULE}" \
  "${RTC_DEBUG_MAX_REPLANS}" \
  "${RUN_NAME}" \
  "${OUTPUT_RUN_ID}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    output_path,
    checkpoint,
    ckpt_name,
    seed,
    task,
    eval_mode,
    trials,
    action_chunk_size,
    replan_steps,
    rtc_enabled,
    rtc_execution_horizon,
    rtc_inference_delay,
    rtc_max_guidance_weight,
    rtc_prefix_attention_schedule,
    rtc_debug_max_replans,
    run_name,
    run_id,
) = sys.argv[1:]
payload = {
    "checkpoint": str(Path(checkpoint).expanduser().resolve()),
    "ckpt_name": ckpt_name,
    "seed": int(seed),
    "task": task,
    "eval_mode": eval_mode,
    "save_mode": os.environ["ROBODOJO_SAVE_MODE"],
    "trials": trials,
    "action_chunk_size": int(action_chunk_size),
    "replan_steps": int(replan_steps),
    "rtc": {
        "enabled": rtc_enabled == "true",
        "execution_horizon": int(rtc_execution_horizon),
        "inference_delay": int(rtc_inference_delay),
        "max_guidance_weight": float(rtc_max_guidance_weight),
        "prefix_attention_schedule": rtc_prefix_attention_schedule,
        "debug_max_replans": int(rtc_debug_max_replans),
    },
    "run_name": run_name,
    "run_id": run_id,
}
path = Path(output_path)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "[RoboDojo] repository=${REPO_ROOT}"
echo "[RoboDojo] simulator=${ROBODOJO_ROOT} checkpoint=${CHECKPOINT_PATH}"
echo "[RoboDojo] python=${ROBODOJO_PYTHON}"
echo "[RoboDojo] task=${task_name} policy-server=${policy_host}:${policy_remote_port} bridge-port=${bridge_port}"
echo "[RoboDojo] mode=${EVAL_MODE} trials=${requested_trials} save_mode=${SAVE_MODE}"
echo "[RoboDojo] action_chunk=${EXPECTED_ACTION_CHUNK_SIZE} replan_steps=${REPLAN_STEPS}"
echo "[RoboDojo] rtc=${RTC_ENABLED} overlap=${RTC_OVERLAP} execution_horizon=${RTC_EXECUTION_HORIZON} inference_delay=${RTC_INFERENCE_DELAY} schedule=${RTC_PREFIX_ATTENTION_SCHEDULE} max_guidance_weight=${RTC_MAX_GUIDANCE_WEIGHT} debug_max_replans=${RTC_DEBUG_MAX_REPLANS}"
echo "[RoboDojo] vector_envs=${NUM_ENVS} eval_batch=${EVAL_BATCH}"
echo "[RoboDojo] output=${OUTPUT_ROOT}"
echo "[RoboDojo] log=${CLIENT_LOG} raw_log=${CLIENT_RAW_LOG}"
echo "[RoboDojo] native-results=${NATIVE_RESULT_ROOT}"

# Use the repository-owned launcher instead of executing XPolicyLab's script
# directly. It makes the overlay import deterministic and supplies the narrow
# server compatibility required by the client image's websockets package.
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" "${ROBODOJO_PYTHON}" -m cogwam.eval.launch_bridge \
  --config_path "${DEPLOY_YAML}" \
  --overrides \
    port="${bridge_port}" \
    host=127.0.0.1 \
    bench_name="${bench_name}" \
    task_name="${task_name}" \
    ckpt_name="${ckpt_name}" \
    env_cfg_type="${env_cfg_type}" \
    action_type="${action_type}" \
    seed="${seed}" \
    policy_server_host="${policy_host}" \
    policy_server_port="${policy_remote_port}" \
    expected_checkpoint_path="${CHECKPOINT_PATH}" \
    eval_batch="${EVAL_BATCH}" \
    replan_interval="${REPLAN_STEPS}" \
    rtc_enabled="${RTC_ENABLED}" \
    rtc_execution_horizon="${RTC_EXECUTION_HORIZON}" \
    rtc_inference_delay="${RTC_INFERENCE_DELAY}" \
    rtc_max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}" \
    rtc_prefix_attention_schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}" \
    rtc_debug_max_replans="${RTC_DEBUG_MAX_REPLANS}" \
    log_planner_text="${LOG_PLANNER_TEXT}" \
    expected_action_chunk_size="${EXPECTED_ACTION_CHUNK_SIZE}" \
    expected_action_dim=14 &
BRIDGE_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  127.0.0.1 "${bridge_port}" "${BRIDGE_PID}" "RoboDojo XPolicy bridge" 600

# Upstream RoboDojo uses this string as a directory name. Detailed checkpoint
# and rollout settings live in OUTPUT_ROOT/run_metadata.json instead.
additional_info="eval"

# Do not call RoboDojo's generic eval_policy.sh here: that script prepends its
# own XPolicyLab directory and can silently import a stale/missing adapter.
# Launch the same official simulator entry directly with our overlay first,
# retaining RoboDojo's bounded PhysX restart behavior.
max_retries="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
kit_args="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
while :; do
  set +e
  (
    # eval_env.py intentionally writes to ./eval_result. All source/config
    # paths are resolved from ROBODOJO_ROOT/PYTHONPATH, so changing only cwd
    # redirects native results without touching the shared simulator tree.
    cd "${NATIVE_WORKDIR}"
    export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
    # RoboDojo's official main re-reads its own XPolicy deploy.yml and
    # otherwise turns a requested vector rollout back into one environment.
    # launch_client.py applies this per-process override without editing the
    # shared simulator checkout.
    export ROBODOJO_EVAL_BATCH="${EVAL_BATCH}"
    exec "${ROBODOJO_PYTHON}" -u -m cogwam.eval.launch_client \
      --task_name "${task_name}" \
      --env_cfg_type "${env_cfg_type}" \
      --num_envs "${NUM_ENVS}" \
      --enable_cameras \
      --kit_args "${kit_args}" \
      --device_id "${env_gpu_id}" \
      --policy_name cogwam \
      --host 127.0.0.1 \
      --port "${bridge_port}" \
      --protocol ws \
      --policy_server_url "ws://127.0.0.1:${bridge_port}" \
      --additional_info "${additional_info}" \
      --seed "${seed}" \
      --headless
  )
  rc=$?
  set -e
  case "${rc}" in
    0)
      break
      ;;
    99|134|139)
      attempt=$((attempt + 1))
      if (( attempt >= max_retries )); then
        echo "[RoboDojo][ERROR] simulator failed rc=${rc} after ${attempt} attempts" >&2
        exit "${rc}"
      fi
      echo "[RoboDojo] simulator restart ${attempt}/${max_retries} after rc=${rc}" >&2
      sleep 5
      ;;
    *)
      exit "${rc}"
      ;;
  esac
done

echo "[RoboDojo] evaluation finished"
echo "[RoboDojo] output=${OUTPUT_ROOT}"
if [[ "${ROBODOJO_SKIP_TABLE1_SUMMARY:-0}" != "1" ]]; then
  SUMMARY_EVAL_ROOT="${ROBODOJO_SUMMARY_EVAL_ROOT:-${MODEL_ROOT}/robodojo_eval_results}"
  if [[ -n "${EXPLICIT_OUTPUT_ROOT}" ]]; then
    SUMMARY_EVAL_ROOT="${ROBODOJO_SUMMARY_EVAL_ROOT:-${OUTPUT_ROOT}}"
  fi
  "${ROBODOJO_PYTHON}" -m cogwam.eval.summarize \
    --eval-root "${SUMMARY_EVAL_ROOT}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --ckpt-name "${ckpt_name}" \
    --seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}"
fi
