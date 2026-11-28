#!/usr/bin/env bash
VAP_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VAP_POLICY_DIR
XPL_ROOT="$(cd "$VAP_POLICY_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
VAP_DEPLOY_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
export POLICY_ENV="${POLICY_ENV:-$VAP_POLICY_DIR/.venv}"
export SIM_ENV="${SIM_ENV:-${ROBODOJO_SIM_ENV:-}}"
export VAP_REPO="$VAP_POLICY_DIR/upstream"
export VAP_NORMALIZER_STATS="$VAP_POLICY_DIR/configs/action_stats.json"
: "${VAP_QWEN_MODEL:?Set VAP_QWEN_MODEL to the local Qwen3.5-9B directory}"
# Complete VAPv2 checkpoints restore all core/VAE tensors. Training and legacy
# paths may still provide Wan; it is never required by this serving launcher.
if [[ "${VAP_REQUIRE_NO_WAN:-0}" == 1 ]]; then
 unset WALLX_MODEL_ROOT VAP_WAN22_TOKENIZER_PATH
fi
export VAP_QWEN_MODEL
export VAP_OUTPUT_ROOT="${VAP_OUTPUT_ROOT:-$VAP_DEPLOY_ROOT/results}"
export VAP_CACHE_ROOT="${VAP_CACHE_ROOT:-$VAP_POLICY_DIR/.cache}"
if [[ "${VAP_LAUNCH_ROLE:-owner}" == client && "${SIM_ENV}" != /* ]]; then
 echo 'VAP split client requires an absolute simulator environment path; set ROBODOJO_SIM_ENV' >&2
 exit 2
fi
[[ -x "$POLICY_ENV/bin/python" ]] || { echo "Install the public policy environment: bash $VAP_POLICY_DIR/install.sh" >&2; exit 2; }
POLICY_ENV="$(cd "$POLICY_ENV" && pwd -P)"
export POLICY_ENV
export VAP_EVAL_TASK="${VAP_EVAL_TASK:-${2:?Evaluation task is required}}"
export VAP_EVAL_SEED="${VAP_EVAL_SEED:-${6:?Evaluation seed is required}}"
# A direct owner gives its inherited children a unique ID. Independent official
# setup-only roles instead derive one identity from their shared request.
if [[ -z "${ROBODOJO_RUN_ID:-}" && "${VAP_RUN_INITIALIZED:-0}" != 1 && "${VAP_LAUNCH_ROLE:-owner}" != owner ]]; then
 eval "$("$POLICY_ENV/bin/python" -I "$VAP_POLICY_DIR/run_identity.py" --shell "$@")"
fi
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-vap_${VAP_EVAL_TASK}_seed${VAP_EVAL_SEED}_$(date +%Y%m%dT%H%M%S)_${PPID:-$$}}"
# Split launchers must keep the same group/task ID when the parent batch runner
# skips bootstrap and supplies only its retained run directory on a later task.
if [[ "${VAP_LAUNCH_ROLE:-owner}" == client && "${ROBODOJO_RUN_ID}" == */* \
        && -n "${VAP_RUN_DIR:-}" && "$ROBODOJO_RUN_ID" == "$VAP_RUN_DIR" ]]; then
 [[ "$(basename "$VAP_RUN_DIR")" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo 'VAP split run directory has an unsafe identity: '"$VAP_RUN_DIR" >&2
  exit 2
 }
 export ROBODOJO_RUN_ID="$(basename "$VAP_RUN_DIR")"
fi
export VAP_RUN_DIR="${VAP_RUN_DIR:-$VAP_OUTPUT_ROOT/$ROBODOJO_RUN_ID}"
export VAP_REQUEST_SIGNATURE="$("$POLICY_ENV/bin/python" -I "$VAP_POLICY_DIR/run_identity.py" --request "$@")"
validate_run_identity() {
 [[ -d "$VAP_RUN_DIR" && ! -L "$VAP_RUN_DIR" && -f "$VAP_RUN_DIR/.vap_run_id" && ! -L "$VAP_RUN_DIR/.vap_run_id" && -f "$VAP_RUN_DIR/.vap_request.json" && ! -L "$VAP_RUN_DIR/.vap_request.json" ]] || {
  echo "Missing VAP run identity; start the server with this VAP_RUN_DIR first: $VAP_RUN_DIR" >&2
  exit 2
 }
 [[ "$(<"$VAP_RUN_DIR/.vap_run_id")" == "$ROBODOJO_RUN_ID" ]] || {
  echo "VAP run identity mismatch: $VAP_RUN_DIR" >&2
  exit 2
 }
 "$POLICY_ENV/bin/python" -I "$VAP_POLICY_DIR/run_identity.py" --validate "$VAP_RUN_DIR" "$@" || exit 2
}
if [[ "${VAP_DRY_RUN:-0}" != 1 ]]; then
 if [[ "${VAP_LAUNCH_ROLE:-owner}" == client || "${VAP_RUN_INITIALIZED:-0}" == 1 ]]; then
  validate_run_identity "$@"
 else
  [[ "${VAP_LAUNCH_ROLE:-owner}" == server || "${VAP_LAUNCH_ROLE:-owner}" == owner ]] || {
   echo "Unknown VAP launch role: $VAP_LAUNCH_ROLE" >&2
   exit 2
  }
  [[ ! -e "$VAP_RUN_DIR" && ! -L "$VAP_RUN_DIR" ]] || { echo "Run exists: $VAP_RUN_DIR" >&2; exit 2; }
  mkdir -p "$(dirname "$VAP_RUN_DIR")"
  mkdir "$VAP_RUN_DIR"
  "$POLICY_ENV/bin/python" -I "$VAP_POLICY_DIR/run_identity.py" --write "$VAP_RUN_DIR" "$@"
  printf '%s\n' "$ROBODOJO_RUN_ID" > "$VAP_RUN_DIR/.vap_run_id"
  printf '%s\n' "$VAP_EVAL_TASK" > "$VAP_RUN_DIR/task.txt"
  printf '%s\n' "${EVAL_ENV_TYPE:-sim}" > "$VAP_RUN_DIR/mode.txt"
 fi
 export VAP_RUN_INITIALIZED=1
 if [[ "${VAP_LAUNCH_ROLE:-owner}" == client && "${EVAL_ENV_TYPE:-sim}" != debug ]]; then
  [[ -n "${SIM_ENV}" ]] || { echo 'Split VAP client requires ROBODOJO_SIM_ENV or SIM_ENV' >&2; exit 2; }
 fi
fi
export HF_HOME="$VAP_CACHE_ROOT/huggingface" TORCH_HOME="$VAP_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$VAP_CACHE_ROOT/xdg" CUDA_CACHE_PATH="$VAP_CACHE_ROOT/cuda"
export TORCHINDUCTOR_CACHE_DIR="$VAP_CACHE_ROOT/torchinductor" TRITON_CACHE_DIR="$VAP_CACHE_ROOT/triton"
export TMPDIR="$VAP_CACHE_ROOT/tmp"
if [[ "${VAP_DRY_RUN:-0}" != 1 ]]; then
 mkdir -p "$TMPDIR" "$XDG_CACHE_HOME/ov/texturecache"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONSTARTUP
export VAP_SOURCE_PATH="$BENCH_ROOT:$XPL_ROOT:$VAP_REPO"
export PYTHONPATH="$VAP_SOURCE_PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XCOMPUTE_SKIP_AUTO_BUILD=1 XCOMPUTE_DISABLE_AUTO_BUILD=1
export VAP_ROPE_FORWARD_CACHE=0 VAP_FUSED_ROPE=0 VAP_POLICY_TIMING=1
export VAP_QWEN_ONLINE_FROM_CACHE=0 VAP_TEXT_FEATURE_CACHE_ONLINE_FALLBACK=0
export VAP_GRIPPER_POSTPROCESS=0
unset VAP_TEXT_FEATURE_CACHE VAP_FIXED_ACTION_CHUNK_SEED VAP_IMAGE_RGB_GAIN
unset ROBODOJO_EXPECTED_LAYOUT_IDS ROBODOJO_ENABLE_GLOBAL_ILLUMINATION
export ROBODOJO_MAX_BASH_RETRIES=1
if [[ -n "$SIM_ENV" ]]; then
 [[ -x "$SIM_ENV/bin/python" ]] || { echo "Simulator/debug client Python not found: $SIM_ENV" >&2; exit 2; }
 SIM_ENV="$(cd "$SIM_ENV" && pwd -P)"
 export SIM_ENV
fi
# Model configuration belongs to the policy process. The client launches its
# own debug/simulator environment and does not consume these exports.
if [[ "${VAP_LAUNCH_ROLE:-owner}" != client ]]; then
 eval "$("$POLICY_ENV/bin/python" -I "$VAP_POLICY_DIR/config_env.py")"
fi
