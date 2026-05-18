#!/usr/bin/env bash
# Usage: bash run_evaluations_variant_agg_drawer.sh [PORT]
# Example: bash run_evaluations_variant_agg_drawer.sh 5556  
# Example: bash run_evaluations_variant_agg_drawer.sh 5556 --headless

set -euo pipefail

PORT="${1:-5556}"
shift  # Remove PORT from arguments
ROBOT_TYPE="google"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/evaluation_results-drawer-${PORT}.log"
mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

# Capture any additional arguments passed to the script
SCRIPT_ARGS=("$@")
echo "Script arguments: ${SCRIPT_ARGS[*]}"

# -------------------- Config --------------------
# Common args (for eval_simpler.py)
COMMON_ARGS=(
  --robot_init_x_range 0.65 0.85 3
  --robot_init_y_range -0.2 0.2 3
  --robot_init_rot_quat_center 0 0 0 1
  --robot_init_rot_rpy_range 0 0 1 0 0 1 0.0 0.0 1
  --obj_init_x_range 0 0 1
  --obj_init_y_range 0 0 1
  --episode_length 113
  --eval_count 1
)

# Helper to run one eval (env + scene + extra args)
run_eval() {
  local env_name="$1"
  local scene_name="$2"
  shift 2
  # remaining "$@" are extra args

  # Build command as an array (safe quoting)
  local -a cmd=(python eval_simpler.py
    --env_name "$env_name"
    --scene_name "$scene_name"
    "${COMMON_ARGS[@]}"
    --robot_type "$ROBOT_TYPE"
    --groot_port "$PORT"
    "${SCRIPT_ARGS[@]}"
  )

  # Append extra args (tokenized)
  if [[ $# -gt 0 ]]; then
    # shellcheck disable=SC2206
    local extra=( $* )
    cmd+=("${extra[@]}")
  fi

  {
    echo
    echo "[$(date '+%F %T')] Running:"
    printf ' %q' "${cmd[@]}"; echo
    echo "--------------------------------------"
  } | tee -a "${LOG_FILE}"

  # Execute and capture output safely
  set +e
  output="$("${cmd[@]}" 2>&1)"
  exit_code=$?
  set -e

  last_lines="$(printf "%s\n" "$output" | tail -n 2)"

  {
    echo "Exit code: ${exit_code}"
    echo "Last 2 lines:"
    echo "${last_lines}"
    echo "======================================"
  } | tee -a "${LOG_FILE}"
}

echo "Starting evaluations on port ${PORT}..." | tee -a "${LOG_FILE}"
echo "======================================" | tee -a "${LOG_FILE}"

# -------------------- Section 1: Base scene --------------------
# Environments to evaluate
env_names=(
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
)

for env_name in "${env_names[@]}"; do
  run_eval "$env_name" "frl_apartment_stage_simple" "--additional_env_build_kwargs shader_dir=rt"
done

# -------------------- Section 2: Backgrounds ----
# Background variants
background_scenes=(
  modern_bedroom_no_roof
  modern_office_no_roof
)
for scene_name in "${background_scenes[@]}"; do
  for env_name in "${env_names[@]}"; do
    run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs shader_dir=rt"
  done
done

# -------------------- Section 3: Lighting ------------
for env_name in "${env_names[@]}"; do
  run_eval "$env_name" "frl_apartment_stage_simple" "--additional_env_build_kwargs shader_dir=rt light_mode=brighter"
  run_eval "$env_name" "frl_apartment_stage_simple" "--additional_env_build_kwargs shader_dir=rt light_mode=darker"
done

# -------------------- Section 4: New cabinet stations ---------------------------
for env_name in "${env_names[@]}"; do
  run_eval "$env_name" "frl_apartment_stage_simple" "--additional_env_build_kwargs shader_dir=rt station_name=mk_station2"
  run_eval "$env_name" "frl_apartment_stage_simple" "--additional_env_build_kwargs shader_dir=rt station_name=mk_station3"
done

echo "All evaluations completed. Results saved to ${LOG_FILE}"
