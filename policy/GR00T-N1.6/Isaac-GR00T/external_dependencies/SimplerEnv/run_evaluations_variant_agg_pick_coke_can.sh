#!/usr/bin/env bash
# Usage: bash run_evaluations_variant_agg_pick_coke_can.sh [PORT]
# Example: bash run_evaluations_variant_agg_pick_coke_can.sh 5556

set -euo pipefail

PORT="${1:-5556}"
shift  # Remove PORT from arguments
ROBOT_TYPE="google"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/evaluation_results-pick_coke_can-${PORT}.log"
mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

# Capture any additional arguments passed to the script
SCRIPT_ARGS=("$@")
echo "Script arguments: ${SCRIPT_ARGS[*]}"

# -------------------- Config --------------------
# Common args (for eval_simpler.py)
COMMON_ARGS=(
  --robot_init_x_range 0.35 0.35 1 
  --robot_init_y_range 0.20 0.20 1 
  --robot_init_rot_quat_center 0 0 0 1
  --robot_init_rot_rpy_range 0 0 1 0 0 1 0 0 1
  --obj_init_x_range -0.35 -0.12 5
  --obj_init_y_range -0.02 0.42 5
  --episode_length 80
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
# Note: shader_dir=rt turns on ray-tracing rendering (policies often rely on shadows for depth cues)
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
coke_can_options_arr=(
    lr_switch=True
    upright=True
    laid_vertically=True
)
for coke_can_option in "${coke_can_options_arr[@]}"; do
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option}"
done

# -------------------- Section 2: Table textures ----
env_name=GraspSingleOpenedCokeCanInScene-v0
background_scenes=(
  Baked_sc1_staging_objaverse_cabinet1_h870
  Baked_sc1_staging_objaverse_cabinet2_h870
)
for scene_name in "${background_scenes[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option}"
  done
done
done

for coke_can_option in "${coke_can_options_arr[@]}"; do
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option} distractor_config=more"
  done
done

# -------------------- Section 3: Distractor ----
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option}"
  done
done

for coke_can_option in "${coke_can_options_arr[@]}"; do
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option} distractor_config=more"
  done
done

# -------------------- Section 4: Backgrounds ------------
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_arr=(
    google_pick_coke_can_1_v4_alt_background
    google_pick_coke_can_1_v4_alt_background_2
)
for scene_name in "${scene_arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option}"
  done
done

# -------------------- Section 5: Lighting ---------------------------
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
for coke_can_option in "${coke_can_options_arr[@]}"; do
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option} slightly_darker_lighting=True"
  run_eval "$env_name" "$scene_name" "--additional_env_build_kwargs ${coke_can_option} slightly_brighter_lighting=True"
done

echo "All evaluations completed. Results saved to ${LOG_FILE}"
