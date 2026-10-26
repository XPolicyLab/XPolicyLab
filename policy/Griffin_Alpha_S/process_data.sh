#!/usr/bin/env bash
# Prepare a Griffin Alpha-S fine-tuning run from RoboDojo demonstrations, in two steps:
#
#   1. Convert the trajectories to a LeRobot v3.0 dataset with the OFFICIAL converter
#      (XPolicyLab/scripts/transform_lerobot_v30_format.py), repo id
#      <bench_name>-<ckpt_name>-<env_cfg_type>-<action_type> under HF_LEROBOT_HOME.
#   2. Rebuild the base checkpoint's processors for that dataset with the plugin's
#      scripts/make_finetune_base.py (camera keys, embodiment prompt, control mode, relative-action
#      stats) and write the result to bases/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/,
#      which train.sh then passes to lerobot-train as --policy.path.
#
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [task_pattern]
#   expert_data_num  episodes per matched task (default: converter default, 200)
#   task_pattern     task glob for "<bench_name>.<task_pattern>.<env_cfg_type>" (default: ckpt_name;
#                    use "*" for every task under the robot, e.g. a cotrain set)
#
# Environment overrides:
#   GRIFFIN_HEAD                    flow (default) or fast: which base head to prepare
#   GRIFFIN_BASE                    base checkpoint (default griffinlabs/Griffin-Alpha-S)
#   GRIFFIN_DATASET_REPO_ID         reuse an existing LeRobot dataset and skip the conversion step
#   GRIFFIN_EMBODIMENT_PROMPT       prompt text describing the robot (default derived from env_cfg_type)
#   GRIFFIN_N_ACTION_STEPS          replan stride baked into the base (default 10; the chunk is 50 steps)
#   GRIFFIN_RELATIVE_ACTIONS        1 (default) predicts arm dims relative to the state, 0 absolute deltas
#   GRIFFIN_RELATIVE_EXCLUDE_JOINTS space-separated action names kept absolute (default: the gripper
#                                   joints derived from utils/robot/_robot_info.json)
#   GRIFFIN_MAKE_BASE_EXTRA_ARGS    extra flags for make_finetune_base.py (e.g. "--no-include_proprio")
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [task_pattern]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
task_pattern=${6:-${ckpt_name}}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
ALPHA_S_ROOT="${GRIFFIN_ALPHA_S_ROOT:-${POLICY_DIR}/alpha-s}"
CONDA_ENV="${GRIFFIN_CONDA_ENV:-griffin_alpha_s}"

if ! python -c "import lerobot_policy_griffin_alpha" >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_repo_id="${GRIFFIN_DATASET_REPO_ID:-${data_setting}}"
base_dir="${POLICY_DIR}/bases/${data_setting}"
head="${GRIFFIN_HEAD:-flow}"
case "${head}" in
  flow) revision="main" ;;
  fast) revision="fast" ;;
  *) echo "[Griffin_Alpha_S] GRIFFIN_HEAD must be flow or fast, got '${head}'" >&2; exit 1 ;;
esac

# ---- 1. Official LeRobot v3.0 conversion -------------------------------------------------------
if [[ -n "${GRIFFIN_DATASET_REPO_ID:-}" ]]; then
  echo "[Griffin_Alpha_S] GRIFFIN_DATASET_REPO_ID set: reusing dataset ${dataset_repo_id}, skipping conversion"
else
  convert_args=("${bench_name}.${task_pattern}.${env_cfg_type}" --repo_id "${dataset_repo_id}")
  if [[ -n "${expert_data_num}" ]]; then
    convert_args+=(--max_episode "${expert_data_num}")
  fi
  echo "[Griffin_Alpha_S] converting ${convert_args[0]} -> LeRobot v3.0 dataset ${dataset_repo_id}"
  (cd "${XPL_ROOT}" && python scripts/transform_lerobot_v30_format.py "${convert_args[@]}")
fi

# ---- 2. Rebuild the base's processors for this dataset -----------------------------------------
# The official converter names motors <left|right|arm>_joint_<i>, arm joints first then the
# end-effector joints, so the gripper names have no "gripper" in them. Relative actions need the
# grippers kept absolute, so derive their names from the robot's arm/ee dims. This is the training
# path, so it reads utils/robot/_robot_info.json like utils/get_action_dim.sh does.
if [[ -n "${GRIFFIN_RELATIVE_EXCLUDE_JOINTS:-}" ]]; then
  read -r -a exclude_joints <<< "${GRIFFIN_RELATIVE_EXCLUDE_JOINTS}"
else
  exclude_joints=()
  while IFS= read -r joint; do exclude_joints+=("${joint}"); done < <(python3 - "${XPL_ROOT}" "${env_cfg_type}" <<'PY'
import json, sys
root, env_cfg_type = sys.argv[1], sys.argv[2]
info = json.load(open(f"{root}/utils/robot/_robot_info.json", encoding="utf-8"))[env_cfg_type]
arm_dims, ee_dims = info["arm_dim"], info["ee_dim"]
prefixes = ["arm"] if len(arm_dims) == 1 else ["left", "right"]
for prefix, arm_dim, ee_dim in zip(prefixes, arm_dims, ee_dims):
    for j in range(arm_dim, arm_dim + ee_dim):
        print(f"{prefix}_joint_{j}")
PY
)
fi
num_arms=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[sys.argv[2]]["arm_dim"]))' \
  "${XPL_ROOT}/utils/robot/_robot_info.json" "${env_cfg_type}")
embodiment_prompt="${GRIFFIN_EMBODIMENT_PROMPT:-${bench_name} ${env_cfg_type} robot, ${num_arms} gripper$([[ "${num_arms}" == 1 ]] || echo s)}"
if [[ "${action_type}" == "joint" ]]; then arm_control_mode="joint"; else arm_control_mode="eef_pose"; fi  # eval supports joint only, see README

make_base_args=(
  --base "${GRIFFIN_BASE:-griffinlabs/Griffin-Alpha-S}" --revision "${revision}"
  --dataset_repo_id "${dataset_repo_id}"
  --output_dir "${base_dir}"
  --image_keys observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist
  --embodiment_prompt "${embodiment_prompt}"
  --arm_control_mode "${arm_control_mode}"
  --n_action_steps "${GRIFFIN_N_ACTION_STEPS:-10}"
)
if [[ "${GRIFFIN_RELATIVE_ACTIONS:-1}" == "0" ]]; then
  make_base_args+=(--no-use_relative_actions)
else
  make_base_args+=(--relative_exclude_joints "${exclude_joints[@]}")
fi
if [[ -n "${GRIFFIN_MAKE_BASE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${GRIFFIN_MAKE_BASE_EXTRA_ARGS}"
  make_base_args+=("${extra_args[@]}")
fi

echo "[Griffin_Alpha_S] head=${head} dataset=${dataset_repo_id} prompt='${embodiment_prompt}' control=${arm_control_mode}"
echo "[Griffin_Alpha_S] writing fine-tune base to ${base_dir}"
python "${ALPHA_S_ROOT}/scripts/make_finetune_base.py" "${make_base_args[@]}"

echo "[Griffin_Alpha_S] done. Next: bash train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_id>"
