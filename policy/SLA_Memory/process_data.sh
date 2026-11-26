#!/bin/bash
# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]
#
# Converts the official RoboDojo HDF5 data in <XPolicyLab parent>/data/<bench_name>/<task>/<robot>/data into the
# LeRobot datasets read by the training configs:
#   RoboDojo       all tasks of env_cfg_type in one dataset: Data/RoboDojo_sim_ee
#   RoboDojo_real  one dataset per task and robot: Data/RoboDojo_real_ee/RoboDojo_real_<task>_<robot>;
#                  env_cfg_type `all` converts every robot.
# ckpt_name is not used: all runs of a benchmark share these datasets.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
    exit 1
fi
bench_name=$1
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
PYTHON_BIN="${POLICY_DIR}/openpi/.venv/bin/python"
SRC_ROOT="${BENCH_ROOT}/data/${bench_name}"

if [[ "${action_type}" != "ee" ]]; then
    echo "[SLA_Memory][ERROR] SLA_Memory trains on end-effector actions; action_type must be ee, got ${action_type}" >&2
    exit 1
fi
if [[ ! -d "${SRC_ROOT}" ]]; then
    echo "[SLA_Memory][ERROR] ${SRC_ROOT} not found; download the HDF5 data with XPolicyLab's scripts/RoboDojo/download_robodojo_data.sh" >&2
    exit 1
fi
# A non-numeric 5th argument is the task list.
max_episodes=""
if [[ "${expert_data_num_or_raw_task_dirs}" =~ ^[0-9]+$ ]]; then
    max_episodes=${expert_data_num_or_raw_task_dirs}
elif [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
    raw_task_dirs=${expert_data_num_or_raw_task_dirs}
fi
if [[ -n "${raw_task_dirs}" ]]; then
    IFS=',' read -r -a tasks <<<"${raw_task_dirs}"
else
    mapfile -t tasks < <(ls "${SRC_ROOT}")
fi
if [[ "${env_cfg_type}" == "all" ]]; then
    mapfile -t robots < <(for task in "${tasks[@]}"; do ls "${SRC_ROOT}/${task}"; done | sort -u)
else
    robots=("${env_cfg_type}")
fi

# The converter reads the HDF5 files through XPolicyLab.utils.data_loader.
export PYTHONPATH="${BENCH_ROOT}"
export PYTHON_BIN SRC_ROOT MAX_EPISODES="${max_episodes}"
cd "${POLICY_DIR}"
case "${bench_name}" in
    RoboDojo)
        # Only tasks recorded on this robot.
        sim_tasks=()
        for task in "${tasks[@]}"; do
            if [[ -d "${SRC_ROOT}/${task}/${robots[0]}/data" ]]; then
                sim_tasks+=("${task}")
            fi
        done
        "${PYTHON_BIN}" openpi/scripts/convert_robodojo_eef.py "${SRC_ROOT}" "${robots[0]}" Data/RoboDojo_sim_ee \
            "${sim_tasks[@]}" ${MAX_EPISODES:+--max-episodes "${MAX_EPISODES}"}
        ;;
    RoboDojo_real)
        mkdir -p Data/RoboDojo_real_ee
        for task in "${tasks[@]}"; do
            for robot in "${robots[@]}"; do
                if [[ -d "${SRC_ROOT}/${task}/${robot}/data" ]]; then
                    echo "${task} ${robot}"
                fi
            done
        done | xargs -P "${SLA_MEMORY_CONVERT_JOBS:-4}" -L 1 bash -c '"${PYTHON_BIN}" openpi/scripts/convert_robodojo_eef.py \
            "${SRC_ROOT}" "$1" "Data/RoboDojo_real_ee/RoboDojo_real_$0_$1" "$0" ${MAX_EPISODES:+--max-episodes "${MAX_EPISODES}"}'
        ;;
    *)
        echo "[SLA_Memory][ERROR] bench_name must be RoboDojo or RoboDojo_real, got ${bench_name}" >&2
        exit 1
        ;;
esac
echo "[SLA_Memory] Data written to ${POLICY_DIR}/Data. train.sh computes the norm stats on its first run."
