#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]
# expert_data_num: optional maximum episode count; empty = use all episodes
# raw_task_dirs: optional source task dir(s) under <workspace>/data/<bench_name>/, comma-separated
#
# Input follows the XPolicyLab layout: this adapter lives at
# <workspace>/XPolicyLab/policy/Spark_0 and trajectories at
# <workspace>/data/<bench_name>/<task>/<env_cfg_type>/data/*.hdf5, beside
# <workspace>/env_cfg/. Set XPL_DATA_ROOT to point at a different `data/`.
# Output: <this dir>/data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/
# (the LeRobot v3 dataset train.sh reads for the same four arguments).

if [[ $# -lt 4 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
    echo "Example: bash process_data.sh SparkArena sft_spark_0 tianji_marvin_wuji ee" >&2
    echo "Example: bash process_data.sh SparkArena sft_spark_0_50ep tianji_marvin_wuji ee 50 stack_bowls,pour_water" >&2
    exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}
raw_task_dirs=${6:-}
if [[ -n "${expert_data_num}" && ! "${expert_data_num}" =~ ^[0-9]+$ ]]; then
    raw_task_dirs="${expert_data_num}"
    expert_data_num=""
fi
if [[ "${action_type}" != "ee" ]]; then
    echo "[process_data] Spark-0 trains on EE poses + Wuji hand joints; action_type must be ee (got ${action_type})" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# <workspace>/XPolicyLab/policy/Spark_0 -> <workspace>
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
XPL_DATA_ROOT="${XPL_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
out_tag="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
data_root="${XPL_DATA_ROOT}/${bench_name}"
max_episode="${expert_data_num:-1000000000}"

if [[ -n "${raw_task_dirs}" ]]; then
    IFS=',' read -r -a tasks <<< "${raw_task_dirs}"
else
    tasks=("${ckpt_name}")
fi

for task in "${tasks[@]}"; do
    input_dir="${data_root}/${task}/${env_cfg_type}/data"
    if [[ ! -d "${input_dir}" ]]; then
        echo "[process_data] Missing input dir: ${input_dir}" >&2
        echo "[process_data] Expected <workspace>/data/<bench_name>/<task>/<env_cfg_type>/data/*.hdf5 with workspace=${WORKSPACE_ROOT}; set XPL_DATA_ROOT to override the data/ location." >&2
        exit 66
    fi
done

echo "[process_data] ${bench_name}/${ckpt_name}/${env_cfg_type} expert_data_num=${expert_data_num:-<all>} tasks=${tasks[*]} (${action_type})"
echo "[process_data] input ${data_root}/<task>/${env_cfg_type}/data -> ${SCRIPT_DIR}/data/${out_tag}/"

python "${SCRIPT_DIR}/tools/transform_lerobot_v30_format_tianji_ee.py" \
    --data_root "${data_root}" \
    --tasks "${tasks[@]}" \
    --env_cfg_type "${env_cfg_type}" \
    --repo_id "${out_tag}" \
    --output_root "${SCRIPT_DIR}/data" \
    --max_episode "${max_episode}"
