#!/bin/bash
# Mem_0 artifact path helpers — exact 4-tuple naming (README §4.2), no fuzzy
# matching and no legacy fallbacks. Source from policy/Mem_0/*.sh.
#
# Dataset:  <policy>/data/<bench>-<ckpt>-<env_cfg>-<action_type>-lerobot
# Ckpt:     <policy>/checkpoints/<bench>-<ckpt>-<env_cfg>-<action_type>-<seed>
# Planning: <ckpt_dir>_planning_merged

mem0_dataset_tag() {
    echo "${1}-${2}-${3}-${4}"
}

mem0_ckpt_run_id() {
    echo "${1}-${2}-${3}-${4}-${5}"
}

mem0_dataset_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    echo "${policy_dir}/data/$(mem0_dataset_tag "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}")-lerobot"
}

mem0_ckpt_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    echo "${policy_dir}/checkpoints/$(mem0_ckpt_run_id "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")"
}

mem0_planning_merged_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    echo "$(mem0_ckpt_dir "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")_planning_merged"
}

mem0_norm_stats_path() {
    local policy_dir=$1 ckpt_name=$2
    echo "${policy_dir}/assets/${ckpt_name}/norm_stats.json"
}
