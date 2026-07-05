#!/bin/bash

xpolicylab_dataset_tag() {
    echo "${1}-${2}-${3}-${4}"
}

xpolicylab_ckpt_run_id() {
    echo "${1}-${2}-${3}-${4}-${5}"
}

xpolicylab_dataset_dir_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local expert_data_num=${6:-}
    local std_tag
    std_tag="$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
    local roots=("${policy_dir}/data" "${policy_dir}/processed_data")
    local root
    for root in "${roots[@]}"; do
        echo "${root}/${std_tag}-lerobot"
    done
    if [[ -n "${expert_data_num}" ]]; then
        for root in "${roots[@]}"; do
            echo "${root}/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-lerobot"
        done
    fi
    for root in "${roots[@]}"; do
        echo "${root}/${bench_name}-${env_cfg_type}-${action_type}-lerobot"
    done
}

xpolicylab_resolve_dataset_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local expert_data_num=${6:-}
    local candidate
    while IFS= read -r candidate; do
        if [[ -d "${candidate}" || -L "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(xpolicylab_dataset_dir_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${expert_data_num}")
    echo "${policy_dir}/data/$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")-lerobot"
}

xpolicylab_ckpt_dir_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6 expert_data_num=${7:-}
    local std_id
    std_id="$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    echo "${policy_dir}/checkpoints/${std_id}"
    if [[ -n "${expert_data_num}" ]]; then
        echo "${policy_dir}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
    fi
    echo "${policy_dir}/checkpoints/${ckpt_name}"
}

xpolicylab_resolve_ckpt_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6 expert_data_num=${7:-}
    local candidate
    for candidate in $(xpolicylab_ckpt_dir_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}"); do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "${policy_dir}/checkpoints/$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
}
