#!/bin/bash

# XPolicyLab README §4.2 artifact naming for LDA_1B (LeRobot v2.1, no -lerobot suffix).

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
        echo "${root}/${std_tag}"
    done
    if [[ -n "${expert_data_num}" ]]; then
        for root in "${roots[@]}"; do
            echo "${root}/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
        done
    fi
    for root in "${roots[@]}"; do
        echo "${root}/cotrain_dataset"
        echo "${root}/${bench_name}-${env_cfg_type}-${action_type}"
    done
}

xpolicylab_resolve_dataset_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local expert_data_num=${6:-}
    local candidate
    while IFS= read -r candidate; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(xpolicylab_dataset_dir_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${expert_data_num}")
    echo "${policy_dir}/data/$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
}

xpolicylab_ckpt_dir_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6 expert_data_num=${7:-}
    local std_id
    std_id="$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    echo "${policy_dir}/checkpoints/${std_id}"
    if [[ -n "${expert_data_num}" ]]; then
        echo "${policy_dir}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
    fi
    echo "${policy_dir}/runs/${bench_name}-${ckpt_name}-${env_cfg_type}-seed${seed}"
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

xpolicylab_resolve_checkpoint_pt() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6 expert_data_num=${7:-}
    local ckpt_dir checkpoints_subdir pt_path
    ckpt_dir="$(xpolicylab_resolve_ckpt_dir "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")"
    for checkpoints_subdir in "${ckpt_dir}/checkpoints" "${ckpt_dir}"; do
        [[ -d "${checkpoints_subdir}" ]] || continue
        pt_path=$(ls -1 "${checkpoints_subdir}"/steps_*_pytorch_model.pt 2>/dev/null \
            | awk -F'steps_|_pytorch_model.pt' '{printf "%s\t%012d\n", $0, $2}' \
            | sort -k2,2n | tail -n1 | cut -f1)
        if [[ -n "${pt_path}" && -f "${pt_path}" ]]; then
            echo "${pt_path}"
            return 0
        fi
    done
    return 1
}
