#!/bin/bash

# Exact artifact naming (no fuzzy matching):
#   dataset:    data/<bench>-<ckpt>-<env>-<action>-lerobot
#   checkpoint: checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>
# At eval time ckpt_name is the full run directory name under checkpoints/,
# so checkpoints/<ckpt_name> is also accepted as an exact location.

xpolicylab_dataset_tag() {
    echo "${1}-${2}-${3}-${4}"
}

xpolicylab_ckpt_run_id() {
    echo "${1}-${2}-${3}-${4}-${5}"
}

xpolicylab_resolve_dataset_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local std_tag
    std_tag="$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
    echo "${policy_dir}/data/${std_tag}-lerobot"
}

xpolicylab_resolve_ckpt_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local std_dir
    std_dir="${policy_dir}/checkpoints/$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    if [[ -d "${std_dir}" ]]; then
        echo "${std_dir}"
        return 0
    fi
    if [[ -d "${policy_dir}/checkpoints/${ckpt_name}" ]]; then
        echo "${policy_dir}/checkpoints/${ckpt_name}"
        return 0
    fi
    echo "${std_dir}"
}
