#!/bin/bash

# XPolicyLab README §4.2 artifact naming for LDA_1B (LeRobot v2.1, no -lerobot suffix).
# Exact naming only (no fuzzy matching):
#   dataset:    data/<bench>-<ckpt>-<env>-<action>
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
    echo "${policy_dir}/data/$(xpolicylab_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
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

xpolicylab_checkpoint_run_dir() {
    local pt_path=$1 pt_dir
    pt_dir="$(dirname "${pt_path}")"
    if [[ "$(basename "${pt_dir}")" == "checkpoints" ]]; then
        dirname "${pt_dir}"
    else
        echo "${pt_dir}"
    fi
}

xpolicylab_is_loadable_checkpoint() {
    local pt_path=$1 run_dir
    [[ -f "${pt_path}" ]] || return 1
    run_dir="$(xpolicylab_checkpoint_run_dir "${pt_path}")"
    [[ -f "${run_dir}/config.yaml" && -f "${run_dir}/dataset_statistics.json" ]]
}

xpolicylab_resolve_checkpoint_pt() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local ckpt_dir checkpoints_subdir pt_path
    ckpt_dir="$(xpolicylab_resolve_ckpt_dir "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")"
    for checkpoints_subdir in "${ckpt_dir}/checkpoints" "${ckpt_dir}"; do
        [[ -d "${checkpoints_subdir}" ]] || continue
        pt_path=$(ls -1 "${checkpoints_subdir}"/steps_*_pytorch_model.pt 2>/dev/null \
            | awk -F'steps_|_pytorch_model.pt' '{printf "%s\t%012d\n", $0, $2}' \
            | sort -k2,2n | tail -n1 | cut -f1)
        if [[ -z "${pt_path}" || ! -f "${pt_path}" ]]; then
            continue
        fi
        if xpolicylab_is_loadable_checkpoint "${pt_path}"; then
            echo "${pt_path}"
            return 0
        fi
    done
    return 1
}
