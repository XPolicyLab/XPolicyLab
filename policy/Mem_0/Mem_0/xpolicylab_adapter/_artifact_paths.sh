#!/bin/bash
# Mem_0 artifact path helpers (README §4.2 + legacy fallbacks).
# Source from policy/Mem_0/*.sh after setting POLICY_DIR.

mem0_dataset_tag() {
    echo "${1}-${2}-${3}-${4}"
}

mem0_ckpt_run_id() {
    echo "${1}-${2}-${3}-${4}-${5}"
}

mem0_legacy_run_id() {
    local bench_name=$1 ckpt_name=$2 env_cfg_type=$3 expert_data_num=$4 action_type=$5 seed=$6
    echo "${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-seed${seed}"
}

mem0_dataset_dir_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local expert_data_num=${6:-}
    local upstream_dir="${policy_dir}/Mem_0"
    local tag
    tag="$(mem0_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"

    # Standard (README §4.2)
    echo "${policy_dir}/data/${tag}-lerobot"
    if [[ -n "${expert_data_num}" ]]; then
        echo "${policy_dir}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-lerobot"
    fi

    # Legacy upstream layout
    if [[ -n "${expert_data_num}" ]]; then
        echo "${upstream_dir}/lerobot_datasets/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
        echo "${upstream_dir}/lerobot_datasets/${bench_name}-cotrain-${env_cfg_type}-${expert_data_num}-${action_type}"
    fi
    echo "${upstream_dir}/lerobot_datasets/${tag}"
}

mem0_resolve_dataset_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local expert_data_num=${6:-}
    local candidate
    while IFS= read -r candidate; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(mem0_dataset_dir_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${expert_data_num}")
    echo "${policy_dir}/data/$(mem0_dataset_tag "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}")-lerobot"
}

mem0_ckpt_dir_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local expert_data_num=${7:-}
    local upstream_dir="${policy_dir}/Mem_0"
    local run_id legacy_id
    run_id="$(mem0_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    legacy_id="$(mem0_legacy_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${expert_data_num:-0}" "${action_type}" "${seed}")"

    # Standard
    echo "${policy_dir}/checkpoints/${run_id}"
    if [[ -n "${expert_data_num}" ]]; then
        echo "${policy_dir}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
    fi

    # Legacy (upstream Mem_0/checkpoints, seed prefix)
    if [[ -n "${expert_data_num}" ]]; then
        echo "${upstream_dir}/checkpoints/${legacy_id}"
        echo "${upstream_dir}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-seed${seed}"
    fi
    echo "${upstream_dir}/checkpoints/${run_id}"
}

mem0_resolve_ckpt_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local expert_data_num=${7:-}
    local candidate
    for candidate in $(mem0_ckpt_dir_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}"); do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "${policy_dir}/checkpoints/$(mem0_ckpt_run_id "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")"
}

mem0_planning_merged_candidates() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local expert_data_num=${7:-}
    local upstream_dir="${policy_dir}/Mem_0"
    local run_id legacy_id
    run_id="$(mem0_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")"
    legacy_id="$(mem0_legacy_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${expert_data_num:-0}" "${action_type}" "${seed}")"

    echo "${policy_dir}/checkpoints/${run_id}_planning_merged"
    if [[ -n "${expert_data_num}" ]]; then
        echo "${policy_dir}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}_planning_merged"
    fi
    echo "${upstream_dir}/checkpoints/${legacy_id}_planning_merged"
    echo "${upstream_dir}/checkpoints/${run_id}_planning_merged"
}

mem0_resolve_planning_merged_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5 seed=$6
    local expert_data_num=${7:-}
    local candidate
    for candidate in $(mem0_planning_merged_candidates "${policy_dir}" "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}"); do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "${policy_dir}/checkpoints/$(mem0_ckpt_run_id "${bench_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")_planning_merged"
}

mem0_norm_stats_candidates() {
    local policy_dir=$1 ckpt_name=$2
    local upstream_dir="${policy_dir}/Mem_0"
    echo "${policy_dir}/assets/${ckpt_name}/norm_stats.json"
    echo "${upstream_dir}/assets/${ckpt_name}/norm_stats.json"
}

mem0_resolve_norm_stats_path() {
    local policy_dir=$1 ckpt_name=$2
    local candidate
    for candidate in $(mem0_norm_stats_candidates "${policy_dir}" "${ckpt_name}"); do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "${policy_dir}/assets/${ckpt_name}/norm_stats.json"
}

mem0_default_dataset_out_dir() {
    local policy_dir=$1 bench_name=$2 ckpt_name=$3 env_cfg_type=$4 action_type=$5
    local tag
    tag="$(mem0_dataset_tag "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}")"
    if [[ "${MEM0_LEGACY_PATHS:-}" == "1" ]]; then
        local expert_data_num=${6:-}
        echo "${policy_dir}/Mem_0/lerobot_datasets/${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
    else
        echo "${policy_dir}/data/${tag}-lerobot"
    fi
}

mem0_default_batch_dataset_out_dir() {
    local policy_dir=$1 dataset_id=$2
    if [[ "${MEM0_LEGACY_PATHS:-}" == "1" ]]; then
        echo "${policy_dir}/Mem_0/lerobot_datasets/${dataset_id}"
    else
        local ckpt_name="${dataset_id#*-}"
        if [[ "${dataset_id}" == *"-cotrain-"* ]]; then
            echo "${policy_dir}/data/${dataset_id}-lerobot"
        else
            echo "${policy_dir}/data/${dataset_id}-lerobot"
        fi
    fi
}
