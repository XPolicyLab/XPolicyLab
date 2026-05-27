#!/bin/bash
set -euo pipefail

policy_name=LDA_1B
dataset_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
expert_data_num=${5}
action_type=${6}
seed=${7}
policy_gpu_id=${8}
env_gpu_id=${9}
policy_conda_env=${10}
eval_env_conda_env=${11}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/${policy_name}"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${POLICY_DIR}/deploy.yml"

# Default checkpoint: latest steps_*_pytorch_model.pt under the conventional
# train.sh ckpt dir. task_name is the simulator task; ckpt_name selects the
# checkpoint subdir and can differ (for example, cotrain).
# Override with `LDA_CHECKPOINT_PATH=...` to point at any other .pt file. The
# `lda.model.framework.base_framework.from_pretrained` loader also requires
# `config.yaml` and `dataset_statistics.json` in the checkpoint's parent's-parent
# (`<ckpt_setting_dir>/{config.yaml,dataset_statistics.json}`), which `train.sh`
# writes automatically.
# Project convention (README): trained checkpoints live under
# `<policy>/checkpoints/<ckpt_setting>/`. ckpt_setting follows DP's 6-tuple
# form `<dataset>-<ckpt_name>-<env_cfg>-<expert_data_num>-<action_type>-<seed>`.
# The legacy `<policy>/runs/<dataset>-<ckpt_name>-<env_cfg>-seed<seed>/` layout
# is still probed as a fallback so in-flight training there stays evaluable
# without manual `LDA_CKPT_ROOT` / `LDA_CHECKPOINT_PATH` overrides.
default_ckpt_setting="${LDA_CKPT_SETTING:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"
default_ckpt_root="${LDA_CKPT_ROOT:-${POLICY_DIR}/checkpoints}"
default_ckpt_dir="${default_ckpt_root}/${default_ckpt_setting}/checkpoints"
legacy_ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-seed${seed}"
legacy_ckpt_dir="${POLICY_DIR}/runs/${legacy_ckpt_setting}/checkpoints"
if [[ -z "${LDA_CHECKPOINT_PATH:-}" ]]; then
    for candidate_dir in "${default_ckpt_dir}" "${legacy_ckpt_dir}"; do
        [[ -d "${candidate_dir}" ]] || continue
        LDA_CHECKPOINT_PATH=$(ls -1 "${candidate_dir}"/steps_*_pytorch_model.pt 2>/dev/null \
            | awk -F'steps_|_pytorch_model.pt' '{printf "%s\t%012d\n", $0, $2}' \
            | sort -k2,2n | tail -n1 | cut -f1)
        [[ -n "${LDA_CHECKPOINT_PATH}" ]] && break
    done
fi
if [[ -z "${LDA_CHECKPOINT_PATH:-}" || ! -f "${LDA_CHECKPOINT_PATH}" ]]; then
    echo -e "\033[31m[ERROR] LDA_CHECKPOINT_PATH is empty or missing: '${LDA_CHECKPOINT_PATH:-}'.\033[0m" >&2
    echo -e "\033[31m        Looked under: ${default_ckpt_dir}/steps_*_pytorch_model.pt\033[0m" >&2
    echo -e "\033[31m                  and: ${legacy_ckpt_dir}/steps_*_pytorch_model.pt\033[0m" >&2
    echo -e "\033[31m        Set LDA_CHECKPOINT_PATH=... or LDA_CKPT_ROOT=... explicitly.\033[0m" >&2
    exit 1
fi
echo -e "\033[33m[INFO] Using checkpoint: ${LDA_CHECKPOINT_PATH}\033[0m"
echo -e "\033[33m[INFO] task_name: ${task_name}; ckpt_name: ${ckpt_name}\033[0m"

FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
cd "${POLICY_DIR}/LDA-1B"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        ckpt_name="${ckpt_name}" \
        checkpoint_path="${LDA_CHECKPOINT_PATH}" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
CUDA_VISIBLE_DEVICES="${env_gpu_id}" bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" "${FREE_PORT}" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" \
    "${additional_info}" "${ROOT_DIR}" "${seed}" "${env_gpu_id}"
echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"
