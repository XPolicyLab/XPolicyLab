#!/usr/bin/env bash
set -euo pipefail

# bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
#
# Optional environment variables:
#   EWAM_DATA_DIR        process_data.sh output (default: data/<bench>-<ckpt>-<env>-<action>)
#   EWAM_PRETRAIN_PATH   EWAM pretrain checkpoint dir or .pt (default: checkpoints/pretrain)
#   EWAM_WAN_PATH        Wan2.2-TI2V-5B dir (default: checkpoints/Wan2.2-TI2V-5B)
#   EWAM_VLM_PATH        Qwen3-VL-2B-Instruct dir (default: checkpoints/Qwen3-VL-2B-Instruct)
#   EWAM_MAX_STEPS       default: 60000 (RoboTwin) / 100000 (RoboDojo), the released checkpoints
#   EWAM_REPORT_TO       tensorboard | wandb | all (default: tensorboard)
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT   multi-node torchrun
#   EWAM_DRY_RUN=1       print the launch command and exit
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
EWAM_DIR="${POLICY_DIR}/EWAM"

case "${bench_name}" in
  RoboTwin) config="${EWAM_DIR}/configs/ewam_robotwin.yaml"; default_steps=60000 ;;
  RoboDojo) config="${EWAM_DIR}/configs/ewam_robodojo.yaml"; default_steps=100000 ;;
  *) echo "[EWAM] unsupported bench_name: ${bench_name} (expected RoboTwin or RoboDojo)" >&2; exit 1 ;;
esac
if [[ "${action_type}" != "joint" ]]; then
  echo "[EWAM] only action_type=joint is supported, got ${action_type}" >&2
  exit 1
fi
action_dim=$(bash "${ROOT_DIR}/XPolicyLab/utils/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
if [[ "${action_dim}" != "14" ]]; then
  echo "[EWAM] EWAM expects a dual 6+1 arm (action dim 14), ${env_cfg_type} has ${action_dim}" >&2
  exit 1
fi

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
data_dir="${EWAM_DATA_DIR:-${POLICY_DIR}/data/${data_setting}}"
pretrain_path="${EWAM_PRETRAIN_PATH:-${POLICY_DIR}/checkpoints/pretrain}"
wan_path="${EWAM_WAN_PATH:-${POLICY_DIR}/checkpoints/Wan2.2-TI2V-5B}"
vlm_path="${EWAM_VLM_PATH:-${POLICY_DIR}/checkpoints/Qwen3-VL-2B-Instruct}"
max_steps="${EWAM_MAX_STEPS:-${default_steps}}"
nproc=$(awk -F',' '{print NF}' <<<"${gpu_id}")

for path in "${data_dir}" "${wan_path}" "${vlm_path}"; do
  [[ -e "${path}" ]] || { echo "[EWAM] not found: ${path}" >&2; exit 1; }
done
if [[ ! -f "${pretrain_path}" && ! -f "${pretrain_path}/mp_rank_00_model_states.pt" \
      && ! -f "${pretrain_path}/pytorch_model/mp_rank_00_model_states.pt" ]]; then
  echo "[EWAM] pretrain checkpoint not found: ${pretrain_path} (bash download_checkpoints.sh pretrain)" >&2
  exit 1
fi

cmd=(python -m torch.distributed.run
  --nnodes="${NNODES:-1}" --nproc_per_node="${nproc}" --node_rank="${NODE_RANK:-0}"
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29100}"
  train/train_ewam.py
  --deepspeed configs/zero1.json
  --config "${config}"
  --dataset_dir "${data_dir}"
  --wan_checkpoint_path "${wan_path}"
  --vlm_checkpoint_path "${vlm_path}"
  --finetune_checkpoint_path "${pretrain_path}"
  --checkpoint_dir "${ckpt_dir}"
  --run_name train
  --tensorboard_log_dir "${ckpt_dir}/tensorboard"
  --report_to "${EWAM_REPORT_TO:-tensorboard}"
  --max_steps "${max_steps}"
  --seed "${seed}")

echo "[EWAM] config=${config}"
echo "[EWAM] data_dir=${data_dir}"
echo "[EWAM] checkpoint_dir=${ckpt_dir}"
echo "[EWAM] gpus=${gpu_id} nnodes=${NNODES:-1} max_steps=${max_steps}"
if [[ "${EWAM_DRY_RUN:-0}" == "1" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${ROOT_DIR}:${EWAM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${EWAM_DIR}"
"${cmd[@]}"

# eval.sh resolves checkpoints/<ckpt_setting>/pytorch_model/; point it at the final step.
if [[ "${NODE_RANK:-0}" == "0" ]]; then
  run_dir="${ckpt_dir}/$(basename "${config}" .yaml)/train"
  last="" best=-1
  for d in "${run_dir}"/checkpoint_step_*; do
    n="${d##*_}"
    if [[ "${n}" =~ ^[0-9]+$ ]] && (( n > best )); then best=${n}; last=${d}; fi
  done
  if [[ -n "${last}" ]]; then
    ln -sfn "${last#"${ckpt_dir}/"}/pytorch_model" "${ckpt_dir}/pytorch_model"
    echo "[EWAM] eval checkpoint: ${ckpt_dir}/pytorch_model -> ${last}/pytorch_model"
  fi
fi
