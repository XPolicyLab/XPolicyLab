#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}" # path to the openvla-oft repo
MODEL_DIR="${MODEL_DIR:-/mnt/xspark-data/xspark_shared/model_weights/openvla-7b}" # path to the pretrained OpenVLA model.
DATA_ROOT="${DATA_ROOT:-/mnt/xspark-data/xspark_shared/tensorflow_datasets}" # path to the TFDS datasets.
export HF_HOME="${HF_HOME:-/mnt/xspark-data/xspark_shared/model_weights/.cache/huggingface}" # Hugging Face cache directory.
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

RUN_ROOT=${1} # path to the directory where the finetuning logs and checkpoints will be saved. Change this to your desired path.
DATASET_NAME=${2} # the name of the dataset to finetune on. This should be the same as the dataset_name in build_tfds_aloha.sh. Change this if you are using a different dataset.
GPU_ID=${3} # the GPU ids to use for finetuning. This should be a comma-separated list of GPU ids. For example, "0,1,2,3" to use GPUs with ids 0, 1, 2, and 3. Change this according to your setup.
SEED=${4:-0}

if [[ "${DATASET_NAME,,}" == *"aloha"* || "${DATASET_NAME,,}" == *"robotwin"* ]]; then
  export OPENVLA_ROBOT_PLATFORM=${OPENVLA_ROBOT_PLATFORM:-ALOHA}
fi
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export WANDB_MODE=${WANDB_MODE:-offline}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NUM_GPUS=$(echo "${GPU_ID}" | tr "," "\n" | sed "/^$/d" | wc -l | tr -d " ")
MIN_FREE_MEM_MIB=${MIN_FREE_MEM_MIB:-20000}
GPU_FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
for gpu in $(echo "${GPU_ID}" | tr "," " "); do
  free_mem=$(echo "${GPU_FREE_MEM}" | sed -n "$((gpu + 1))p" | tr -d " ")
  if [ -z "${free_mem}" ]; then
    echo "[finetune.sh] Failed to query free memory for GPU ${gpu}." >&2
    exit 1
  fi
  if [ "${free_mem}" -lt "${MIN_FREE_MEM_MIB}" ]; then
    echo "[finetune.sh] GPU ${gpu} has only ${free_mem} MiB free, below required ${MIN_FREE_MEM_MIB} MiB." >&2
    echo "[finetune.sh] Choose different GPUs or wait for the current jobs to finish." >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}"
mkdir -p "${HF_HOME}"

cd "${REPO_ROOT}"

torchrun --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}" vla-scripts/finetune.py \
  --vla_path "${MODEL_DIR}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT}" \
  --seed "${SEED}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project "openvla-oft-local" \
  --run_id_note "${NUM_GPUS}gpu--cuda${GPU_ID//,/}--3img--proprio--film" # you can change the wandb_project and run_id_note according to your preference for logging the finetuning runs in Weights & Biases.
