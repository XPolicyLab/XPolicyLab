#!/bin/bash


# * usage: ./train.sh CONFIG_NAME NGPUS_PER_NODE

config_name=${1}
ngpus_per_node=${2}
PY_ARGS=${@:3}

RISE_CONDA_ENV="${RISE_CONDA_ENV:-/mnt/nfs/miniconda3/envs/RISE}"
RISE_TORCHRUN="${RISE_TORCHRUN:-${RISE_CONDA_ENV}/bin/torchrun}"

# cd to the directory of the script
cd $(dirname $(realpath $0))

export PATH="${RISE_CONDA_ENV}/bin:${PATH}"
export WANDB_MODE=offline
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

if [[ "$PY_ARGS" == *"--resume"* ]]; then
  echo "Resuming training..."
  "${RISE_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name $PY_ARGS
else
  echo "Overwriting training..."
  "${RISE_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name --overwrite $PY_ARGS
fi
