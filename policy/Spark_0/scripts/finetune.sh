#!/bin/bash
set -euo pipefail

# Generic single-node (default 8-GPU) launcher for any model_configs/<name>.yaml.
# Dataset / embodiment / batch come from the YAML (data_mixture etc.); this
# wrapper only picks the recipe, GPU set, and checkpoint directory name.
#
# Usage:
#   bash scripts/finetune.sh <train_config>                # GPUs 0-7
#   bash scripts/finetune.sh <train_config> 0,1,2,3        # other GPU set
#   PER_DEVICE_BS=4 bash scripts/finetune.sh <train_config>
#   SEED=1 CKPT_NAME=my_run bash scripts/finetune.sh <train_config>
#
# Any further argument is forwarded as an OmegaConf override:
#   bash scripts/finetune.sh <train_config> 0,1 trainer.max_train_steps=20000
#
# Env knobs: GPUS, CKPT_NAME (default = train_config stem), SEED, BENCH_NAME,
# ENV_CFG_TYPE (default tianji_marvin_wuji)
# (default SparkArena), PER_DEVICE_BS, SKIP_DA3_CHECK, plus everything train.sh
# understands (SPARK_0_DATA_ROOT, SPARK_0_CKPT_ROOT, SPARK_0_WANDB_ENTITY,
# SPARK_0_ACCELERATE_CONFIG, multi-node SPARK_0_NUM_MACHINES/..., etc.).
# Checkpoints: checkpoints/<BENCH_NAME>-<CKPT_NAME>-<ENV_CFG_TYPE>-ee-<seed>/
#
# Keeping CKPT_NAME at its default (= the config stem) also makes eval resolve
# the model config by name; either way the checkpoint records the stem, so
# setup_eval_policy_server.sh finds the live model config (see its RTC block).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "${SCRIPT_DIR}")"

usage() {
    echo "Usage: bash scripts/finetune.sh <train_config> [gpus] [omega_overrides...]" >&2
    echo "  train_config  YAML stem under model_configs/ (or path to a .yaml)" >&2
    echo "  gpus          comma-separated ids (default: GPUS or 0,1,2,3,4,5,6,7)" >&2
    echo "Available configs:" >&2
    for f in "${ADAPTER_DIR}/model_configs/"*.yaml; do
        echo "  $(basename "${f%.yaml}")" >&2
    done
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

train_config="${1}"
shift

# Optional GPU list: next positional unless it looks like an override (a=b / --flag).
gpus="${GPUS:-0,1,2,3,4,5,6,7}"
if [[ $# -ge 1 && $1 != *=* && $1 != -* ]]; then
    gpus=$1
    shift
fi

# Strip a trailing .yaml / directory so CKPT_NAME defaults to the stem.
config_stem="$(basename "${train_config}")"
config_stem="${config_stem%.yaml}"
ckpt_name="${CKPT_NAME:-${config_stem}}"
seed="${SEED:-0}"
bench_name="${BENCH_NAME:-SparkArena}"
env_cfg_type="${ENV_CFG_TYPE:-tianji_marvin_wuji}"

# Resolve config early for a clear error (train.sh also checks).
if [[ -f "${train_config}" ]]; then
    config_yaml="$(realpath "${train_config}")"
else
    config_yaml="${ADAPTER_DIR}/model_configs/${config_stem}.yaml"
fi
if [[ ! -f "${config_yaml}" ]]; then
    echo "[finetune] ERROR: model config not found: ${config_yaml}" >&2
    usage
    exit 1
fi
# Prefer the stem so train.sh writes a stable .generated/<stem>_*.yaml name.
train_config="${config_stem}"

# Fail fast when a DA3-enabled recipe is missing teacher weights. The check
# reads spatial_alignment.enable from the YAML, not the recipe filename.
# train.sh repeats this on the generated yaml (the copy training actually
# loads); this early check fails before GPU / override setup.
bash "${SCRIPT_DIR}/check_da3_teacher.sh" "${config_yaml}"

extra_args=("$@")
if [[ -n "${PER_DEVICE_BS:-}" ]]; then
    extra_args+=("datasets.vla_data.per_device_batch_size=${PER_DEVICE_BS}")
fi

echo "[finetune] train_config=${train_config}"
echo "[finetune] ckpt_name=${ckpt_name}  bench=${bench_name}  seed=${seed}  gpus=${gpus}"
echo "[finetune] checkpoints -> ${ADAPTER_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-ee-${seed}/"

# Positional names only shape the run/checkpoint directory. Actual datasets
# come from the YAML data_mixture block when present.
exec bash "${ADAPTER_DIR}/train.sh" \
    "${bench_name}" "${ckpt_name}" "${env_cfg_type}" ee "${seed}" "${gpus}" \
    "${train_config}" ${extra_args[@]+"${extra_args[@]}"}
