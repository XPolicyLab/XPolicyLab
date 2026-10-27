#!/bin/bash
# Train Rex-M1-preview from the Xiaomi-Robotics-1-5B weights on a dataset
# converted by process_data.sh.
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [hydra_overrides...]
#
# The first six arguments follow the XPolicyLab convention; anything after them
# is forwarded verbatim to hydra, e.g. trainer.accumulate_grad_batches=2.
#
#   bench_name       RoboDojo (simulation)
#   ckpt_name        the same task selector passed to process_data.sh. It
#                    selects the converted dataset and the generated data
#                    config, and names the checkpoint directory
#   env_cfg_type     robot subdirectory the dataset was converted for
#   action_type      must be ee; the model emits end-effector deltas only
#   seed             training seed, also part of the checkpoint directory name
#   gpu_id           comma-separated local GPU ids, e.g. 0 or 0,1,2,3
#
# Every path is anchored on this script's own location, so the policy directory
# can be relocated freely. Override the defaults through the environment:
#   OUTPUT_DIR       converted dataset root (must match process_data.sh)
#   DATA_CONFIG_NAME data config basename under configs/data (must match too)
#   PRETRAINED_PATH  pretrained weights (default: checkpoints/pretrained_ckpt/model_states.pt)
#   RUN_ROOT         training output root (default: train_runs/<ckpt_setting>)
#   PROJECT          project name used in the artifact path (default: rex_m1)
#   MAX_STEPS        trainer.max_steps (default: 60000)
#   SAVE_INTERVAL    trainer.save_interval (default: 20000)
#   ASYNC_TRAIN      model.params.model.async_train (default: true)
#   MAX_LENGTH       per-sample token budget of the collate (default: 20000)
#   WANDB_MODE       wandb mode (default: offline; set to online to upload)
#   ALLOW_NON_EE_ACTION  set to 1 to train with action_type != ee anyway
#
# Multi-node: WORLD_SIZE, RANK, MASTER_ADDR, and MASTER_PORT are read by the
# vendored launcher; gpu_id sets the per-node GPU count. The reference run used
# 4 nodes x 8 GPUs at batch_size 8 (global batch 256).

set -euo pipefail

usage() {
    sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^#\{1\} \{0,1\}//'
}

if [[ "$#" -lt 6 ]]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
hydra_overrides=("$@")

if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[Rex_M1_preview] unsupported bench_name='${bench_name}'; expected 'RoboDojo'." >&2
    exit 1
fi

# The packed 60-dim action carries end-effector slots only (ACTION_PARTS in
# mibot/utils/io.py), so a joint-space run produces a checkpoint model.py
# rejects at eval startup. Fail here instead of after the training budget.
if [[ "${action_type}" != "ee" && "${ALLOW_NON_EE_ACTION:-0}" != "1" ]]; then
    echo "[Rex_M1_preview] action_type='${action_type}' cannot be deployed:" \
         "the model emits end-effector deltas only, and model.py accepts" \
         "'ee' alone at startup." >&2
    echo "[Rex_M1_preview] set ALLOW_NON_EE_ACTION=1 to train anyway." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REX_DIR="${POLICY_DIR}/rex_m1"

# Naming mirrors process_data.sh so a converted dataset is found without
# repeating any path: commas in a multi-task ckpt_name become underscores.
data_setting="${bench_name}-${ckpt_name//,/_}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
output_dir=${OUTPUT_DIR:-"${REX_DIR}/data/${data_setting}"}
default_config_name="$(echo "${bench_name}" | tr 'A-Z' 'a-z')_${ckpt_name//,/_}_${env_cfg_type}"
data_config_name=${DATA_CONFIG_NAME:-"${default_config_name}"}
data_config_path="${REX_DIR}/configs/data/${data_config_name}.yaml"

pretrained_path=${PRETRAINED_PATH:-"${POLICY_DIR}/checkpoints/pretrained_ckpt/model_states.pt"}
run_root=${RUN_ROOT:-"${POLICY_DIR}/train_runs/${ckpt_setting}"}
project=${PROJECT:-rex_m1}
# Where tools/train.py writes config.py and last.ckpt/, per mibot's
# process_save_cfg: <default_root_dir>/project_<project>/<exp_name>/.
artifact_dir="${run_root}/project_${project}/${ckpt_setting}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

if [[ ! -d "${output_dir}/data" ]]; then
    echo "[Rex_M1_preview] converted dataset not found: ${output_dir}/data" >&2
    echo "  run: bash process_data.sh ${bench_name} ${ckpt_name}" \
         "${env_cfg_type} ${action_type}" >&2
    exit 1
fi

if [[ ! -f "${data_config_path}" ]]; then
    echo "[Rex_M1_preview] data config not found: ${data_config_path}" >&2
    echo "  run process_data.sh first, or set DATA_CONFIG_NAME." >&2
    exit 1
fi

if [[ ! -f "${pretrained_path}" ]]; then
    echo "[Rex_M1_preview] pretrained weights not found: ${pretrained_path}" >&2
    echo "  download model_states.pt from Xiaomi-Robotics-1-5B" \
         "into checkpoints/pretrained_ckpt/, or set PRETRAINED_PATH." >&2
    exit 1
fi

# Expose the standard checkpoints/<ckpt_setting> path that model.py resolves
# from ckpt_name. The artifact dir is created up front so the link never dangles.
if [[ -e "${ckpt_dir}" && ! -L "${ckpt_dir}" ]]; then
    echo "[Rex_M1_preview] refusing to replace non-symlink ${ckpt_dir}" >&2
    echo "  move it away, or pick another ckpt_name/seed." >&2
    exit 1
fi
mkdir -p "${artifact_dir}" "${POLICY_DIR}/checkpoints"
ln -sfn "${artifact_dir}" "${ckpt_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export RESOURCE_GPU="${RESOURCE_GPU:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
# Offline by default: tools/train.py always builds a WandbLogger, and an
# unauthenticated online run would block on the interactive login prompt.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[Rex_M1_preview] dataset     : ${output_dir}/data"
echo "[Rex_M1_preview] data config : configs/data/${data_config_name}.yaml"
echo "[Rex_M1_preview] pretrained  : ${pretrained_path}"
echo "[Rex_M1_preview] run root    : ${run_root}"
echo "[Rex_M1_preview] checkpoint  : ${ckpt_dir} -> ${artifact_dir}"
echo "[Rex_M1_preview] gpus        : ${gpu_id}" \
     "(${RESOURCE_GPU} per node, ${WORLD_SIZE:-1} node(s))"
echo "[Rex_M1_preview] wandb mode  : ${WANDB_MODE}"

# Run from REX_DIR: the launcher puts it on PYTHONPATH and process_save_cfg
# also dumps the resolved config to ./assets/config.py.
cd "${REX_DIR}"

bash scripts/train.sh \
    "data=${data_config_name}" \
    "data.params.train_datasets.paths=[${output_dir}/data]" \
    "trainer.project=${project}" \
    "trainer.exp_name=${ckpt_setting}" \
    "trainer.default_root_dir=${run_root}" \
    "trainer.seed=${seed}" \
    "trainer.max_steps=${MAX_STEPS:-60000}" \
    "trainer.save_interval=${SAVE_INTERVAL:-20000}" \
    "model.params.pretrained=${pretrained_path}" \
    "model.params.model.async_train=${ASYNC_TRAIN:-true}" \
    "hydra.run.dir=${run_root}/hydra/rank_\${oc.env:RANK,0}" \
    ${hydra_overrides[@]+"${hydra_overrides[@]}"}

echo "[Rex_M1_preview] training finished. Evaluate with:"
echo "  bash eval.sh ${bench_name} <task_name> ${ckpt_setting} ${env_cfg_type}" \
     "${action_type} ${seed} 0 0 <policy_conda_env> <eval_env_conda_env>"
