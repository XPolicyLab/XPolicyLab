#!/usr/bin/env bash
set -euo pipefail

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
XR0_ROOT="${POLICY_DIR}/xiaomi_robotics_0/xr0"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
converted_data_root="${XR0_CONVERTED_DATA_ROOT:-${POLICY_DIR}/data/${data_setting}}"
data_config_name="${XR0_DATA_CONFIG_NAME:-${data_setting}}"
data_config_path="${XR0_ROOT}/configs/data/${data_config_name}.yaml"
# XR0 writes artifacts to <default_root_dir>/project_xr0/<exp_name>/; train into a
# run dir and expose the standard checkpoints/<ckpt_setting> path via symlink below.
run_root="${POLICY_DIR}/train_runs/${ckpt_setting}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
pretrained_path="${XR0_PRETRAINED_PATH:-${XR0_ROOT}/pretrained_ckpt/xr0_pretrained.pt}"

if [[ ! -d "${converted_data_root}/json" ]]; then
  echo "Converted dataset not found: ${converted_data_root}/json" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${data_config_path}" ]]; then
  echo "Data config not found: ${data_config_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${pretrained_path}" ]]; then
  echo "Pretrained checkpoint not found: ${pretrained_path}" >&2
  echo "Download Xiaomi-Robotics-0-Pretrain and run weight_convert.py, or set XR0_PRETRAINED_PATH." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export RESOURCE_GPU="${RESOURCE_GPU:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${run_root}" "${POLICY_DIR}/checkpoints"

echo "[Xiaomi_Robotics_0] converted_data_root=${converted_data_root}"
echo "[Xiaomi_Robotics_0] data_config=${data_config_path}"
echo "[Xiaomi_Robotics_0] pretrained_path=${pretrained_path}"
echo "[Xiaomi_Robotics_0] run_root=${run_root}"
echo "[Xiaomi_Robotics_0] checkpoint_dir=${ckpt_dir}"
echo "[Xiaomi_Robotics_0] seed=${seed}"
echo "[Xiaomi_Robotics_0] gpu_id=${gpu_id}"
echo "[Xiaomi_Robotics_0] resource_gpu=${RESOURCE_GPU}"

# XR0 artifact dir (config.py + last.ckpt) as written by mibot process_save_cfg.
xr0_artifact_dir="${run_root}/project_xr0/${ckpt_setting}"
if [[ -e "${ckpt_dir}" && ! -L "${ckpt_dir}" ]]; then
  echo "Refusing to overwrite existing non-symlink checkpoint dir: ${ckpt_dir}" >&2
  echo "Move it away or pick another ckpt_name/seed." >&2
  exit 1
fi
ln -sfn "${xr0_artifact_dir}" "${ckpt_dir}"
echo "[Xiaomi_Robotics_0] linked ${ckpt_dir} -> ${xr0_artifact_dir}"

cd "${XR0_ROOT}"

bash scripts/train.sh \
  "data=${data_config_name}" \
  "trainer.project=xr0" \
  "trainer.exp_name=${ckpt_setting}" \
  "trainer.default_root_dir=${run_root}" \
  "trainer.seed=${seed}" \
  "model.params.model.pretrained=${pretrained_path}" \
  "model.params.model.async_train=${XR0_ASYNC_TRAIN:-false}" \
  "trainer.max_steps=${XR0_MAX_STEPS:-30000}" \
  "trainer.save_interval=${XR0_SAVE_INTERVAL:-5000}"
