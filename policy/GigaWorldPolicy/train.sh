#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5
seed=$6
gpu_id=$7

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_DIR="${POLICY_DIR}/giga_world_policy"
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
ckpt_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
data_dir="${GIGAWORLD_DATA_DIR:-${POLICY_DIR}/data/${data_setting}}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
config_path="${ckpt_dir}/xpolicylab_train_config.json"
base_config="${GIGAWORLD_BASE_CONFIG:-${INNER_DIR}/config.json}"
norm_path="${GIGAWORLD_NORM_PATH:-${INNER_DIR}/norm_stats_delta.json}"
pretrained_path="${GIGAWORLD_PRETRAINED_PATH:-/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B-Diffusers/}"

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

python - "${base_config}" "${config_path}" "${ckpt_dir}" "${data_dir}" "${norm_path}" "${pretrained_path}" "${gpu_id}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

base_config, output_config, ckpt_dir, data_dir, norm_path, pretrained_path, gpu_id, seed = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    config = json.load(f)

gpu_ids = [int(item) for item in gpu_id.split(",") if item.strip() != ""]
if not gpu_ids:
    raise ValueError("gpu_id must contain at least one GPU id")

config["project_dir"] = ckpt_dir
config.setdefault("launch", {})["gpu_ids"] = gpu_ids

train_loader = config.setdefault("dataloaders", {}).setdefault("train", {})
data_or_config = train_loader.setdefault("data_or_config", [{}])
if not data_or_config:
    data_or_config.append({})
data_or_config[0]["data_path"] = data_dir

transform = train_loader.setdefault("transform", {})
transform["norm_path"] = [norm_path]

models = config.setdefault("models", {})
models["pretrained"] = pretrained_path
models["view_dir"] = ckpt_dir

train = config.setdefault("train", {})
# giga-train asserts seed > 0, while XPolicyLab commonly uses seed 0.
train["seed"] = max(1, int(seed))

Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
PY

echo "[GigaWorldPolicy] data_dir=${data_dir}"
echo "[GigaWorldPolicy] checkpoint_dir=${ckpt_dir}"
echo "[GigaWorldPolicy] config=${config_path}"

if [[ "${GIGAWORLD_DRY_RUN:-0}" == "1" ]]; then
  echo "[GigaWorldPolicy] dry run enabled, skip training launch"
  exit 0
fi

cd "${INNER_DIR}"
python -m scripts.train --config "${config_path}"
