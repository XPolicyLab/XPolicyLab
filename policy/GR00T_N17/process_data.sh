#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>" >&2
  exit 1
fi

dataset_name=$1
ckpt_name=$2
env_cfg_type=$3
expert_data_num=$4
action_type=$5

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
DATA_ROOT="${GR00T_LEROBOT_HOME:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set GR00T_LEROBOT_HOME to the LeRobot datasets root." >&2
  exit 1
fi
data_setting="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
dataset_path="${DATA_ROOT}/${data_setting}"
modality_config="${POLICY_DIR}/configs/${env_cfg_type}_config.py"

resolve_src_dataset() {
  if [[ -n "${GR00T_SRC_DATASET:-}" ]]; then
    echo "${GR00T_SRC_DATASET}"
    return
  fi
  case "${env_cfg_type}" in
    arx_x5)
      echo "RoboDojo_sim_arx-x5_v30"
      ;;
    *)
      echo "Unsupported env_cfg_type: ${env_cfg_type}. Set GR00T_SRC_DATASET explicitly." >&2
      exit 1
      ;;
  esac
}

write_modality_json() {
  case "${env_cfg_type}" in
    arx_x5)
      cat > "${dataset_path}/meta/modality.json" <<'EOF'
{
  "state": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "action": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "video": {
    "front": { "original_key": "observation.images.cam_high" },
    "left_wrist": { "original_key": "observation.images.cam_left_wrist" },
    "right_wrist": { "original_key": "observation.images.cam_right_wrist" }
  },
  "annotation": {
    "human.task_description": { "original_key": "task_index" }
  }
}
EOF
      ;;
    *)
      echo "No modality.json template for env_cfg_type=${env_cfg_type}" >&2
      exit 1
      ;;
  esac
}

write_modality_config() {
  mkdir -p "${POLICY_DIR}/configs"
  case "${env_cfg_type}" in
    arx_x5)
      cat > "${modality_config}" <<'PY'
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

robodojo_arx_x5_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    robodojo_arx_x5_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
PY
      ;;
    *)
      echo "No modality config template for env_cfg_type=${env_cfg_type}" >&2
      exit 1
      ;;
  esac
}

install_conversion_deps() {
  if python -c "import lerobot" >/dev/null 2>&1; then
    return
  fi
  echo "[GR00T_N17] Installing lerobot conversion dependencies into gr00t .venv..."
  GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python \
    "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
}

src_dataset="$(resolve_src_dataset)"
src_path="${DATA_ROOT}/${src_dataset}"

echo "[GR00T_N17] dataset_name=${dataset_name}"
echo "[GR00T_N17] ckpt_name=${ckpt_name}"
echo "[GR00T_N17] env_cfg_type=${env_cfg_type}"
echo "[GR00T_N17] expert_data_num=${expert_data_num}"
echo "[GR00T_N17] action_type=${action_type}"
echo "[GR00T_N17] src_dataset=${src_path}"
echo "[GR00T_N17] output_dataset=${dataset_path}"

cd "${GR00T_ROOT}"
source .venv/bin/activate

if [[ ! -d "${src_path}" ]]; then
  echo "Source dataset not found: ${src_path}" >&2
  exit 1
fi

if [[ ! -d "${dataset_path}" ]]; then
  echo "[GR00T_N17] Copy source dataset..."
  cp -a "${src_path}" "${dataset_path}"
fi

codebase_version="$(python -c "import json; print(json.load(open('${dataset_path}/meta/info.json'))['codebase_version'])")"
echo "[GR00T_N17] current codebase_version=${codebase_version}"

if [[ "${codebase_version}" == "v3.0" ]]; then
  echo "[GR00T_N17] Converting LeRobot v3.0 -> v2.1..."
  install_conversion_deps
  uv run --no-sync python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --root "${DATA_ROOT}" \
    --repo-id "${data_setting}"
  codebase_version="$(python -c "import json; print(json.load(open('${dataset_path}/meta/info.json'))['codebase_version'])")"
fi

if [[ "${codebase_version}" != "v2.1" ]]; then
  echo "Expected codebase_version=v2.1 after conversion, got ${codebase_version}" >&2
  exit 1
fi

echo "[GR00T_N17] Writing modality.json and modality config..."
write_modality_json
write_modality_config

echo "[GR00T_N17] Generating dataset stats..."
uv run --no-sync python gr00t/data/stats.py \
  --dataset-path "${dataset_path}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${modality_config}"

echo "[GR00T_N17] process_data done."
echo "[GR00T_N17] dataset_path=${dataset_path}"
echo "[GR00T_N17] modality_config=${modality_config}"
