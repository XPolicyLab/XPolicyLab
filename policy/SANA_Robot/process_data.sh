#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Output: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/{sana_robot_zip/, normalization.json}
#   SANA_ROBOT_ROOT          the checkout of the SANA repository (default: Sana/ in this directory)
#   SANA_ROBOT_LEROBOT_DIR   the official LeRobot v3.0 export (default: <workspace>/data/RoboDojo_lerobot_v30_video)
#   SANA_ROBOT_LATENTS=1     also encode the video latents once, so that training does not run the video VAE
#   SANA_ROBOT_VAE           the LTX-2 video VAE directory, for the latents
if [[ $# -lt 4 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
    exit 1
fi
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SANA_DIR="${SANA_ROBOT_ROOT:-${SCRIPT_DIR}/Sana}"
source_dir="${SANA_ROBOT_LEROBOT_DIR:-${BENCH_ROOT}/data/RoboDojo_lerobot_v30_video}"
output_dir="${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ "${env_cfg_type}" != "arx_x5" ]]; then
    echo "[SANA_Robot] the converter is written for the RoboDojo ARX-X5 robot (env_cfg_type arx_x5), got ${env_cfg_type}" >&2
    exit 1
fi
if [[ ! -f "${source_dir}/meta/info.json" ]]; then
    echo "[SANA_Robot] no LeRobot v3.0 dataset at ${source_dir}; download it with" >&2
    echo "  bash scripts/RoboDojo/download_robodojo_data.sh lerobot_v3.0" >&2
    echo "or point SANA_ROBOT_LEROBOT_DIR at it" >&2
    exit 1
fi

limit=()
if [[ -n "${expert_data_num}" ]]; then
    limit=(--episode-count "${expert_data_num}")
fi

cd "${SANA_DIR}"
bash scripts/data/convert_robodojo.sh "${source_dir}" "${output_dir}" ${limit[@]+"${limit[@]}"}
if [[ "${SANA_ROBOT_LATENTS:-0}" == "1" ]]; then
    bash scripts/data/build_robodojo_latents.sh "${output_dir}" "${SANA_ROBOT_VAE:-checkpoints/LTX-2.3-Diffusers}"
fi

echo "[SANA_Robot] dataset written to ${output_dir}"
