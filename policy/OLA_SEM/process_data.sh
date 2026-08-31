#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_name=${1:?Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [dataset_root]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
source_root=${5:-${OLA_SEM_DATASET_ROOT:-}}

if [[ "${bench_name}" != "RoboTwin" ]]; then
    echo "[ERROR] OLA-SEM data integration supports bench_name=RoboTwin; got ${bench_name}." >&2
    exit 1
fi
if [[ "${env_cfg_type}" != "aloha_agilex" || "${action_type}" != "joint" ]]; then
    echo "[ERROR] OLA-SEM supports env_cfg_type=aloha_agilex and action_type=joint." >&2
    exit 1
fi
if [[ -z "${source_root}" ]]; then
    echo "[ERROR] Pass the converted dataset root as argument 5 or set OLA_SEM_DATASET_ROOT." >&2
    echo "[ERROR] Preparation guide: ${SCRIPT_DIR}/README.md#data-processing" >&2
    exit 1
fi
if [[ ! -d "${source_root}/clean" || ! -d "${source_root}/randomized" ]]; then
    echo "[ERROR] Expected clean/ and randomized/ below ${source_root}." >&2
    exit 1
fi

for required in qpos videos umt5_wan language_action; do
    if ! find "${source_root}/clean" -mindepth 2 -maxdepth 2 -type d -name "${required}" -print -quit | grep -q .; then
        echo "[ERROR] No clean task contains required ${required}/ data under ${source_root}." >&2
        exit 1
    fi
done

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
target="${SCRIPT_DIR}/data/${data_setting}"
mkdir -p "${SCRIPT_DIR}/data"
if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[ERROR] Refusing to replace non-symlink dataset path: ${target}" >&2
    exit 1
fi
if [[ -L "${target}" ]]; then
    unlink "${target}"
fi
ln -s "$(realpath "${source_root}")" "${target}"

echo "[OLA_SEM] prepared ${target} -> $(realpath "${source_root}")"
