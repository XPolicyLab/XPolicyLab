#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  LEROBOT_DATA_PATH=/path/to/robotwin2_lerobot \
  bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

WorldScape Policy 2.0 trains from RoboTwin 2.0 LeRobot v2 data through
worldscape-policy/src/worldscape_policy/data/adapters/lerobot.py. This script
does not rewrite episodes; it
  1. links LEROBOT_DATA_PATH to data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
     (the DATA_ROOT that train.sh expects), and
  2. writes the WorldScape native metadata (meta/modality.json, meta/embodiment.json,
     meta/stats.json, ...) with
     worldscape-policy/tools/data/convert_lerobot_to_native_meta.py --embodiment robotwin2
     for the dataset (or for every child dataset that has meta/info.json).
Extra arguments are forwarded to convert_lerobot_to_native_meta.py
(e.g. --state-keys/--action-keys/--task-key/--force; see worldscape-policy/tools/data/README.md).

train.sh additionally needs ZSCORE_STATS_PATH: a JSON file with 14-dim global
z-score statistics of the whole DATA_ROOT in the form
  {"state": {"default": {"global_mean": [...], "global_std": [...]}},
   "action": {"default": {"global_mean": [...], "global_std": [...]}}}
(see worldscape-policy/src/worldscape_policy/data/normalization.py).

Environment:
  LEROBOT_DATA_PATH       (required) RoboTwin 2.0 LeRobot v2 dataset root
  WORLDSCAPE_POLICY_ROOT  WorldScape Policy source (default: ./worldscape-policy)
  ACTION_HORIZON          24 (default) or 48
EOF
}

if [ "$#" -lt 4 ]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
worldscape_root="${WORLDSCAPE_POLICY_ROOT:-${SCRIPT_DIR}/worldscape-policy}"
converter="${worldscape_root}/tools/data/convert_lerobot_to_native_meta.py"
if [ ! -f "${converter}" ]; then
    echo "[WorldScape_Policy_2-data] converter not found: ${converter}" >&2
    exit 1
fi
if [ "${action_type}" != "joint" ]; then
    echo "[WorldScape_Policy_2-data] only action_type=joint is supported (got ${action_type})" >&2
    exit 1
fi
if [ -z "${LEROBOT_DATA_PATH:-}" ]; then
    usage >&2
    echo "[WorldScape_Policy_2-data] set LEROBOT_DATA_PATH to a RoboTwin 2.0 LeRobot v2 dataset root" >&2
    exit 1
fi
src="$(cd "${LEROBOT_DATA_PATH}" && pwd)"

# Discover datasets: either the root itself or its children carrying meta/info.json.
datasets=()
if [ -f "${src}/meta/info.json" ]; then
    datasets+=("${src}")
else
    for child in "${src}"/*/; do
        [ -f "${child}meta/info.json" ] && datasets+=("${child%/}")
    done
fi
if [ "${#datasets[@]}" -eq 0 ]; then
    echo "[WorldScape_Policy_2-data] no LeRobot dataset (meta/info.json) under ${src}" >&2
    exit 1
fi

data_tag="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
link="${SCRIPT_DIR}/data/${data_tag}"
mkdir -p "${SCRIPT_DIR}/data"
if [ -L "${link}" ] || [ -e "${link}" ]; then
    echo "[WorldScape_Policy_2-data] ${link} already exists; leaving it in place"
else
    ln -s "${src}" "${link}"
    echo "[WorldScape_Policy_2-data] linked ${link} -> ${src}"
fi

cd "${worldscape_root}"
export PYTHONPATH="${worldscape_root}/src:${worldscape_root}${PYTHONPATH:+:${PYTHONPATH}}"
for ds in "${datasets[@]}"; do
    echo "[WorldScape_Policy_2-data] native metadata for ${ds}"
    python "${converter}" \
        --dataset-path "${ds}" \
        --embodiment robotwin2 \
        --action-horizon "${ACTION_HORIZON:-24}" \
        "$@"
done

echo "[WorldScape_Policy_2-data] done. Train with:"
echo "  DATA_ROOT=${link} ZSCORE_STATS_PATH=<dataset_stats.json> PRETRAINED_MODEL_PATH=<wsp_2_pretrain> \\"
echo "  bash train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} <seed> <gpu_id>"
