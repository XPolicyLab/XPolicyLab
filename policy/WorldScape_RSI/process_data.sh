#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  DATA_ROOT=/path/to/RoboDojo \
  bash process_data.sh RoboDojo <ckpt_name> arx_x5 joint [--overwrite]

RoboDojo reads official HDF5 episodes in place and generates zscore_stats.json,
q99_stats.json and a temporal .npz index. No images are decoded or rewritten.
Optional environment: ROBODOJO_PREPARED_DIR (default: data/<run-tag>),
ROBODOJO_MODEL (flash: H24/C1, memory: H24/C24; default flash),
ROBODOJO_PREPARE_WORKERS (4), ROBODOJO_Q99_MAX_SAMPLES (2000000; 0 = all frames),
ACTION_HORIZON (24), WAM_MAX_CHUNK_SIZE (model default), DATASET_STEP_STRIDE (1),
ROBODOJO_CHUNK_POLICY (uniform), SEED (62), WSP_PYTHON / CONDA_ENV.

EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

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
if [[ "${bench_name}" != "RoboDojo" ]]; then
    echo "[WorldScape_RSI-data] only bench_name=RoboDojo is supported" >&2
    exit 2
fi
if [[ "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
    echo "[WorldScape_RSI-data] RoboDojo preparation supports arx_x5 joint" >&2
    exit 1
fi
: "${DATA_ROOT:?Set DATA_ROOT to the official RoboDojo HDF5 dataset root}"
source "${worldscape_root}/recipes/common/robodojo_model.sh"
overwrite=()
for arg in "$@"; do
    if [[ "${arg}" == "--overwrite" ]]; then
        overwrite=(--overwrite)
    else
        echo "[WorldScape_RSI-data] unknown argument: ${arg}" >&2
        exit 1
    fi
done
python_bin="${WSP_PYTHON:-${CONDA_ENV:+${CONDA_ENV}/bin/python}}"
python_bin="${python_bin:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
python_bin="${python_bin:-python}"
export PYTHONPATH="${worldscape_root}/src:${worldscape_root}${PYTHONPATH:+:${PYTHONPATH}}"
# Read the same resolved geometry as training; no second set of shell defaults.
geometry="$("${python_bin}" - "${worldscape_root}/configs/posttrain/${robodojo_config_name}" <<'PY'
import sys
from worldscape_policy.cli.config_composition import load_composed_config
from worldscape_policy.cli.config_profiles import resolve_config_profiles

config = resolve_config_profiles(load_composed_config(sys.argv[1]))
dataset = config.data_loader.dataset_kwargs
print(dataset.action_horizon, dataset.max_chunk_size, dataset.step_stride,
      dataset.chunk_policy, dataset.seed)
PY
)"
read -r horizon chunks stride chunk_policy seed <<< "$geometry"
data_root="$(cd "${DATA_ROOT}" && pwd)"
prepared="${ROBODOJO_PREPARED_DIR:-${SCRIPT_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
mkdir -p "${prepared}"
prepared="$(cd "${prepared}" && pwd)"
workers="${ROBODOJO_PREPARE_WORKERS:-4}"
zscore="${prepared}/zscore_stats.json"
q99="${prepared}/q99_stats.json"
index="${prepared}/index_h${horizon}_c${chunks}_${chunk_policy}_stride${stride}.npz"
for output in "${zscore}" "${q99}" "${index}"; do
    if [[ -e "${output}" && ${#overwrite[@]} -eq 0 ]]; then
        echo "[WorldScape_RSI-data] ${output} already exists; use --overwrite" >&2
        exit 1
    fi
done
"${python_bin}" -m tools.data.compute_robodojo_zscore_stats \
    --data-root "${data_root}" --output "${zscore}" --workers "${workers}" "${overwrite[@]}"
"${python_bin}" -m tools.data.compute_robodojo_joint_stats \
    --data-root "${data_root}" --output "${q99}" --workers "${workers}" \
    --max-samples "${ROBODOJO_Q99_MAX_SAMPLES:-2000000}" --seed "${seed}" "${overwrite[@]}"
"${python_bin}" -m tools.data.build_robodojo_index \
    --data-root "${data_root}" --stats-path "${zscore}" --output "${index}" \
    --action-horizon "${horizon}" --max-chunk-size "${chunks}" \
    --chunk-policy "${chunk_policy}" --step-stride "${stride}" \
    --metadata-workers "${workers}" "${overwrite[@]}"
echo "[WorldScape_RSI-data] done. Set these variables before training:"
printf 'export ROBODOJO_MODEL=%q\n' "${ROBODOJO_MODEL}"
printf 'export ACTION_HORIZON=%q\n' "${horizon}"
printf 'export WAM_MAX_CHUNK_SIZE=%q\n' "${chunks}"
printf 'export ROBODOJO_CHUNK_POLICY=%q\n' "${chunk_policy}"
printf 'export DATASET_STEP_STRIDE=%q\n' "${stride}"
printf 'export DATA_ROOT=%q\n' "${data_root}"
printf 'export ROBODOJO_ZSCORE_STATS=%q\n' "${zscore}"
printf 'export ROBODOJO_JOINT_STATS=%q\n' "${zscore}"
printf 'export ROBODOJO_INDEX_PATH=%q\n' "${index}"
exit 0
