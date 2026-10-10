#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || ${1:-} == "--help" ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id[,gpu_id...]> [post-train options]"
    echo "Required: FOCUS_VLWA_INIT_CHECKPOINT. Optional: FOCUS_VLWA_STAGE=joint|frozen, FOCUS_VLWA_DATASET, FOCUS_VLWA_PYTHON, FOCUS_VLWA_DRY_RUN=1."
    [[ ${1:-} == "--help" ]] && exit 0
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_ids=$6
shift 6
for argument in "$@"; do
    case "${argument}" in
        --dataset|--dataset=*|--output-dir|--output-dir=*|--seed|--seed=*|--stage|--stage=*|--init-checkpoint|--init-checkpoint=*|--norm-stats-dir|--norm-stats-dir=*)
            echo "[TRAIN][ERROR] Configure ${argument} through standard arguments or FOCUS_VLWA environment variables." >&2
            exit 1
            ;;
    esac
done
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
if [[ ${env_cfg_type} != "arx_x5" || ${action_type} != "joint" ]]; then
    echo "[TRAIN][ERROR] ME_Brain_1 supports arx_x5 joint training only." >&2
    exit 1
fi
if [[ ! ${seed} =~ ^[0-9]+$ || ! ${gpu_ids} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[TRAIN][ERROR] seed must be nonnegative and gpu_id must be a comma-separated list of GPU indices." >&2
    exit 1
fi
action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
if [[ ${action_dim} != "14" ]]; then
    echo "[TRAIN][ERROR] The selected model requires a 14-dimensional robot." >&2
    exit 1
fi

stage=${FOCUS_VLWA_STAGE:-joint}
if [[ ${stage} != "joint" && ${stage} != "frozen" ]]; then
    echo "[TRAIN][ERROR] FOCUS_VLWA_STAGE must be joint or frozen." >&2
    exit 1
fi
init_checkpoint=${FOCUS_VLWA_INIT_CHECKPOINT:?Set FOCUS_VLWA_INIT_CHECKPOINT to an initialization checkpoint directory}
if [[ ${init_checkpoint} != /* ]]; then
    init_checkpoint="${PWD}/${init_checkpoint}"
fi
norm_stats_dir=${FOCUS_VLWA_NORM_STATS_DIR:-${init_checkpoint}/assets/arx_x5_sim}
for file in "${init_checkpoint}/model.safetensors" "${init_checkpoint}/model_config.json" "${norm_stats_dir}/norm_stats.json"; do
    if [[ ! -f ${file} ]]; then
        echo "[TRAIN][ERROR] Required initialization asset not found: ${file}" >&2
        exit 1
    fi
done
python_bin=${FOCUS_VLWA_PYTHON:-${POLICY_DIR}/source/.venv/bin/python}
if [[ ! -x ${python_bin} ]]; then
    echo "[TRAIN][ERROR] Python environment not found: ${python_bin}; run bash install.sh train or train-v2." >&2
    exit 1
fi

dataset=${FOCUS_VLWA_LEROBOT_REPOS:-${FOCUS_VLWA_DATASET:-${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}/shards/*}}
"${python_bin}" "${POLICY_DIR}/process_data.py" --dataset "${dataset}" --check-only
output_dir="${POLICY_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export PYTHONPATH="${POLICY_DIR}/source/src:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
IFS=',' read -r -a devices <<< "${gpu_ids}"
for ((index = 0; index < ${#devices[@]}; index++)); do
    for ((other = 0; other < index; other++)); do
        if [[ ${devices[index]} == "${devices[other]}" ]]; then
            echo "[TRAIN][ERROR] GPU indices must not repeat." >&2
            exit 1
        fi
    done
done
command=("${python_bin}")
if [[ ${#devices[@]} -gt 1 ]]; then
    command+=(-m torch.distributed.run --standalone --nproc-per-node "${#devices[@]}" --module focus_vlwa.scripts.post_train)
else
    command+=(-m focus_vlwa.scripts.post_train)
fi
command+=(--dataset "${dataset}" --init-checkpoint "${init_checkpoint}" --norm-stats-dir "${norm_stats_dir}"
    --output-dir "${output_dir}" --stage "${stage}" --seed "${seed}")
if [[ -n ${FOCUS_VLWA_TOKENIZER_PATH:-} ]]; then
    command+=(--tokenizer-path "${FOCUS_VLWA_TOKENIZER_PATH}")
fi
command+=("$@")
printf '[TRAIN] '
printf '%q ' "${command[@]}"
printf '\n'
if [[ ${FOCUS_VLWA_DRY_RUN:-0} == "1" ]]; then
    exit 0
fi
exec "${command[@]}"
