#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [source_path]

Links HDF5 data into policy/RDT_1B/data/<4-tuple>/ and pre-encodes language embeddings
into policy/RDT_1B/lang_embeds/<4-tuple>/ (nothing is written into the shared dataset).

Source path resolution (first match wins):
  1. source_path argument
  2. ${RAW_DATA_ROOT} environment variable
  3. <XPolicyLab>/data/<bench_name>/<ckpt_name>
  4. <XPolicyLab>/data/<bench_name>_<ckpt_name>

Optional:
  --overwrite       Re-encode all lang_embed.pt files
  --skip-encode     Only create the data symlink
  --gpu N           GPU for T5 encoding (default: 0)

Example:
  bash process_data.sh RoboDojo stack_bowls arx_x5 joint
EOF
}

if [[ "$#" -lt 4 ]]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
shift 4

source_path=""
overwrite=0
skip_encode=0
encode_gpu=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite) overwrite=1 ;;
        --skip-encode) skip_encode=1 ;;
        --gpu)
            shift
            encode_gpu="${1:?missing value for --gpu}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "${source_path}" ]]; then
                source_path=$1
            else
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
    shift
done

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${POLICY_DIR}/../.." && pwd)"
DATA_TAG="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
DATA_DIR="${POLICY_DIR}/data/${DATA_TAG}"
LANG_EMBED_DIR="${POLICY_DIR}/lang_embeds"

WEIGHTS_DIR="${POLICY_DIR}/weights/RDT"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-${WEIGHTS_DIR}/t5-v1_1-xxl}"

resolve_source() {
    local candidate=""
    if [[ -n "${source_path}" ]]; then
        candidate="${source_path}"
    elif [[ -n "${RAW_DATA_ROOT:-}" ]]; then
        candidate="${RAW_DATA_ROOT}"
    elif [[ -d "${ROOT_DIR}/data/${bench_name}/${ckpt_name}" ]]; then
        candidate="${ROOT_DIR}/data/${bench_name}/${ckpt_name}"
    elif [[ -d "${ROOT_DIR}/data/${bench_name}_${ckpt_name}" ]]; then
        candidate="${ROOT_DIR}/data/${bench_name}_${ckpt_name}"
    else
        return 1
    fi
    cd "${candidate}" && pwd
}

if ! SRC_DIR="$(resolve_source)"; then
    echo "[RDT_1B] HDF5 source not found." >&2
    echo "[RDT_1B] Pass source_path or set RAW_DATA_ROOT, or place data at:" >&2
    echo "         ${ROOT_DIR}/data/${bench_name}/${ckpt_name}" >&2
    echo "         ${ROOT_DIR}/data/${bench_name}_${ckpt_name}" >&2
    exit 1
fi

if ! find "${SRC_DIR}" -name "*.hdf5" -print -quit | grep -q .; then
    echo "[RDT_1B] No .hdf5 files under ${SRC_DIR}" >&2
    exit 1
fi

mkdir -p "${POLICY_DIR}/data" "${LANG_EMBED_DIR}"
if [[ -e "${DATA_DIR}" && ! -L "${DATA_DIR}" ]]; then
    echo "[RDT_1B] ${DATA_DIR} exists and is not a symlink; remove it first." >&2
    exit 1
fi
ln -sfn "${SRC_DIR}" "${DATA_DIR}"
echo "[RDT_1B] data/${DATA_TAG} -> ${SRC_DIR}"

if [[ "${skip_encode}" -eq 1 ]]; then
    echo "[RDT_1B] Skipped language embedding (--skip-encode)."
    exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[RDT_1B] conda not found." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RDT_CONDA_ENV:-rdt_1b}"

encode_args=(--data_root "${DATA_DIR}" --output_root "${LANG_EMBED_DIR}" --model_path "${TEXT_ENCODER_NAME}" --env_pattern "${env_cfg_type}" --gpu "${encode_gpu}")
if [[ "${overwrite}" -eq 1 ]]; then
    encode_args+=(--overwrite)
fi

cd "${POLICY_DIR}/rdt"
PYTHONPATH=. python scripts/encode_robodojo_lang.py "${encode_args[@]}"
echo "[RDT_1B] lang_embeds/${DATA_TAG}/ ready"
