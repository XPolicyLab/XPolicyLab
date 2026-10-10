#!/usr/bin/env bash
set -euo pipefail

# Standard XPolicyLab form:
#   bash process_data.sh SparkArena egovla tianji_marvin_wuji joint [limit_per_task]
# A zero-argument full scan and the one-number smoke form are also supported.
limit_per_task=""
case "$#" in
  0)
    ;;
  1)
    [[ "$1" =~ ^[0-9]+$ ]] || {
      echo "usage: bash process_data.sh [SparkArena egovla tianji_marvin_wuji joint [limit_per_task]]" >&2
      exit 2
    }
    limit_per_task="$1"
    ;;
  4|5)
    [[ "$1" == "SparkArena" && "$2" == "egovla" && "$3" == "tianji_marvin_wuji" && "$4" == "joint" ]] || {
      echo "this entry point requires SparkArena egovla tianji_marvin_wuji joint" >&2
      exit 2
    }
    limit_per_task="${5:-}"
    ;;
  *)
    echo "usage: bash process_data.sh [SparkArena egovla tianji_marvin_wuji joint [limit_per_task]]" >&2
    exit 2
    ;;
esac

policy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="${SPARKARENA_RAW_ROOT:-${policy_dir}/data/raw/spark0_bench_7tasks}"
output_root="${policy_dir}/data/SparkArena-egovla-tianji_marvin_wuji-joint"
python_bin="${EGOVLA_PYTHON_BIN:-python3}"
val_percent="${SPARKARENA_VAL_PERCENT:-5}"

export PYTHONPATH="${policy_dir}/.deps:${policy_dir}/EgoVLA_Release/VILA:${policy_dir}/EgoVLA_Release:${policy_dir}:${PYTHONPATH:-}"

args=(
  "${policy_dir}/EgoVLA_Release/human_plan/dataset_preprocessing/sparkarena/process_data.py"
  --source-root "${source_root}"
  --output "${output_root}"
  --val-percent "${val_percent}"
)
if [[ -n "${limit_per_task}" ]]; then
  args+=(--limit-per-task "${limit_per_task}")
fi

exec "${python_bin}" "${args[@]}"
