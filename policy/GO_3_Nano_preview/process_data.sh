#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: process_data.sh [<bench_name> <ckpt_name> <env_cfg_type> <action_type> [expected_episode_count]]'
  echo 'With no arguments, use the data paths in configs/runtime.conf.'
  exit 0
fi
source "$POLICY_DIR/configs/runtime.conf"
source "$POLICY_DIR/tools/policy_env.sh"
args=()
if (( $# != 0 )); then
  if (( $# < 4 || $# > 5 )); then echo 'Expected no arguments, or four/five positional arguments.' >&2; exit 2; fi
  [[ "${1,,}" == robotwin && "$4" == joint ]] || { echo 'Supported: processed RoboTwin / joint' >&2; exit 2; }
  args+=(--bench-name "$1" --ckpt-name "$2" --env-cfg-type "$3" --action-type "$4")
fi
if [[ -n "$SFT_PREPARED_DIR" ]]; then args+=(--output "$SFT_PREPARED_DIR"); fi
if [[ -n "${5:-}" ]]; then args+=(--expected-episodes "$5"); fi
go3_policy_environment
go3_require_policy_python
exec "$POLICY_PYTHON" -m go3_nano_preview.sft.prepare_data \
  --jsonl "${SFT_JSONL_PATH:?Set SFT_JSONL_PATH}" --data-root "${SFT_DATA_ROOT:?Set SFT_DATA_ROOT}" \
  "${args[@]}"
