#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 10 ]]; then
  printf '%s\n' 'Official Rule/WAM: eval.sh BENCH TASK CKPT ROBOT ACTION SEED POLICY_GPU ENV_GPU POLICY_ENV ENV_ENV' >&2
  exit 2
fi
if [[ -f "${9}/bin/python" ]]; then
  runner=("${9}/bin/python")
elif [[ -f "${9}" && -x "${9}" ]]; then
  runner=("${9}")
else
  runner=(conda run --no-capture-output -n "${9}" python)
fi
exec "${runner[@]}" "$ROOT/tools/official_launch.py" "$@"
