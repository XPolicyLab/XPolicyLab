#!/usr/bin/env bash
# Run a program-harness experiment, then rebuild the HTML views of its runs directory.
#
#   bash scripts/run_and_view.sh EXPERIMENT [harness program options...]
#
# Takes the same arguments as `scripts/harness program`, which runs unchanged; the
# views are rebuilt whether it succeeds or fails, and its exit status is kept.
# Use the interpreter of the active environment, or set PYTHON.
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3}"

output="runs"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --output) output="${args[i + 1]:-runs}" ;;
    --output=*) output="${args[i]#--output=}" ;;
  esac
done

"$python" "$script_dir/harness" program "$@"
status=$?
if [ -d "$output" ]; then
  "$python" "$script_dir/view_runs.py" "$output" >&2  # keep stdout for harness program's JSON
fi
exit "$status"
