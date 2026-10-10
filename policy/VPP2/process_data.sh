#!/usr/bin/env bash
# Convert the public EE16 export, then build the fixed split and sampling index.
source "$(dirname "${BASH_SOURCE[0]}")/training_common.sh"
case "${1:-}" in
  convert|prepare) stage=$1; shift ;;
  *) echo 'Usage: bash process_data.sh {convert|prepare} [arguments]' >&2; exit 2 ;;
esac
exec bash "${vpp2_source}/scripts/robodojo/${stage}.sh" "$@"
