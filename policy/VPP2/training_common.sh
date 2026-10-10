#!/usr/bin/env bash
set -euo pipefail
export VPP2_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vpp2_source="${VPP2_POLICY_DIR}/upstream"
if [[ ! -f "${vpp2_source}/scripts/robodojo/train.sh" ]]; then
  echo 'Missing VPP2 source. Run bash install.sh <env_name> train first.' >&2
  exit 1
fi
