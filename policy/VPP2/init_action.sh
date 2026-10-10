#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/training_common.sh"
exec bash "${vpp2_source}/scripts/robodojo/init_action.sh" \
  --config "${VPP2_POLICY_DIR}/train_100k.yaml" "$@"
