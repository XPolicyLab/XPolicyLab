#!/usr/bin/env bash
# Forward the official 0-100k joint recipe and key=value overrides.
source "$(dirname "${BASH_SOURCE[0]}")/training_common.sh"
xpl_root="$(cd "${VPP2_POLICY_DIR}/../.." && pwd)"
joint_dim=$(bash "${xpl_root}/utils/get_action_dim.sh" "${xpl_root}/.." arx_x5)
# arx_x5 replaces each of its two 6D joint vectors with a 7D quaternion pose.
if (( joint_dim + 2 != 16 )); then
  echo 'The registered arx_x5 dimensions no longer match the released EE16 recipe.' >&2
  exit 1
fi
exec bash "${vpp2_source}/scripts/robodojo/train.sh" \
  --config "${VPP2_POLICY_DIR}/train_100k.yaml" "$@"
