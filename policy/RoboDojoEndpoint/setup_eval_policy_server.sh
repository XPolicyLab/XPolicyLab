#!/usr/bin/env bash
set -euo pipefail
# Standard policy-side arguments; the local process performs transport only.
if [[ $# -lt 9 ]]; then
  echo 'Expected the nine standard policy-server arguments' >&2
  exit 2
fi
[[ "$4" == arx_x5 && "$5" == joint ]] || {
  echo 'Supported configuration: arx_x5 / joint' >&2; exit 2;
}
: "${POLICY_ENDPOINT_URL:?Set POLICY_ENDPOINT_URL to the provided ws:// or wss:// endpoint}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$8/bin/activate" ]]; then
  source "$8/bin/activate"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$8"
fi
exec python "$SCRIPT_DIR/ws_bridge.py" --url "$POLICY_ENDPOINT_URL" --port "$9" --host "${10:-127.0.0.1}"
