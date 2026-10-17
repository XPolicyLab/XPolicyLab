#!/usr/bin/env bash
set -euo pipefail
# Eight standard policy-side arguments, then optional port base and bind host.
if [[ $# -lt 8 ]]; then
  echo 'Usage: serve_pool.sh bench task ckpt env_cfg action seed gpu policy_env [port_base] [host]' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
port_base="${9:-19100}"
count="${POLICY_SERVER_COUNT:-8}"
[[ "$port_base" =~ ^[0-9]+$ && "$count" =~ ^[1-8]$ ]] || { echo 'Invalid port base or server count' >&2; exit 2; }
(( port_base >= 1024 && port_base + count - 1 <= 65535 )) || exit 2
pids=()
ports=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((index=0; index<count; index++)); do
  port=$((port_base + index))
  bash "$SCRIPT_DIR/setup_eval_policy_server.sh" "${@:1:8}" "$port" "${10:-127.0.0.1}" &
  pids+=("$!")
  ports+=("$port")
done
for ((index=0; index<count; index++)); do
  bash "$XPL_ROOT/utils/wait_for_policy_server.sh" 127.0.0.1 "${ports[index]}" "${pids[index]}" 'Pool policy server' 120
done
port_list="$(IFS=,; echo "${ports[*]}")"
echo "[POOL] ready: $port_list"
wait -n "${pids[@]}"
# A failed pool member stops the pool rather than silently reducing capacity.
exit 1
