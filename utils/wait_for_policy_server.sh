#!/bin/bash
# Wait until policy server opens TCP port or the server process exits.
# Usage: wait_for_policy_server.sh <host> <port> <server_pid> [label] [timeout_sec]
set -euo pipefail

host=${1:?host required}
port=${2:?port required}
pid=${3:?server pid required}
label=${4:-Policy server}
timeout_sec=${5:-360}

for _ in $(seq 1 "${timeout_sec}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo -e "\033[31m[ERROR] ${label} (PID=${pid}) exited before opening port ${port}.\033[0m" >&2
        exit 1
    fi
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('${host}', int('${port}'))); s.close()" >/dev/null 2>&1; then
        echo -e "\033[32m[MAIN] ${label} ready on ${host}:${port} (PID=${pid})\033[0m"
        exit 0
    fi
    sleep 1
done

echo -e "\033[31m[ERROR] ${label} timed out after ${timeout_sec}s waiting for port ${port}.\033[0m" >&2
exit 1
