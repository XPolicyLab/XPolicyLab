#!/bin/bash
# Wait until policy server process is listening on TCP port or the process exits.
# Uses LISTEN-state checks only (no TCP connect) to avoid websocket handshake errors.
# Usage: wait_for_policy_server.sh <host> <port> <server_pid> [label] [timeout_sec]
set -euo pipefail

host=${1:?host required}
port=${2:?port required}
pid=${3:?server pid required}
label=${4:-Policy server}
timeout_sec=${5:-360}

_port_is_listening() {
    local listen_port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${listen_port}" 2>/dev/null | grep -q LISTEN
        return $?
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${listen_port}" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if command -v netstat >/dev/null 2>&1; then
        netstat -an 2>/dev/null | grep -Eq "[\.:]${listen_port}[[:space:]].*LISTEN"
        return $?
    fi
    return 1
}

for _ in $(seq 1 "${timeout_sec}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo -e "\033[31m[ERROR] ${label} (PID=${pid}) exited before opening port ${port}.\033[0m" >&2
        exit 1
    fi
    if _port_is_listening "${port}"; then
        echo -e "\033[32m[MAIN] ${label} ready on ${host}:${port} (PID=${pid})\033[0m"
        exit 0
    fi
    sleep 1
done

echo -e "\033[31m[ERROR] ${label} timed out after ${timeout_sec}s waiting for port ${port}.\033[0m" >&2
exit 1
