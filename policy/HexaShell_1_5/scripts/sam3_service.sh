#!/usr/bin/env bash
# The machine's shared SAM3 service (services/sam3_server.py).
#
#   scripts/sam3_service.sh start GPU PORT     e.g. start 7 18300
#   scripts/sam3_service.sh stop PORT
#   scripts/sam3_service.sh status PORT
#   scripts/sam3_service.sh restart GPU PORT
#
# Log and pid file: logs/sam3_<host>_<port>.log and .pid in the policy folder.
# Python: $SAM3_PYTHON (default: the python on PATH); checkpoint: $SAM3_CHECKPOINT_PATH (config.env).
# Extra server options can be passed in $SAM3_SERVER_ARGS (e.g. "--cache-images 24").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/config.env" ]]; then set -a; source "$ROOT/config.env"; set +a; fi   # config.env.example
PY="${SAM3_PYTHON:-$(command -v python || true)}"
CKPT="${SAM3_CHECKPOINT_PATH:-}"
HOST="$(hostname -s)"
LOGDIR="$ROOT/logs"
NOPROXY=(env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy)

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

number() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "not a number: '${1:-}'" >&2; usage; }; }

paths() {  # paths PORT
    LOG="$LOGDIR/sam3_${HOST}_$1.log"
    PIDF="$LOGDIR/sam3_${HOST}_$1.pid"
}

# true if the pid file names a live sam3_server of ours on this port; sets PID
alive() {
    PID=""
    [[ -f "$PIDF" ]] || return 1
    PID="$(tr -dc '0-9' < "$PIDF")"
    [[ -n "$PID" && -r "/proc/$PID/cmdline" ]] || return 1
    [[ "$(stat -c %u "/proc/$PID")" == "$(id -u)" ]] || return 1
    local cmd
    cmd="$(tr '\0' ' ' < "/proc/$PID/cmdline")"
    [[ "$cmd" == *services/sam3_server.py* && "$cmd" == *"--port $PORT "* ]]
}

health() {  # health PORT -> prints the JSON, fails if nothing answers
    "${NOPROXY[@]}" curl -sf --max-time 10 "http://127.0.0.1:$1/health"
}

start() {
    GPU="$1" PORT="$2"
    number "$GPU"; number "$PORT"
    paths "$PORT"
    mkdir -p "$LOGDIR"
    if alive; then
        echo "already running: pid $PID, port $PORT, log $LOG"
        return 0
    fi
    if health "$PORT" > /dev/null 2>&1; then
        echo "port $PORT already answers /health, but not from our pid file ($PIDF); not starting" >&2
        return 1
    fi
    [[ -x "$PY" ]] || { echo "python not found: $PY" >&2; return 1; }
    [[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; return 1; }
    rm -f "$PIDF"
    echo "=== $(date '+%F %T') start on GPU $GPU, port $PORT, by $(id -un)" >> "$LOG"
    # own session, so closing the starting shell does not stop it; the server writes the pid file.
    # MALLOC_ARENA_MAX: one request thread per connection would otherwise grow many malloc arenas.
    # shellcheck disable=SC2086
    "${NOPROXY[@]}" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 MALLOC_ARENA_MAX=4 \
        setsid nohup "$PY" "$ROOT/services/sam3_server.py" --port "$PORT" --checkpoint "$CKPT" \
        --pid-file "$PIDF" ${SAM3_SERVER_ARGS:-} >> "$LOG" 2>&1 < /dev/null &
    local launcher=$! i
    for ((i = 0; i < 300; i++)); do
        if alive && health "$PORT" > /dev/null 2>&1; then
            echo "running: pid $PID, GPU $GPU, port $PORT, log $LOG"
            return 0
        fi
        if ! alive && ! kill -0 "$launcher" 2> /dev/null && ((i > 30)); then
            break
        fi
        sleep 1
    done
    echo "SAM3 service did not come up; last log lines ($LOG):" >&2
    tail -n 20 "$LOG" >&2
    return 1
}

stop() {
    PORT="$1"
    number "$PORT"
    paths "$PORT"
    if ! alive; then
        echo "not running (port $PORT)"
        rm -f "$PIDF"
        return 0
    fi
    kill -TERM "$PID"
    local i
    for ((i = 0; i < 40; i++)); do
        if ! kill -0 "$PID" 2> /dev/null; then
            echo "stopped pid $PID (port $PORT)"
            rm -f "$PIDF"
            return 0
        fi
        sleep 0.5
    done
    echo "pid $PID ignored SIGTERM for 20 s; sending SIGKILL" >&2
    kill -KILL "$PID" 2> /dev/null || true
    rm -f "$PIDF"
}

status() {
    PORT="$1"
    number "$PORT"
    paths "$PORT"
    if alive; then
        echo "running: pid $PID, port $PORT, log $LOG"
        if health "$PORT"; then
            echo
            return 0
        fi
        echo "but /health does not answer (still loading? see the log)"
        return 1
    fi
    echo "not running (port $PORT, pid file $PIDF)"
    if health "$PORT" > /dev/null 2>&1; then
        echo "note: something else answers /health on port $PORT"
    fi
    return 3
}

case "${1:-}" in
    start) [[ $# -eq 3 ]] || usage; start "$2" "$3" ;;
    stop) [[ $# -eq 2 ]] || usage; stop "$2" ;;
    status) [[ $# -eq 2 ]] || usage; status "$2" ;;
    restart) [[ $# -eq 3 ]] || usage; stop "$3"; start "$2" "$3" ;;
    *) usage ;;
esac
