#!/usr/bin/env bash
# Run one model episode inside the agent container.
#   run_agent.sh RUN_DIR PROMPT_TEXT
# Environment:
#   LANE         lane name, one network pair per lane (default 0)
#   ROBO_PORT    host port of robo-server's agent interface (default 28700)
#   WALL, IMAGE; the model settings come from config.env
# robo-server must listen on the gateway address of the lane's external network
# (printed by `run_agent.sh --gateway`).
# Public packaging: image naming and deployment paths; see metadata/source_manifest.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$ROOT/config.env" ]; then set -a; source "$ROOT/config.env"; set +a; fi
export ROBOSHELL_PROFILE=open
LANE=${LANE:-0}
IMAGE=${IMAGE:-joyai-vl/agent:preview}
NET_INT=roboshell-int-$LANE
NET_EXT=roboshell-ext-$LANE
EGRESS=roboshell-egress-$LANE
MODEL_BASE_PATH=${MODEL_BASE_PATH:-/v1}
MODEL_PREFIX="${MODEL_BASE_PATH%/}/chat/completions"

# Lane networks. Docker's default address pools hold only ~30 networks per host (15 x 172.x/16, 16 x 192.168.x/20);
# when they are exhausted, fall back to a small explicit /28 from 10.202.0.0/16 chosen by a hash of the lane name
# (a few retries on collision), which allows hundreds of lanes.
make_net() {  # make_net NAME [--internal]
  local name=$1; shift
  docker network inspect "$name" >/dev/null 2>&1 && return 0
  docker network create "$@" "$name" >/dev/null 2>&1 && return 0
  local h i sub subnet gateway
  h=$(printf '%s' "$name" | cksum | cut -d' ' -f1)
  for i in 0 1 2 3 4 5 6 7 8 9; do
    sub=$(( (h + i * 7919) % 4096 ))
    subnet="10.202.$((sub / 16)).$(( (sub % 16) * 16 ))/28"
    gateway="10.202.$((sub / 16)).$(( (sub % 16) * 16 + 1 ))"
    if docker network create "$@" --subnet "$subnet" --gateway "$gateway" "$name" >/dev/null 2>&1; then return 0; fi
  done
  echo "could not create docker network $name" >&2; return 1
}
make_net "$NET_INT" --internal
make_net "$NET_EXT"
GATEWAY=$(docker network inspect "$NET_EXT" --format '{{(index .IPAM.Config 0).Gateway}}')
[ -n "$GATEWAY" ] || { echo "Docker network $NET_EXT has no IPv4 gateway; check its IPAM configuration" >&2; exit 1; }
if [ "${1:-}" = "--gateway" ]; then echo "$GATEWAY"; exit 0; fi

RUN=$1; PROMPT=$2
mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)  # docker needs absolute paths for bind mounts
: "${MODEL_UPSTREAM:?set MODEL_UPSTREAM in config.env}"
: "${MODEL_KEY_FILE:?set MODEL_KEY_FILE in config.env}"
[ -r "$MODEL_KEY_FILE" ] || { echo "cannot read $MODEL_KEY_FILE" >&2; exit 1; }
ROBO_PORT=${ROBO_PORT:-28700}
export MODEL=${MODEL:-gpt-6-astra}
export HEXASHELL_REASONING_EFFORT=${HEXASHELL_REASONING_EFFORT:-max}
mkdir -p "$RUN/work" "$RUN/egress_dump"

docker rm -f "$EGRESS" >/dev/null 2>&1 || true
docker run -d --name "$EGRESS" --network "$NET_EXT" \
  -v "$ROOT/agent/api/egress_proxy.py:/egress_proxy.py:ro" -v "$MODEL_KEY_FILE:/run/model-key:ro" \
  -e MODEL_UPSTREAM="$MODEL_UPSTREAM" -e MODEL_KEY_FILE=/run/model-key -e MODEL_PREFIX="$MODEL_PREFIX" \
  -e ROBO_UPSTREAM="http://$GATEWAY:$ROBO_PORT" \
  -e DUMP_DIR=/dump -v "$RUN/egress_dump:/dump" \
  --user "$(id -u):$(id -g)" "$IMAGE" python3 /egress_proxy.py >/dev/null
cleanup() { docker logs "$EGRESS" > "$RUN/egress.log" 2>&1 || true; docker rm -f "$EGRESS" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker network connect --alias egress "$NET_INT" "$EGRESS"

ROBOT15_PACKAGE="$ROOT/vendor/robot15/hexashell/robot"
START=$(date +%s)
set +e
timeout "${WALL:-5400}" docker run --rm --name "roboshell-agent-$LANE" --network "$NET_INT" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROBOSHELL_ROBOT15=1 \
  -e "ROBO_PROFILE=${ROBOSHELL_PROFILE:-}" -e MODEL -e HEXASHELL_REASONING_EFFORT \
  -e HEXASHELL_MAX_TURNS -e HEXASHELL_MAX_TOKENS -e HEXASHELL_REQUEST_TIMEOUT_S \
  -e HEXASHELL_CHAT_TEMPLATE_KWARGS -e HEXASHELL_HISTORY_REASONING \
  -e HEXASHELL_STREAM -e HEXASHELL_TEMPERATURE -e "HEXASHELL_BUDGET_S=${WALL:-5400}" \
  -e "MODEL_API_URL=http://egress:8080$MODEL_PREFIX" \
  -e "PYTHONPATH=/opt/robo/client:/opt/robo" \
  -v "$RUN/work:/work" \
  -v "$ROOT/agent/api/run_policy.py:/opt/robo/runner/run_policy.py:ro" \
  -v "$ROOT/agent/api/prompts:/opt/robo/runner/prompts:ro" \
  -v "$ROOT/roboshell/client:/opt/robo/client:ro" \
  -v "$ROBOT15_PACKAGE:/opt/robo/hexashell/robot:ro" \
  "$IMAGE" python3 /opt/robo/runner/run_policy.py "$PROMPT" > "$RUN/model_events.jsonl" 2> "$RUN/model_stderr.log" < /dev/null
RC=$?
set -e
docker rm -f "roboshell-agent-$LANE" >/dev/null 2>&1 || true
printf '{"rc": %s, "wall_s": %s}\n' "$RC" "$(( $(date +%s) - START ))" > "$RUN/run_meta.json"
cat "$RUN/run_meta.json"
exit "$RC"
