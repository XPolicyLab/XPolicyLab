#!/usr/bin/env bash
# Run one Codex episode inside the agent container.
#   run_agent.sh RUN_DIR PROMPT_TEXT
# Environment:
#   LANE         lane name, one network pair per lane (default 0)
#   ROBO_PORT    host port of robo-server's agent interface (default 28700)
#   WALL, IMAGE, AGENTS_MD; the model settings come from config.env
# robo-server must listen on the gateway address of the lane's external network
# (printed by `run_agent.sh --gateway`).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$ROOT/config.env" ]; then set -a; source "$ROOT/config.env"; set +a; fi
LANE=${LANE:-0}
IMAGE=${IMAGE:-roboshell/agent-codex:0.0.1}
NET_INT=roboshell-int-$LANE
NET_EXT=roboshell-ext-$LANE
EGRESS=roboshell-egress-$LANE
MODEL_BASE_PATH=${MODEL_BASE_PATH:-/v1}
MODEL_PREFIX=${MODEL_PREFIX:-/$(echo "$MODEL_BASE_PATH" | cut -d/ -f2)/}

docker network inspect "$NET_INT" >/dev/null 2>&1 || docker network create --internal "$NET_INT" >/dev/null
docker network inspect "$NET_EXT" >/dev/null 2>&1 || docker network create "$NET_EXT" >/dev/null
GATEWAY=$(docker network inspect "$NET_EXT" --format '{{(index .IPAM.Config 0).Gateway}}')
if [ "${1:-}" = "--gateway" ]; then echo "$GATEWAY"; exit 0; fi

RUN=$1; PROMPT=$2
mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)  # docker needs absolute paths for bind mounts
: "${MODEL_UPSTREAM:?set MODEL_UPSTREAM in config.env}"
: "${MODEL_KEY_FILE:?set MODEL_KEY_FILE in config.env}"
[ -r "$MODEL_KEY_FILE" ] || { echo "cannot read $MODEL_KEY_FILE" >&2; exit 1; }
[ -f "${AGENTS_MD:-$ROOT/agent/AGENTS.md}" ] || "$ROOT/agent/build_docs.sh" >/dev/null
ROBO_PORT=${ROBO_PORT:-28700}
MODEL=${MODEL:-gpt-6-astra}; EFFORT=${EFFORT:-medium}
AGENTS_MD=${AGENTS_MD:-$ROOT/agent/AGENTS.md}
mkdir -p "$RUN/work" "$RUN/codex_home" "$RUN/egress_dump"
cp "$AGENTS_MD" "$RUN/work/AGENTS.md"
# Codex inside the container talks to the egress proxy; the placeholder key is replaced there
cat > "$RUN/codex_home/config.toml" <<TOML
model = "$MODEL"
model_provider = "gateway"
model_reasoning_effort = "$EFFORT"

[model_providers.gateway]
name = "gateway"
base_url = "http://egress:8080$MODEL_BASE_PATH"
wire_api = "responses"
env_key = "GATEWAY_PLACEHOLDER"
${MODEL_QUERY_PARAMS:+query_params = { $MODEL_QUERY_PARAMS \}}
TOML

docker rm -f "$EGRESS" >/dev/null 2>&1 || true
docker run -d --name "$EGRESS" --network "$NET_EXT" \
  -v "$ROOT/agent/codex/egress_proxy.py:/egress_proxy.py:ro" -v "$MODEL_KEY_FILE:/run/model-key:ro" \
  -e MODEL_UPSTREAM="$MODEL_UPSTREAM" -e MODEL_KEY_FILE=/run/model-key -e MODEL_PREFIX="$MODEL_PREFIX" \
  -e ROBO_UPSTREAM="http://$GATEWAY:$ROBO_PORT" \
  -e DUMP_DIR=/dump -v "$RUN/egress_dump:/dump" \
  --user "$(id -u):$(id -g)" "$IMAGE" python3 /egress_proxy.py >/dev/null
docker network connect --alias egress "$NET_INT" "$EGRESS"
cleanup() { docker logs "$EGRESS" > "$RUN/egress.log" 2>&1 || true; docker rm -f "$EGRESS" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "$PROMPT" = "--shell" ]; then shift 2; AGENT_CMD=("$@"); else
DISABLED=(memories multi_agent multi_agent_v2 apps plugins image_generation browser_use computer_use skill_search goals hooks)
AGENT_CMD=(codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox
  $(printf -- '--disable %s ' "${DISABLED[@]}")
  -C /work -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"" -o /work/.last_message.txt "$PROMPT")
fi

START=$(date +%s)
set +e
timeout "${WALL:-3600}" docker run --rm --name "roboshell-agent-$LANE" --network "$NET_INT" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e CODEX_HOME=/codex-home \
  -v "$RUN/work:/work" -v "$RUN/codex_home:/codex-home" \
  -v "$ROOT/roboshell/client/robo.py:/opt/robo/robo.py:ro" \
  "$IMAGE" "${AGENT_CMD[@]}" > "$RUN/codex_events.jsonl" 2> "$RUN/codex_stderr.log" < /dev/null
RC=$?
set -e
docker rm -f "roboshell-agent-$LANE" >/dev/null 2>&1 || true
[ -f "$RUN/work/.last_message.txt" ] && mv "$RUN/work/.last_message.txt" "$RUN/last_message.txt"
echo "{\"rc\": $RC, \"wall_s\": $(( $(date +%s) - START )), \"model\": \"$MODEL\", \"effort\": \"$EFFORT\", \"codex\": \"$(cat "$ROOT/agent/codex/CODEX_VERSION" 2>/dev/null)\", \"image\": \"$IMAGE\"}" > "$RUN/run_meta.json"
cat "$RUN/run_meta.json"
