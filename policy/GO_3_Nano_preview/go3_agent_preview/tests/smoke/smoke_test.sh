#!/usr/bin/env bash
# smoke_test.sh -- prove this host can run a go3_agent_preview evaluation end to end,
# before spending hours on the full official sweep.
#
# Four tasks x ten layouts, one machine, both branches. Every episode in this
# adapter starts in the code branch; the WAM branch is reached only when a task
# hits one of its declared abandonment checkpoints. So the smoke test has two
# jobs and reports them separately:
#
#   * the run completed the requested number of distinct layouts (environment,
#     assets, env_cfg, client/server transport, action shapes);
#   * the code branch actually executed its program (its tape), and -- when a
#     switch occurred -- the WAM backend answered and produced actions
#     (events.jsonl: switch_boundary -> wam_chunk -> first_wam_action).
#
# A run with no switch is reported as such, never as a pass of the WAM branch.
#
# Nothing about our own cluster is baked in: every path comes from a flag or the
# environment. Point it at any RoboDojo checkout and it runs where the official
# eval runs.
#
#   ./smoke_test.sh --robodojo-root /path/to/RoboDojo --policy-env /path/to/venv \
#                   --eval-env RoboDojo
#
# Exits: 0 all checks pass, 1 preflight, 2 a run failed, 3 checks failed, 4 no
# WAM backend was reachable while the run needed one.
set -uo pipefail

# ---------------------------------------------------------------- defaults
ROBODOJO_ROOT="${ROBODOJO_ROOT:-}"
POLICY_DIR="${POLICY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Tasks with short code-side plans and no PhysX refill. A task whose code planning
# runs for minutes per request (general_pickup) is left out on purpose: under the
# default 120 s request budget it fails on any host, which would say nothing about
# this one.
TASKS="${SMOKE_TASKS:-stack_blocks,push_T,deposit_coin,play_Xylophone}"
LAYOUTS="${SMOKE_LAYOUTS:-0-9}"
SEED="${SMOKE_SEED:-0}"
ENV_CFG="${SMOKE_ENV_CFG:-arx_x5}"
ACTION_TYPE="${SMOKE_ACTION_TYPE:-joint}"
EVAL_ENV="${SMOKE_EVAL_ENV:-RoboDojo}"
POLICY_ENV="${SMOKE_POLICY_ENV:-}"
ENV_SCRIPT="${SMOKE_ENV_SCRIPT:-}"
POLICY_GPU="${SMOKE_POLICY_GPU:-0}"
ENV_GPU="${SMOKE_ENV_GPU:-0}"
WAM_GPU="${SMOKE_WAM_GPU:-}"
REQUEST_TIMEOUT="${SMOKE_REQUEST_TIMEOUT:-}"
OUT="${SMOKE_OUT:-}"
KEEP=0
DEBUG=0
START_BACKEND=1

usage() {
  cat <<'EOF'
Usage: ./smoke_test.sh --robodojo-root PATH --policy-env ENV [options]

Required (or set ROBODOJO_ROOT / SMOKE_POLICY_ENV):
  --robodojo-root PATH   RoboDojo checkout that owns scripts/robodojo.sh
  --policy-env ENV       Policy conda env, uv env, or a prefix directory holding bin/python

Options:
  --tasks LIST           comma-separated tasks
                         (default: stack_blocks,push_T,deposit_coin,play_Xylophone)
  --layouts SPEC         layout ids: 0-9, or 0-4,25-29 (default: 0-9)
  --seed N               layout set / eval seed (default: 0)
  --env-cfg NAME         env config stem (default: arx_x5)
  --action-type NAME     joint (this adapter supports joint only)
  --eval-env ENV         simulator environment (default: RoboDojo)
  --env-script PATH      shell script sourced before each eval, for host setup
                         (module loads, LD_LIBRARY_PATH, Isaac paths). Nothing
                         host-specific is built in.
  --policy-gpu ID        GPU for the policy server (default: 0)
  --env-gpu ID           GPU for Isaac Sim (default: 0)
  --wam-gpu ID           GPU for the WAM backend (default: --policy-gpu). Give it
                         its own card if you can: Isaac Sim with 10 envs plus the
                         backend can exceed 24 GB.
  --request-timeout SEC  XPL_REQUEST_TIMEOUT_S exported to both processes.
                         Default: leave the client default (120 s). Raising it
                         raises the WAM budget, which is min(geact_request_timeout_s,
                         this - 10).
  --no-backend           do not start a backend; check only what already runs
  --out DIR              artifacts root (default: <robodojo>/outputs/go3_smoke_<stamp>)
  --keep                 keep artifacts even when everything passes
  --debug                print every command before running it
  -h, --help             this text

WAM backend. Read entirely from the environment so no checkpoint path is guessed:
  WAM_LAUNCHER    script that starts the backend, e.g. <policy>/wam/run_geact_backend.sh
  WAM_HEALTH_URL  health endpoint (default: http://127.0.0.1:<geact_server_port>/healthz)
  CKPT STAT_PATH WAM_VAE_DIR WAM_QWEN_DIR WAM_LATENT_STATS_PATH PYTHON_BIN
                  passed through to the launcher, which validates them itself
Without WAM_LAUNCHER the backend step is skipped and any switch is reported as a
failure to exercise WAM, not as a pass.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robodojo-root)   ROBODOJO_ROOT="$2"; shift 2 ;;
    --policy-dir)      POLICY_DIR="$2"; shift 2 ;;
    --tasks)           TASKS="$2"; shift 2 ;;
    --layouts)         LAYOUTS="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --env-cfg)         ENV_CFG="$2"; shift 2 ;;
    --action-type)     ACTION_TYPE="$2"; shift 2 ;;
    --eval-env)        EVAL_ENV="$2"; shift 2 ;;
    --policy-env)      POLICY_ENV="$2"; shift 2 ;;
    --env-script)      ENV_SCRIPT="$2"; shift 2 ;;
    --policy-gpu)      POLICY_GPU="$2"; shift 2 ;;
    --env-gpu)         ENV_GPU="$2"; shift 2 ;;
    --wam-gpu)         WAM_GPU="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --no-backend)      START_BACKEND=0; shift ;;
    --out)             OUT="$2"; shift 2 ;;
    --keep)            KEEP=1; shift ;;
    --debug)           DEBUG=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 1 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n=== %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
die()  { printf '[fail] %s\n' "$*" >&2; exit "${2:-1}"; }

# ---------------------------------------------------------------- preflight
step "preflight"
[[ -n "$ROBODOJO_ROOT" ]] || die "no --robodojo-root; this script assumes no checkout path"
[[ -d "$ROBODOJO_ROOT" ]] || die "no such directory: $ROBODOJO_ROOT"
ROBODOJO_ROOT="$(cd "$ROBODOJO_ROOT" && pwd)"
ROBO="$ROBODOJO_ROOT/scripts/robodojo.sh"
[[ -f "$ROBO" ]] || die "$ROBODOJO_ROOT is not a RoboDojo checkout (no scripts/robodojo.sh)"
[[ -f "$POLICY_DIR/deploy.yml" ]] || die "$POLICY_DIR is not a policy dir (no deploy.yml)"
POLICY_DIR="$(cd "$POLICY_DIR" && pwd)"
[[ -n "$POLICY_ENV" ]] || die "no --policy-env (or SMOKE_POLICY_ENV); this script assumes no env name"

command -v python3 >/dev/null || die "python3 not on PATH"
command -v curl >/dev/null || warn "curl not found; a TCP connect is used as the health probe"

resolve_python() {
  local env="$1" guess
  [[ -x "$env/bin/python" ]] && { printf '%s\n' "$env/bin/python"; return 0; }
  guess="$ROBODOJO_ROOT/../.conda/envs/$env/bin/python"
  [[ -x "$guess" ]] && { printf '%s\n' "$guess"; return 0; }
  if [[ -n "${CONDA_PREFIX:-}" && -x "$(dirname "$CONDA_PREFIX")/$env/bin/python" ]]; then
    printf '%s\n' "$(dirname "$CONDA_PREFIX")/$env/bin/python"; return 0
  fi
  return 1
}
POLICY_PY="$(resolve_python "$POLICY_ENV")" \
  || die "cannot resolve --policy-env '$POLICY_ENV' to a python; pass a prefix directory holding bin/python or a conda env name that exists"

ENV_CFG_FOUND=""
for candidate in "$ROBODOJO_ROOT/../env_cfg/$ENV_CFG.yml" "$ROBODOJO_ROOT/env_cfg/$ENV_CFG.yml"; do
  [[ -f "$candidate" ]] && ENV_CFG_FOUND="$candidate" && break
done
if [[ -n "$ENV_CFG_FOUND" ]]; then
  say "env_cfg:  $ENV_CFG_FOUND"
else
  warn "no env_cfg/$ENV_CFG.yml beside $ROBODOJO_ROOT; env_cfg/ lives in the parent workspace and both branches read it in their constructor"
fi
say "policy:   $POLICY_DIR"
say "python:   $POLICY_PY"

# The interpreter passed as --policy-env must be the one that will run the policy
# server, so validate the config with it rather than with a bare python3: a wrong
# env shows up here in one line instead of as a traceback after Isaac starts.
DEPLOY_CHECK="$(
"$POLICY_PY" - "$POLICY_DIR/deploy.yml" "$ACTION_TYPE" "$ENV_CFG" <<'PY' 2>&1
import sys
try:
    import yaml
except ModuleNotFoundError:
    sys.exit("PyYAML is not installed in this environment")
try:
    cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
except OSError as error:
    sys.exit(f"cannot read {sys.argv[1]}: {error}")
path, action_type, env_cfg = sys.argv[1:4]
if action_type != "joint":
    sys.exit(f"this adapter supports action_type=joint only, got {action_type!r}")
declared = cfg.get("action_type")
if declared not in (None, action_type):
    sys.exit(f"{path}: action_type={declared!r} contradicts --action-type {action_type!r}")
if env_cfg != "arx_x5":
    sys.exit(f"skill switching requires env_cfg_type=arx_x5, got {env_cfg!r}")
if cfg.get("memory_enabled") or cfg.get("bbox_enabled"):
    sys.exit("memory_enabled and bbox_enabled must stay false in this release")
PY
)" || die "$POLICY_PY cannot load $POLICY_DIR/deploy.yml: $DEPLOY_CHECK"

LAYOUT_IDS="$(python3 - "$LAYOUTS" <<'PY'
import sys
ids = []
for part in sys.argv[1].split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        first, last = (int(x) for x in part.split("-", 1))
        if last < first:
            sys.exit(f"empty layout range {part}")
        ids += list(range(first, last + 1))
    else:
        ids.append(int(part))
print(" ".join(str(i) for i in dict.fromkeys(ids)))
PY
)" || die "bad --layouts '$LAYOUTS'"
COUNT="$(printf '%s' "$LAYOUT_IDS" | tr ' ' '\n' | grep -c .)"
[[ "$COUNT" -gt 0 ]] || die "layout spec '$LAYOUTS' expanded to nothing"
say "tasks:    $TASKS"
say "layouts:  $(printf '%s' "$LAYOUT_IDS" | tr ' ' ',')  ($COUNT per task)"

if [[ -n "$REQUEST_TIMEOUT" ]]; then
  say "timeout:  XPL_REQUEST_TIMEOUT_S=$REQUEST_TIMEOUT (WAM budget becomes min(geact_request_timeout_s, $((REQUEST_TIMEOUT - 10))) s)"
else
  say "timeout:  client default (120 s); WAM budget is 110 s"
fi

# ---------------------------------------------------------------- artifacts
STAMP="$(date +%Y%m%d-%H%M%S)"
[[ -n "$OUT" ]] || OUT="$ROBODOJO_ROOT/outputs/go3_smoke_$STAMP"
mkdir -p "$OUT"/{logs,events,configs,manifests}
OUT="$(cd "$OUT" && pwd)"
RESULTS_ROOT="$ROBODOJO_ROOT/eval_result/RoboDojo"
mkdir -p "$RESULTS_ROOT"
say "artifacts: $OUT"

SCRATCH="${TMPDIR:-/tmp}/go3_smoke.$$"
mkdir -p "$SCRATCH"
BACKEND_PIDFILE="$OUT/wam-backend.log.pid"
cleanup() {
  [[ -n "${BACKEND_PIDFILE:-}" && -s "$BACKEND_PIDFILE" ]] || return 0
  [[ "${REUSED_BACKEND:-0}" == 1 ]] && return 0
  local pid; pid="$(cat "$BACKEND_PIDFILE" 2>/dev/null)" || return 0
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  say "stopped the WAM backend this script started (pgid $pid)"
}
trap cleanup EXIT

# ---------------------------------------------------------------- wam backend
GEACT_SETTINGS="$("$POLICY_PY" - "$POLICY_DIR/deploy.yml" <<'PY' 2>&1
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(int(cfg.get("geact_server_port", 8001)), cfg.get("geact_server_host", "127.0.0.1"))
PY
)" || die "cannot read geact_server_host/port from $POLICY_DIR/deploy.yml: $GEACT_SETTINGS"
GEACT_PORT="${GEACT_SETTINGS%% *}"
GEACT_HOST="${GEACT_SETTINGS##* }"
WAM_HEALTH_URL="${WAM_HEALTH_URL:-http://${GEACT_HOST}:${GEACT_PORT}/healthz}"
REUSED_BACKEND=0

backend_up() {
  if command -v curl >/dev/null; then
    local code
    code="$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$WAM_HEALTH_URL" 2>/dev/null)"
    # 200 is /healthz answering; 426 is the websocket server answering a plain HTTP
    # request without running its /healthz hook (that hook registers with the
    # pre-14 websockets signature, so on websockets >= 14 it never matches).
    # Either way the server is listening.
    [[ "$code" == 200 || "$code" == 426 ]]
    return
  fi
  local hostport="${WAM_HEALTH_URL#*://}"; hostport="${hostport%%/*}"
  timeout 3 bash -c ">/dev/tcp/${hostport%%:*}/${hostport##*:}" 2>/dev/null
}

start_backend() {
  step "wam backend"
  if backend_up; then
    say "a backend already answers $WAM_HEALTH_URL; reusing it and leaving it running"
    REUSED_BACKEND=1; return 0
  fi
  if [[ $START_BACKEND == 0 || -z "${WAM_LAUNCHER:-}" ]]; then
    warn "no backend is running and none will be started (WAM_LAUNCHER unset or --no-backend)."
    warn "Any episode that switches to WAM will fail; those failures are reported, not hidden."
    return 0
  fi
  [[ -f "$WAM_LAUNCHER" ]] || die "WAM_LAUNCHER=$WAM_LAUNCHER does not exist"
  local gpu="${WAM_GPU:-$POLICY_GPU}" log="$OUT/wam-backend.log"
  [[ "$gpu" == "$ENV_GPU" ]] && warn "the backend shares GPU $gpu with Isaac Sim; pass --wam-gpu to give it its own card"
  say "starting $WAM_LAUNCHER on GPU $gpu (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" setsid bash -c "echo \$\$ > '$BACKEND_PIDFILE'; exec '$WAM_LAUNCHER'" \
    >"$log" 2>&1 </dev/null &
  local waited=0
  while (( waited < 900 )); do
    if backend_up; then say "backend healthy at $WAM_HEALTH_URL after ${waited}s"; return 0; fi
    if [[ -s $BACKEND_PIDFILE ]] && ! kill -0 "$(cat "$BACKEND_PIDFILE")" 2>/dev/null; then
      warn "backend exited before answering; tail of $log:"; tail -25 "$log" | sed 's/^/  /'
      return 1
    fi
    sleep 5; waited=$((waited + 5))
    (( waited % 60 == 0 )) && say "  waiting for backend... ${waited}s"
  done
  warn "backend did not answer within 900s; tail of $log:"; tail -25 "$log" | sed 's/^/  /'
  return 1
}

# ---------------------------------------------------------------- configs
# One deploy config per task. env_cfg_type and task_name are supplied by the
# launch script, so only the trace directory differs; writing it here keeps the
# shipped deploy.yml untouched.
make_config() {
  local task="$1" dst="$2"
  "$POLICY_PY" - "$POLICY_DIR/deploy.yml" "$dst" "$OUT/events/$task" <<'PY' 2>&1
import sys
import yaml
src, dst, events = sys.argv[1:4]
cfg = yaml.safe_load(open(src, encoding="utf-8")) or {}
code = cfg.setdefault("code", {})
code["out_dir"] = events
with open(dst, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
PY
}

# RoboDojo takes layouts from the front of a fixed list, so a specific set is
# expressed through its resume manifest: wanted ids stay in the queue, everything
# else is marked abandoned. Anything already scored in this run is carried over --
# writing a blank manifest would discard the run's own history.
write_manifest() {
  local task="$1" run_id="$2" ids="$3"
  python3 - "$ROBODOJO_ROOT" "$RESULTS_ROOT" "$(basename "$POLICY_DIR")" "$ENV_CFG" "$SEED" \
           "$ACTION_TYPE" "$task" "$run_id" "$ids" <<'PY'
import json, sys
from pathlib import Path

root, results_root, policy, env_cfg, seed, action_type, task, run_id, ids = sys.argv[1:10]
wanted = [int(i) for i in ids.split()]

layout_dir = Path(root, "Assets", "Eval_Layout", "RoboDojo", env_cfg, str(seed))
if not layout_dir.is_dir():
    sys.exit(f"no layout dir {layout_dir}")
count = sum(1 for p in layout_dir.iterdir()
            if p.suffix == ".json" and p.stem.rsplit("_", 1)[0] == task
            and p.stem.rsplit("_", 1)[-1].isdigit())
if count == 0:
    sys.exit(f"no layouts for {task} under {layout_dir}")
bad = [i for i in wanted if not 0 <= i < count]
if bad:
    sys.exit(f"layout ids {bad} outside 0..{count - 1} for {task}")

result_dir = Path(results_root, task, policy, env_cfg, f"{seed}_ckpt_name=smoke,action_type={action_type}")
result_dir.mkdir(parents=True, exist_ok=True)
manifest = result_dir / f"_resume_{run_id}.json"

carried = []
durable = result_dir / run_id / "_result.json"
if durable.is_file():
    try:
        for _, detail in sorted(json.loads(durable.read_text())["details"].items(),
                                key=lambda kv: int(kv[0])):
            if int(detail["layout_id"]) in wanted:
                continue
            carried.append({"layout_id": int(detail["layout_id"]),
                            "success": bool(detail["success"]),
                            "score": float(detail.get("score", 0.0))})
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        sys.exit(f"cannot read {durable} ({error}); refusing to overwrite it")
done = {d["layout_id"] for d in carried}
payload = {
    "run_id": run_id, "save_dir": str(result_dir / run_id), "task_name": task,
    "policy_name": policy, "config_name": env_cfg, "eval_seed": int(seed),
    "additional_info": f"ckpt_name=smoke,action_type={action_type}",
    "success_nums": sum(1 for d in carried if d["success"]),
    "fail_nums": sum(1 for d in carried if not d["success"]),
    "unstable_nums": 0, "total_score": float(sum(d["score"] for d in carried)),
    "completed_layout_ids": sorted(done),
    "abandoned_layout_ids": [i for i in range(count) if i not in wanted and i not in done],
    "details": {str(i): d for i, d in enumerate(carried)}, "restart_count": 0,
}
tmp = Path(f"{manifest}.tmp")
tmp.write_text(json.dumps(payload, indent=2))
tmp.replace(manifest)
print(f"[smoke] {task}: keeping {wanted} of {count}, carrying {len(carried)} already scored")
PY
}

# ---------------------------------------------------------------- run
start_backend; BACKEND_RC=$?

RUN_FAILED=0
for task in ${TASKS//,/ }; do
  run_id="smoke-${task}-s0"
  log="$OUT/logs/${task}.log"
  cfg="$OUT/configs/deploy-${task}.yml"
  make_config "$task" "$cfg" || { warn "could not write $cfg"; RUN_FAILED=1; continue; }
  step "$task ($COUNT layouts)"
  if ! write_manifest "$task" "$run_id" "$LAYOUT_IDS" >"$OUT/manifests/${task}.log" 2>&1; then
    sed 's/^/  /' "$OUT/manifests/${task}.log"
    warn "could not pin layouts for $task"; RUN_FAILED=1; continue
  fi
  rc=0
  # DEPLOY_CONFIG is the supported way to hand the server a different deploy.yml.
  # XPL_REQUEST_TIMEOUT_S must reach the policy server too, not only the client.
  if [[ -n "$ENV_SCRIPT" ]]; then
    env_args=(env "DEPLOY_CONFIG=$cfg" "ROBODOJO_EVAL_ROOT=$RESULTS_ROOT" "ROBODOJO_RUN_ID=$run_id")
    [[ -n "$REQUEST_TIMEOUT" ]] && env_args+=("XPL_REQUEST_TIMEOUT_S=$REQUEST_TIMEOUT")
    "${env_args[@]}" bash -c "source '$ENV_SCRIPT' && exec bash '$ROBO' eval \
        --policy-dir '$POLICY_DIR' --task '$task' --ckpt smoke --env-cfg '$ENV_CFG' \
        --action-type '$ACTION_TYPE' --seed '$SEED' --policy-gpu '$POLICY_GPU' \
        --env-gpu '$ENV_GPU' --policy-env '$POLICY_ENV' --eval-env '$EVAL_ENV' \
        --eval-num '$COUNT'" >"$log" 2>&1 || rc=$?
  else
    env_args=(env "DEPLOY_CONFIG=$cfg" "ROBODOJO_EVAL_ROOT=$RESULTS_ROOT" "ROBODOJO_RUN_ID=$run_id")
    [[ -n "$REQUEST_TIMEOUT" ]] && env_args+=("XPL_REQUEST_TIMEOUT_S=$REQUEST_TIMEOUT")
    "${env_args[@]}" bash "$ROBO" eval --policy-dir "$POLICY_DIR" --task "$task" --ckpt smoke \
      --env-cfg "$ENV_CFG" --action-type "$ACTION_TYPE" --seed "$SEED" \
      --policy-gpu "$POLICY_GPU" --env-gpu "$ENV_GPU" \
      --policy-env "$POLICY_ENV" --eval-env "$EVAL_ENV" \
      --eval-num "$COUNT" >"$log" 2>&1 || rc=$?
  fi
  say "$(tr '\r' '\n' < "$log" 2>/dev/null | sed '/^$/d' | tail -n 1 | cut -c1-140)"
  [[ $rc == 0 ]] || { warn "$task exited rc=$rc (log: $log)"; RUN_FAILED=1; }
done

# ---------------------------------------------------------------- verify
step "verify"
python3 "$(dirname "${BASH_SOURCE[0]}")/check_smoke.py" \
  --results-root "$RESULTS_ROOT" --events-root "$OUT/events" --log-dir "$OUT/logs" \
  --tasks "$TASKS" --expect-layouts "$COUNT" --expect-seed "$SEED" \
  --json-out "$OUT/verify.json"
VERIFY=$?

say ""
case "$VERIFY" in
  0) say "PASS: every task completed $COUNT distinct layouts, the code branch ran, and every"
     say "      switch that occurred was served by the WAM backend."
     say "artifacts: $OUT"
     [[ $KEEP == 1 ]] || say "(logs and results are kept under $OUT)"
     exit 0 ;;
  3) say "FAIL: checks failed; see the per-task problems above and $OUT/logs."
     exit 3 ;;
  4) say "FAIL: a task switched to WAM but the backend never answered."
     [[ $BACKEND_RC != 0 ]] && say "      the backend this script started did not come up (see $OUT/wam-backend.log)."
     exit 4 ;;
  *) say "FAIL: a task run did not finish; see $OUT/logs."
     exit 2 ;;
esac
