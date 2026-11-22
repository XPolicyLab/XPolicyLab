#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 10 ]] || { echo 'Expected BENCH TASK CKPT ENV_CFG ACTION SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV' >&2; exit 2; }
export POLICY_ENV="$9" SIM_ENV="${10}"
VAP_POLICY_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(cd "$VAP_POLICY_DIR/../../.." && pwd)"
VAP_DEPLOY_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
export VAP_LAUNCH_ROLE=owner
source "$VAP_POLICY_DIR/common.sh"
SERVER_PID= CLIENT_PID=
stop_group() {
 local pid="$1"
 [[ -n "$pid" ]] || return 0
 kill -TERM -- "-$pid" 2>/dev/null || true
 for _ in {1..10}; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 0.5; done
 kill -KILL -- "-$pid" 2>/dev/null || true
 wait "$pid" 2>/dev/null || true
}
cleanup() {
 local rc=$?
 trap - EXIT INT TERM
 set +e
 stop_group "$CLIENT_PID"
 stop_group "$SERVER_PID"
 printf "%s" "$rc" > "$VAP_RUN_DIR/exit_code.txt"
 if [[ "$rc" -eq 0 ]]; then printf '[MAIN] eval finished\n'; fi
 exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$POLICY_ENV/bin/python" -I - "$@" <<'PY'
import json,os,sys,time
from pathlib import Path
keys=['benchmark','task','checkpoint','env_cfg_type','action_type','seed','policy_gpu','sim_gpu','policy_env','sim_env']
data=dict(zip(keys,sys.argv[1:]));data.update(policy_env=os.environ['POLICY_ENV'],sim_env=os.environ['SIM_ENV'],start_time=time.time(),mode=os.environ.get('EVAL_ENV_TYPE','sim'),eval_num=os.environ.get('EVAL_NUM','native'),encoded_debug=os.environ.get('DEBUG_OBS_ENCODED','0'))
(Path(os.environ['VAP_RUN_DIR'])/'request.json').write_text(json.dumps(data,indent=2))
PY
port="${VAP_POLICY_PORT:-$(bash "$XPL_ROOT/utils/get_free_port.sh")}"
setsid bash "$VAP_POLICY_DIR/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$port" > "$VAP_RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$port" "$SERVER_PID" 'VAP policy server' 1800
setsid bash "$VAP_POLICY_DIR/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" "ckpt_name=$3,action_type=$5" "$port" localhost > "$VAP_RUN_DIR/client.log" 2>&1 &
CLIENT_PID=$!
wait "$CLIENT_PID"
CLIENT_PID=
