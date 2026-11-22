#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 9 ]] || { echo 'Expected BENCH TASK CKPT ENV_CFG ACTION SEED GPU POLICY_ENV PORT [HOST]' >&2; exit 2; }
POLICY_ENV="$8"
export VAP_LAUNCH_ROLE=server
source "$(dirname "$0")/common.sh"
export PATH="$POLICY_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$7"
if [[ -n "${VAP_TEXT_ENCODER_GPU:-}" && "$VAP_TEXT_ENCODER_GPU" != "$7" ]]; then
 [[ "$7" != *,* ]] || { echo 'Pass one policy GPU when using VAP_TEXT_ENCODER_GPU' >&2; exit 2; }
 export CUDA_VISIBLE_DEVICES="$7,$VAP_TEXT_ENCODER_GPU"
 export VAP_TEXT_ENCODER_DEVICE=cuda:1
fi
cd "$BENCH_ROOT"
"$POLICY_ENV/bin/python" -I - "$VAP_RUN_DIR/policy_runtime.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'policy_python': sys.executable, 'policy_prefix': sys.prefix}, indent=2) + '\n')
PY
if [[ "${VAP_REQUIRE_NO_WAN:-0}" == 1 ]]; then
 exec "$POLICY_ENV/bin/python" -I -u -c 'import os,sys; from runpy import run_path; sys.path[:0] = os.environ["VAP_SOURCE_PATH"].split(os.pathsep); run_path(sys.argv.pop(1), run_name="__main__")' "$VAP_POLICY_DIR/audit_no_wan.py" --config_path "$VAP_POLICY_DIR/deploy.yml" --overrides \
  bench_name="$1" task_name="$2" ckpt_name="$3" env_cfg_type="$4" action_type="$5" seed="$6" policy_name=vap port="$9" host="${10:-localhost}"
fi
exec "$POLICY_ENV/bin/python" -I -u -c 'import os,sys; from runpy import run_module; sys.path[:0] = os.environ["VAP_SOURCE_PATH"].split(os.pathsep); run_module("XPolicyLab.policy.vap.server", run_name="__main__", alter_sys=True)' --config_path "$VAP_POLICY_DIR/deploy.yml" --overrides \
 bench_name="$1" task_name="$2" ckpt_name="$3" env_cfg_type="$4" action_type="$5" seed="$6" policy_name=vap port="$9" host="${10:-localhost}"
