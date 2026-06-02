#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XONE_ROOT="$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"

export PYTHONPATH="${XPOLICYLAB_ROOT}:${XONE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"

# 按需修改以下参数
exec python -m real_env.run_real_env_workbench \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --policy_name ACT \
  --ckpt_setting RoboDojo_real-stack_bowls-piper-200-joint \
  --host 127.0.0.1 \
  --port 12345 \
  --eval_episode_num 10
