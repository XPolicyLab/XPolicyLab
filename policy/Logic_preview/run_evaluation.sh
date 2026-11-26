#!/usr/bin/env bash
# Run from inside an official RoboDojo + XPolicyLab checkout.
set -eo pipefail
usage() {
  echo 'Usage: bash run_evaluation.sh [debug|smoke|full [--seeds SEED ...]]'
}
mode=full
if [[ $# -gt 0 && "$1" != --seeds ]]; then
  mode=$1
  shift
fi
case "$mode" in
  debug|smoke|full) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
layout_seeds=(0 1 2)
if (( $# )); then
  if [[ "$mode" != full || "$1" != --seeds || $# -lt 2 ]]; then
    usage >&2; exit 2
  fi
  shift
  layout_seeds=()
  for seed in "$@"; do
    if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ || " ${layout_seeds[*]} " == *" $seed "* ]]; then
      echo "Expected distinct non-negative layout seeds; got '$seed'." >&2; exit 2
    fi
    layout_seeds+=("$seed")
  done
fi
policy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl="$(cd "$policy_dir/../.." && pwd)"
bench="$(cd "$xpl/.." && pwd)"
if [[ ! -f "$xpl/setup_policy_server.py" || ! -f "$bench/scripts/robodojo.sh" ]]; then
  echo 'Install policy/Logic_preview under RoboDojo/XPolicyLab before running evaluation.' >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
sim_env=${EVAL_ENV:-RoboDojo}
policy_gpu=${POLICY_GPU:-0,1,2,3}
env_gpu=${ENV_GPU:-4}
chunk=${LAYOUTS_PER_SHARD:-12}
if [[ ! "$chunk" =~ ^(0|[1-9][0-9]*)$ || ! "$env_gpu" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo 'Expected a non-negative LAYOUTS_PER_SHARD and comma-separated ENV_GPU ids.' >&2
  exit 2
fi
first_env_gpu=${env_gpu%%,*}
if [[ "$mode" == full && "$chunk" != 0 ]]; then
  export LOGIC_PREVIEW_RUN_ID=${LOGIC_PREVIEW_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(cat /proc/sys/kernel/random/uuid)}
  if [[ ! "$LOGIC_PREVIEW_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]]; then
    echo 'Invalid LOGIC_PREVIEW_RUN_ID' >&2
    exit 2
  fi
  export LOGIC_PREVIEW_SEED_ORDER="${layout_seeds[*]}"
  printf 'Shared run ID: %s (reuse this ID to resume)\n' "$LOGIC_PREVIEW_RUN_ID"
fi
if [[ "$policy_gpu" == *,* ]]; then
  export LOGIC_PREVIEW_INFERENCE_DEVICE_MAP=${LOGIC_PREVIEW_INFERENCE_DEVICE_MAP:-balanced}
fi
ckpt=joint2b_746_s010000
# A CUDA build can be reused even if the current process cannot see a GPU.
# install.sh checks device access before installing policy dependencies.
has_policy_runtime() {
  python - <<'PY'
import re
import sys
import torch
import torchvision

def major_minor(version):
    return tuple(map(int, re.match(r"(\d+)\.(\d+)", version).groups()))

sys.exit(0 if (sys.version_info >= (3, 10) and torch.version.cuda
               and major_minor(torch.__version__) >= (2, 5)
               and major_minor(torchvision.__version__) >= (0, 20)) else 1)
PY
}
# Avoid replacing a package already in use by the simulator when sharing its env.
can_share_sim_deps() {
  python - "$xpl/pyproject.toml" "$policy_dir/model_code/pyproject.toml" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    from pip._vendor import tomli as tomllib
from pip._vendor.packaging.requirements import Requirement

for path in sys.argv[1:]:
    with open(path, "rb") as project:
        dependencies = tomllib.load(project)["project"]["dependencies"]
    for item in map(Requirement, dependencies):
        if item.marker and not item.marker.evaluate():
            continue
        try:
            installed = version(item.name)
        except PackageNotFoundError:
            continue
        if not item.specifier.contains(installed, prereleases=True):
            sys.exit(f"The simulator has {item.name} {installed}, but the policy requires {item}.")
PY
}
# Concurrent launchers may share a writable Conda prefix and editable installs.
if [[ "$mode" == full && "$chunk" != 0 ]]; then
  mkdir -p "$bench/eval_result/.logic-preview"
  exec {setup_lock}>"$bench/eval_result/.logic-preview/setup.lock"
  flock -x "$setup_lock"
fi
conda activate "$sim_env"
sim_prefix=$CONDA_PREFIX
policy_env=${POLICY_ENV:-logic_preview}
if ! conda activate "$policy_env" 2>/dev/null; then
  conda activate "$sim_env"
  if [[ -z "${POLICY_ENV:-}" ]] && has_policy_runtime >/dev/null 2>&1 && can_share_sim_deps; then
    policy_env=$sim_env
  else
    policy_python=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2] if sys.version_info >= (3, 10) else "3.10")')
    if [[ "$policy_env" == */* ]]; then
      conda create -y -p "$policy_env" "python=$policy_python" pip
    else
      conda create -y -n "$policy_env" "python=$policy_python" pip
    fi
    conda activate "$policy_env"
  fi
fi
if [[ -n "${POLICY_ENV:-}" && "$CONDA_PREFIX" == "$sim_prefix" ]]; then
  can_share_sim_deps || { echo 'Select a separate POLICY_ENV to preserve the simulator packages.' >&2; exit 1; }
fi
printf 'Policy environment: %s (%s)\nSimulator environment: %s\n' "$policy_env" "$CONDA_PREFIX" "$sim_env"
python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "The policy requires Python 3.10 or later.")'
if ! has_policy_runtime >/dev/null 2>&1; then
  python -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi
bash "$policy_dir/install.sh"
if [[ -n "${setup_lock:-}" ]]; then
  exec {setup_lock}>&-
fi
if [[ -n "${LOGIC_PREVIEW_WEIGHTS_DIR:-}" ]]; then
  LOGIC_PREVIEW_WEIGHTS_DIR="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$LOGIC_PREVIEW_WEIGHTS_DIR")"
  export LOGIC_PREVIEW_WEIGHTS_DIR
fi
weights_start=$SECONDS
bash "$policy_dir/download_checkpoints.sh"
printf 'Weight preparation completed in %ss\n' "$((SECONDS - weights_start))"
if [[ "$mode" == debug ]]; then
  for encoded in 0 1; do
    EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED="$encoded" bash "$policy_dir/eval.sh" \
      RoboDojo stack_bowls "$ckpt" arx_x5 ee 1 "$policy_gpu" "$first_env_gpu" "$policy_env" "$sim_env"
  done
  exit 0
fi

cd "$bench"
server=
cleanup() {
  if [[ -n "$server" ]]; then
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    server=
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
conda activate "$sim_env"
client=(bash scripts/robodojo.sh client --dataset RoboDojo --policy-name Logic_preview
        --policy-host 127.0.0.1 --env-cfg arx_x5
        --action-type ee --ckpt "$ckpt" --env-gpu "$first_env_gpu")
if [[ "$mode" == smoke ]]; then
  layout_seeds=(1)
  client+=(--task stack_bowls --eval-num 1)
else
  export LOGIC_PREVIEW_PACKED_SERVER=1
  export EVAL_ENV_TYPE=sim
  unset EVAL_NUM
  client+=(--gpu-ids "$env_gpu" --eval-num native)
fi
for seed in "${layout_seeds[@]}"; do
  seed_start=$SECONDS
  port="$(bash "$xpl/utils/get_free_port.sh")"
  bash "$policy_dir/setup_eval_policy_server.sh" RoboDojo stack_bowls "$ckpt" \
    arx_x5 ee "$seed" "$policy_gpu" "$policy_env" "$port" 127.0.0.1 &
  server=$!
  bash "$xpl/utils/wait_for_policy_server.sh" 127.0.0.1 "$port" "$server" "Logic_preview" 1200
  printf 'Seed %s policy ready in %ss\n' "$seed" "$((SECONDS - seed_start))"
  if [[ "$mode" == full && "$chunk" != 0 ]]; then
    python "$policy_dir/packed_eval.py" --gpus "$env_gpu" --chunk "$chunk" \
      --port "$port" --seed "$seed" --ckpt "$ckpt"
  else
    ports="$port"
    if [[ "$mode" == full ]]; then
      IFS=',' read -ra workers <<< "$env_gpu"
      for ((i=1; i<${#workers[@]}; i++)); do ports+=",$port"; done
    fi
    "${client[@]}" --policy-port "$ports" --seed "$seed"
  fi
  cleanup
  printf 'Seed %s completed in %ss\n' "$seed" "$((SECONDS - seed_start))"
done
if [[ "$mode" == full ]]; then
  mkdir -p eval_result
  flock eval_result/.logic-preview-summary.lock bash scripts/robodojo.sh summarize
fi
