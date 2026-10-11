# Environment of the simulator side. Source this file; settings come from config.env.
ROBOSHELL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ROBOSHELL_ROOT
if [ -f "$ROBOSHELL_ROOT/config.env" ]; then set -a; source "$ROBOSHELL_ROOT/config.env"; set +a; fi
: "${ROBODOJO_REPO:=$(cd "$ROBOSHELL_ROOT/../../.." && pwd)}"   # XPolicyLab/policy/RoboBigWater sits inside the RoboDojo checkout
: "${ISAAC_PYTHON:?set ISAAC_PYTHON in config.env}"
export ROBODOJO_REPO

if [ -n "${TMPDIR:-}" ]; then mkdir -p "$TMPDIR"; export TMPDIR; else unset TMPDIR; fi
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N
export PYTHONNOUSERSITE=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# ffmpeg for the episode videos lives in the Isaac environment
export PATH=$(dirname "$ISAAC_PYTHON"):$PATH
if [ -n "${SIM_CACHE:-}" ]; then
  export XDG_CACHE_HOME=$SIM_CACHE/cache XDG_DATA_HOME=$SIM_CACHE/data XDG_CONFIG_HOME=$SIM_CACHE/config XDG_STATE_HOME=$SIM_CACHE/state
  export NVIDIA_USER_CACHE_PATH=$SIM_CACHE/nv OMNI_USER_CONFIG_PATH=$SIM_CACHE/omniverse
  mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$NVIDIA_USER_CACHE_PATH" "$OMNI_USER_CONFIG_PATH"
fi
if [ -n "${EXTRA_LD_LIBRARY_PATH:-}" ]; then export LD_LIBRARY_PATH=$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
export PYTHONPATH=$ROBOSHELL_ROOT:$ROBODOJO_REPO:$ROBODOJO_REPO/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}
KIT_ARGS="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"

# run_on_gpu GPU OWNER MIN_FREE_MIB COMMAND...
run_on_gpu() {
  local gpu=$1 owner=$2 need=$3; shift 3
  if [ -n "${GPU_LOCK:-}" ] && [ -x "$GPU_LOCK" ]; then
    "$GPU_LOCK" "$gpu" "$owner" --min-free-mib "$need" -- "$@"
    return
  fi
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  if [ "$free" -lt "$need" ]; then echo "GPU $gpu has $free MiB free, $need needed" >&2; return 75; fi
  "$@"
}
