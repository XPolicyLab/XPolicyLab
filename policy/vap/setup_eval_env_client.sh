#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 10 ]] || { echo 'Expected standard XPolicyLab client arguments' >&2; exit 2; }
SIM_ENV="$8"
[[ -n "${SIM_ENV}" || -n "${ROBODOJO_SIM_ENV:-}" ]] || {
 echo 'Split VAP client requires ROBODOJO_SIM_ENV or SIM_ENV' >&2
 exit 2
}
export VAP_LAUNCH_ROLE=client
# The official client receives only SIM_ENV; server policy packages are not imported here.
export POLICY_ENV="${POLICY_ENV:-${ROBODOJO_SIM_ENV:-$SIM_ENV}}"
source "$(dirname "$0")/common.sh"
# The official simulator installer uses the same noninteractive Kit EULA flag.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
 [[ -x "$SIM_ENV/bin/python" ]] || { echo 'Debug requires a separate client Python with numpy, opencv-python-headless, PyYAML and websockets' >&2; exit 2; }
 cd "$BENCH_ROOT"
 export PATH="$SIM_ENV/bin:$PATH"
 exec "$SIM_ENV/bin/python" -u -m XPolicyLab.policy.vap.debug_client --bench "$1" --task "$2" --env_cfg "$4" --port "${10}" --host "${11:-localhost}"
fi
[[ -x "$SIM_ENV/bin/python" ]] || { echo 'Set SIM_ENV to the separately installed RoboDojo simulator environment' >&2; exit 2; }
export LD_LIBRARY_PATH="$SIM_ENV/lib:${LD_LIBRARY_PATH:-}"
for source in "$BENCH_ROOT"/third_party/IsaacLab/source/*; do
 [[ ! -d "$source" ]] || export PYTHONPATH="$source:$PYTHONPATH"
done
export PYTHONPATH="$BENCH_ROOT/third_party/curobo:$PYTHONPATH"
if [[ -f "$VAP_DEPLOY_ROOT/configs/runtime/nvidia_icd.json" ]]; then
 export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$VAP_DEPLOY_ROOT/configs/runtime/nvidia_icd.json}"
 export VK_DRIVER_FILES="${VK_DRIVER_FILES:-$VK_ICD_FILENAMES}"
 export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$VAP_DEPLOY_ROOT/configs/runtime/10_nvidia_egl.json}"
 export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
exec bash "$XPL_ROOT/utils/setup_env_client.sh" "$XPL_ROOT/utils" "$VAP_POLICY_DIR/deploy.yml" "$SIM_ENV" "${10}" "$1" "$2" "$4" vap "$9" "$BENCH_ROOT" "$6" "$7" "${11:-localhost}"
