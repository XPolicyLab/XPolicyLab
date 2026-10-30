#!/usr/bin/env bash
# Launch-time environment of the official RoboDojo evaluator for AGP (sourced; the default eval_env of
# XPolicyLab/policy/AGP/setup_eval_env_client.sh). Packaged (this script in XPolicyLab/policy/AGP/agp_runtime/scripts,
# scripts/package_policy.py) the checkout is the one the policy folder is in; in the development tree W it is
# $AGP_EVAL_ROOT/RoboDojo, else W/../robodojo-eval/RoboDojo (agp_core/paths.py).
#
#   1. Sources scripts/activate_sim.sh next to this script: AGP_HOME (default ~/.agp-robodojo when packaged),
#      $AGP_HOME/.venv-sim on PATH, the caches and TMPDIR under AGP_HOME, PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1.
#   2. Replaces its PYTHONPATH with the checkout and its XPolicyLab (env, utils, task and most env/* subpackages are
#      namespace packages: every sys.path entry holding such a folder contributes a portion, so nothing but the
#      checkout may hold them), plus the checkout's own IsaacLab and cuRobo package roots (EVAL_THIRD_PARTY_FROM_CHECKOUT,
#      default 1: a package found on sys.path wins over the editable finders of .venv-sim).
#   3. Exports ROBODOJO_ROOT (the checkout) and AGP_EVAL_ROOT (the folder holding it).
_agp_code="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(basename "$_agp_code")" == "agp_runtime" && -f "$_agp_code/../deploy.yml" ]]; then
  _agp_checkout="$(cd "$_agp_code/../../../.." && pwd -P)"
else
  _agp_checkout="${AGP_EVAL_ROOT:-$(cd "$_agp_code/.." && pwd -P)/robodojo-eval}/RoboDojo"
fi
# shellcheck disable=SC1091
source "$_agp_code/scripts/activate_sim.sh"

export ROBODOJO_ROOT="$_agp_checkout"
AGP_EVAL_ROOT="$(dirname "$_agp_checkout")"
export AGP_EVAL_ROOT
PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
if [[ "${EVAL_THIRD_PARTY_FROM_CHECKOUT:-1}" == "1" ]]; then
  _tp="$ROBODOJO_ROOT/third_party"
  PYTHONPATH="$PYTHONPATH:$_tp/IsaacLab/source/isaaclab:$_tp/IsaacLab/source/isaaclab_assets:$_tp/IsaacLab/source/isaaclab_tasks:$_tp/curobo"
  unset _tp
fi
export PYTHONPATH
unset _agp_code _agp_checkout
