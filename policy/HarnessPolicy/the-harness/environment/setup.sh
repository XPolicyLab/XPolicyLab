#!/usr/bin/env bash
# Create or update the project conda environment (default name: harness-policy).
#
# One environment serves the framework, the RoboDojo simulator worker, the
# kernel that runs model-written programs, and learned-skill servers. Core
# simulator versions are pinned in environment/constraints.txt and every
# later layer is installed under those constraints, so a project dependency
# can never move torch, numpy, or Isaac.
#
#   bash environment/setup.sh                 # all steps
#   bash environment/setup.sh --from project  # resume at a step
#   bash environment/setup.sh --only lock     # one step
#
# Steps: conda torch isaacsim isaaclab curobo robodojo robotwin project skills docs verify lock
#
# Requires a RoboDojo checkout with submodules at HARNESS_ROBODOJO_ROOT
# (default third_party/RoboDojo); see third_party/sources.toml for revisions.
# Simulator assets are a separate download (RoboDojo scripts/init_assets.sh).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${HARNESS_ENV:-harness-policy}"
ROBODOJO_ROOT="$(cd "${HARNESS_ROBODOJO_ROOT:-$ROOT/third_party/RoboDojo}" 2>/dev/null && pwd || true)"
CONSTRAINTS="$ROOT/environment/constraints.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
STEPS=(conda torch isaacsim isaaclab curobo robodojo robotwin project skills docs verify lock)

export PIP_USER=0 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y TERM="${TERM:-xterm-256color}"
[ "$TERM" = dumb ] && export TERM=xterm-256color

info() { printf '\033[1;32m>>> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m[ERROR] %s\033[0m\n' "$*" >&2; exit 1; }
pip_c() { python -m pip install --constraint "$CONSTRAINTS" "$@"; }
need_robodojo() {
  [ -n "$ROBODOJO_ROOT" ] && [ -f "$ROBODOJO_ROOT/scripts/install.sh" ] \
    || fail "RoboDojo checkout not found. Set HARNESS_ROBODOJO_ROOT or run: python3 scripts/install_plan.py robodojo"
}

activate() {
  local conda_exe="${CONDA_EXE:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}"
  [ -x "$conda_exe" ] || fail "conda not found; install Miniconda first"
  # shellcheck disable=SC1091
  source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    [ "$1" = create ] || fail "Environment $ENV_NAME does not exist; run the conda step"
    info "Creating conda environment $ENV_NAME (Python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
  [ "$(python -c 'import sys; print(sys.version_info[:2])')" = "(3, 11)" ] || fail "$ENV_NAME must be Python 3.11"
}

step_conda() { :; }

step_torch() {
  info "PyTorch 2.7.0 (CUDA 12.8)"
  python -m pip install --upgrade "pip<26"
  pip_c torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "$TORCH_INDEX"
}

step_isaacsim() {
  info "Isaac Sim 5.1"
  pip_c "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
}

step_isaaclab() {
  need_robodojo
  info "Isaac Lab from $ROBODOJO_ROOT/third_party/IsaacLab"
  (cd "$ROBODOJO_ROOT/third_party/IsaacLab" && PIP_CONSTRAINT="$CONSTRAINTS" \
     ISAACLAB_RL_FRAMEWORK=none ./isaaclab.sh --install none)
}

step_curobo() {
  need_robodojo
  info "cuRobo from $ROBODOJO_ROOT/third_party/curobo"
  (cd "$ROBODOJO_ROOT/third_party/curobo" && pip_c -e ".[cu12]" --no-build-isolation)
}

step_robodojo() {
  need_robodojo
  info "RoboDojo runtime requirements and XPolicyLab client"
  pip_c -r "$ROBODOJO_ROOT/scripts/requirements.txt"
  pip_c opencv-python-headless==4.11.0.86 pillow matplotlib scikit-learn
  # Overrides the websockets==12.0 pin of isaacsim-kernel; see constraints.txt.
  pip_c websockets
  pip_c -e "$ROBODOJO_ROOT/XPolicyLab"
}

step_robotwin() {
  # RoboTwin 2.0 (SAPIEN) for the RoboTwin benchmark, in the same environment. Upstream
  # pins torch 2.4.1 / sapien 3.0.0b1 / its own cuRobo 0.7.8; the simulator and planners
  # run on the pinned core, sapien 3.0.1 (cp311) and the cuRobo already installed.
  # sapien's opencv-python requirement would add a second cv2 next to
  # opencv-python-headless, so the three wheels go in without dependencies (their other
  # requirements are in the environment already).
  local root="${HARNESS_ROBOTWIN_ROOT:-$ROOT/third_party/RoboTwin}"
  if [ ! -d "$root" ]; then
    info "RoboTwin not found at $root; skipping (see third_party/sources.toml)"
    return
  fi
  info "RoboTwin simulator dependencies"
  pip_c --no-deps sapien==3.0.1 mplib==0.2.1 toppra==0.6.10
  # The two source edits RoboTwin's scripts/_install.sh makes (UTF-8 URDF reads; mplib
  # screw plans that touch an object are allowed, as RoboTwin's grasps need).
  local site
  site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$site/sapien/wrapper/urdf_loader.py"
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "$site/mplib/planner.py"
  # RoboTwin's cuRobo v0.7.8, built in place for RoboTwin processes only.
  bash "$ROOT/environment/robotwin_curobo.sh"
}

step_project() {
  info "harness-policy (editable)"
  pip_c -e "$ROOT"
}

step_skills() {
  info "Learned-skill and perception inference dependencies"
  pip_c -r "$ROOT/environment/requirements-skills.txt"
  pip_c --no-deps mmengine==0.10.7
  need_robodojo
  # G0.5 declares python<3.11 and pins training/sim stacks; inference runs here.
  pip_c --no-deps --ignore-requires-python -e "$ROBODOJO_ROOT/XPolicyLab/policy/G05/G05"
  # OpenWAM world-action model for predict_future (training dependencies left out).
  pip_c --no-deps -e "$ROBODOJO_ROOT/XPolicyLab/policy/OpenWAM/OpenWAM"
  # SAM 3 for segment_text; see third_party/sources.toml for the revision.
  if [ -d "${SAM3_ROOT:-}" ]; then
    pip_c --no-deps -e "$SAM3_ROOT"
  else
    info "SAM3_ROOT not set; skipping the sam3 package (see third_party/sources.toml)"
  fi
}

step_docs() {
  info "Documentation site (mkdocs serve / mkdocs build --strict)"
  pip_c -r "$ROOT/environment/requirements-docs.txt"
}

step_verify() {
  info "Verifying the pinned core"
  python "$ROOT/environment/check.py"
}

step_lock() {
  info "Writing environment/lock/"
  mkdir -p "$ROOT/environment/lock"
  conda list --explicit > "$ROOT/environment/lock/conda-linux-64.txt"
  # Editable installs point at local checkouts; record them as comments.
  python -m pip freeze --exclude-editable > "$ROOT/environment/lock/pip-freeze.txt"
  python -m pip list --editable --format=freeze | sed 's/^/# editable: /' >> "$ROOT/environment/lock/pip-freeze.txt"
}

main() {
  local from="" only=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --from) from="$2"; shift 2 ;;
      --only) only="$2"; shift 2 ;;
      -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
      *) fail "Unknown argument: $1" ;;
    esac
  done
  for name in $from $only; do
    printf '%s\n' "${STEPS[@]}" | grep -qx "$name" || fail "Unknown step: $name"
  done
  activate "$([ -z "$from$only" ] || [ "$from$only" = conda ] && echo create || echo existing)"
  local started=0
  [ -z "$from" ] && started=1
  for name in "${STEPS[@]}"; do
    [ "$name" = "$from" ] && started=1
    if [ -n "$only" ]; then
      [ "$name" = "$only" ] && "step_$name"
    elif [ "$started" = 1 ]; then
      "step_$name"
    fi
  done
  info "Done. Activate with: source environment/activate.sh"
}

main "$@"
