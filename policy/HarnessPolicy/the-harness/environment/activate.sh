# Source this file to enter the project environment:
#   source environment/activate.sh
# Machine-specific settings (API keys, HF_ENDPOINT, TMPDIR) belong in
# ~/.harness_env, which is sourced last and never committed.

_harness_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_harness_conda="${CONDA_EXE:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}"
# shellcheck disable=SC1091
source "$("$_harness_conda" info --base)/etc/profile.d/conda.sh"
conda activate "${HARNESS_ENV:-harness-policy}"

export HARNESS_ROOT="$_harness_root"
export HARNESS_ROBODOJO_ROOT="${HARNESS_ROBODOJO_ROOT:-$_harness_root/third_party/RoboDojo}"
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
# The docs pin MkDocs 1.6 (environment/requirements-docs.txt); its plugins' MkDocs 2.0 notice is moot.
export DISABLE_MKDOCS_2_WARNING=true
if [ -f "$HOME/.harness_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.harness_env"
fi
unset _harness_root _harness_conda
