#!/usr/bin/env bash
# Provision a repository-local snapshot from a working host RoboDojo installation.
# Supply a source checkout with complete Assets and initialized third_party trees.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$(realpath "${1:?usage: prepare_robodojo.sh SOURCE_CHECKOUT SOURCE_CONDA_PREFIX}")"
source_env="$(realpath "${2:?missing source conda prefix}")"
target_root="$repo_root/vendor/robodojo"
target_env="$repo_root/.venv-robodojo"
mkdir -p "$repo_root/vendor"
if [[ "$source_root" == "$target_root" || "$source_env" == "$target_env" ]]; then
    echo "Source and destination must be independent installations" >&2
    exit 1
fi
if [[ -e "$target_root" && ! -f "$target_root/.embodiedevo-provisioned.json" ]]; then
    echo "Existing vendor snapshot has no EmbodiedEvo provenance marker; use a fresh checkout for provisioning." >&2
    exit 1
fi
if [[ ! -x "$target_env/bin/python" ]]; then
    conda create --prefix "$target_env" --clone "$source_env" --offline -y
fi
if [[ ! -f "$target_root/.embodiedevo-provisioned.json" ]]; then
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='eval_result' --exclude='smoke_results' --exclude='logs' \
      --exclude='XPolicyLab/policy' --exclude='outputs' \
      "$source_root/" "$target_root/"
    "$target_env/bin/python" "$repo_root/scripts/relocate_robodojo.py" \
      --source-root "$source_root" --source-env "$source_env"
fi
"$target_env/bin/python" -m pip install --no-deps packaging==23.0
"$target_env/bin/python" -m pip install --no-deps --no-build-isolation -e "$repo_root"
echo "RoboDojo prepared: $target_env/bin/python"
echo "Run scripts.smoke_robodojo before starting an agent."
