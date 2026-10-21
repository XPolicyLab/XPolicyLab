#!/usr/bin/env bash
# Usage: bash download_checkpoint.sh [task|all] [ref]
# Downloads the released RunWu-RSI checkpoint (ckpt_name=RunWu: one <bundle id>.json per task, plus demos/ where a task
# uses a demonstration) from https://github.com/RunWu-RSI/RunWu-RSI into checkpoints/RunWu/<task>/.
# <task> is a task name (for example stack_bowls) or "all" (default) for the 42 tasks. [ref] is a tag or commit
# (default: $RUNWU_RSI_REF, else v0.1.0). Only the requested directories are fetched (shallow, sparse git checkout).
set -euo pipefail
task=${1:-all}
ref=${2:-${RUNWU_RSI_REF:-v0.1.0}}
repo_url=${RUNWU_RSI_REPO:-https://github.com/RunWu-RSI/RunWu-RSI.git}
if [[ ! "${task}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[download] invalid task: ${task}" >&2
    exit 1
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
released="xpolicylab_policy/checkpoints/RunWu"
if [[ "${task}" == "all" ]]; then pattern="/${released}/"; else pattern="/${released}/${task}/"; fi
git -C "${workdir}" init -q
git -C "${workdir}" remote add origin "${repo_url}"
git -C "${workdir}" sparse-checkout set --no-cone "${pattern}"
git -C "${workdir}" fetch -q --depth 1 --filter=blob:none origin "${ref}"
git -C "${workdir}" checkout -q FETCH_HEAD
source_dir="${workdir}/${released}"
if [[ ! -d "${source_dir}" ]] || [[ -z "$(ls -A "${source_dir}")" ]]; then
    echo "[download] no checkpoint for ${task} at ${ref} in ${repo_url}" >&2
    exit 1
fi
mkdir -p "${script_dir}/checkpoints/RunWu"
for dir in "${source_dir}"/*/; do
    name="$(basename "${dir}")"
    target="${script_dir}/checkpoints/RunWu/${name}"
    if [[ -e "${target}" || -L "${target}" ]]; then
        echo "[download] ${target} exists, skipping (remove it to download again)" >&2
        continue
    fi
    cp -R "${dir%/}" "${target}"
    echo "[download] ${name} -> ${target}"
done
