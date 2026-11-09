#!/usr/bin/env bash
set -euo pipefail

policy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "$policy_dir/../.." && pwd)"
source_dir="$policy_dir/OrimaPI"

python -m pip install -c "$source_dir/configs/code-policy-constraints.txt" \
  -e "$xpl_root" -e "$source_dir[code-skills,code-depth,xpolicylab]"
