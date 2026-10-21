#!/usr/bin/env bash
# Usage: bash install.sh /path/to/policy/python
# Installs XPolicyLab and runwu_rsi into the policy server's Python environment.
# runwu_rsi comes from, in order:
#   1. $RUNWU_RSI_SRC: a local RunWu-RSI checkout, or a pip requirement such as
#      "runwu-rsi[runwu] @ https://github.com/RunWu-RSI/RunWu-RSI/archive/<ref>.tar.gz";
#   2. the RunWu-RSI checkout this directory is linked from, if any;
#   3. the source archive of https://github.com/RunWu-RSI/RunWu-RSI at $RUNWU_RSI_REF (a tag or commit; default: v0.1.0).
# The default is an archive, not "git+https://...": pip's git installs also clone every submodule of the repository
# (RoboDojo, IsaacLab, cuRobo), which are gigabytes that the policy server does not need.
set -euo pipefail
policy_python=${1:?Usage: install.sh /path/to/policy/python}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xpl_root="$(cd "${script_dir}/../.." && pwd)"
rsi_root="$(cd "$(cd "${script_dir}" && pwd -P)/.." && pwd)"
rsi_url="https://github.com/RunWu-RSI/RunWu-RSI"
if [[ -n "${RUNWU_RSI_SRC:-}" ]]; then
    if [[ -d "${RUNWU_RSI_SRC}" ]]; then rsi_package=(-e "${RUNWU_RSI_SRC}[runwu]"); else rsi_package=("${RUNWU_RSI_SRC}"); fi
elif grep -q '^name = "runwu-rsi"' "${rsi_root}/pyproject.toml" 2>/dev/null; then
    rsi_package=(-e "${rsi_root}[runwu]")
else
    rsi_package=("runwu-rsi[runwu] @ ${rsi_url}/archive/${RUNWU_RSI_REF:-v0.1.0}.tar.gz")
fi
packages=(-e "${xpl_root}" "${rsi_package[@]}")
if "$policy_python" -m pip --version >/dev/null 2>&1; then
    exec "$policy_python" -m pip install "${packages[@]}"
fi
exec uv pip install --python "$policy_python" "${packages[@]}"
