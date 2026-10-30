#!/usr/bin/env bash
# Shared interpreter selection for install.sh and wam/install.sh.
#
# One environment serves both the policy server and the WAM backend, so both
# installers must resolve the same interpreter. Sourced, not executed; defines:
#
#   XPL_PYTHON      interpreter to install into and to run the import check with
#   XPL_ENV_DIR     its prefix directory, when it is a virtual environment
#   XPL_CREATED     1 when this call created the environment
#
# Precedence (first match wins):
#   1. PYTHON_BIN            an interpreter or environment the caller already has
#   2. GEACT_ENV_DIR         an environment directory to create or reuse
#   3. <XPolicyLab>/.venv-geact
#
# The base interpreter for step 2/3 is XPL_BASE_PYTHON, then GEACT_BASE_PYTHON,
# then python3; the WAM pins need 3.11 or 3.12, so an unusable base is rejected
# here rather than halfway through a pip install.

_xpl_pick_python() {
  local root="$1" base candidate

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    [[ -x "$PYTHON_BIN" ]] || {
      echo "[error] PYTHON_BIN=$PYTHON_BIN is not executable" >&2
      return 1
    }
    XPL_PYTHON="$PYTHON_BIN"
    XPL_ENV_DIR="$(cd "$(dirname "$PYTHON_BIN")/.." 2>/dev/null && pwd || echo "")"
    XPL_CREATED=0
    return 0
  fi

  XPL_ENV_DIR="${GEACT_ENV_DIR:-$root/.venv-geact}"
  XPL_PYTHON="$XPL_ENV_DIR/bin/python"
  XPL_CREATED=0
  if [[ ! -x "$XPL_PYTHON" ]]; then
    base="${XPL_BASE_PYTHON:-${GEACT_BASE_PYTHON:-python3}}"
    command -v "$base" >/dev/null || {
      echo "[error] base interpreter not found: $base" >&2
      return 1
    }
    "$base" -m venv "$XPL_ENV_DIR" || return 1
    XPL_CREATED=1
  fi
  return 0
}

_xpl_check_python_version() {
  local version
  version="$("$XPL_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || {
    echo "[error] cannot run $XPL_PYTHON" >&2
    return 1
  }
  case "$version" in
    3.11|3.12) return 0 ;;
    *)
      echo "[error] $XPL_PYTHON is Python $version; the inference pins need 3.11 or 3.12" >&2
      echo "        set PYTHON_BIN to a 3.11/3.12 interpreter, or set XPL_BASE_PYTHON/GEACT_BASE_PYTHON" >&2
      echo "        to one and remove ${XPL_ENV_DIR:-the target environment} so the installer builds it" >&2
      return 1
      ;;
  esac
}
