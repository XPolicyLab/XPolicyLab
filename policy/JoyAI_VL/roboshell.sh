#!/usr/bin/env bash
# HexaShell-preview entry point. Run from anywhere; settings come from config.env next to this file.
#   ./roboshell.sh setup                      build the agent image and install the policy-server deps
#   ./roboshell.sh check                      verify source, manifests and syntax
# The official evaluation is eval.sh in this directory (XPolicyLab adapter entry).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cmd=${1:-help}; shift || true
case "$cmd" in
  setup)
    [ -f "$ROOT/config.env" ] || { echo "copy config.env.example to config.env and edit it first"; exit 1; }
    "$ROOT/agent/api/build.sh"
    "$ROOT/scripts/install_policy_deps.sh"
    source "$ROOT/scripts/sim_env.sh"
    echo "setup done" ;;
  check) exec python3 "$ROOT/tools/verify_release.py" "$@" ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac
