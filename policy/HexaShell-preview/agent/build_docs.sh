#!/usr/bin/env bash
# Codex reads AGENTS.md. The manual itself is INTERFACE.md.
set -euo pipefail
cd "$(dirname "$0")"
python3 ../roboshell/docs.py build >/dev/null && cp build/AGENTS_base.md AGENTS.md
wc -c INTERFACE.md
