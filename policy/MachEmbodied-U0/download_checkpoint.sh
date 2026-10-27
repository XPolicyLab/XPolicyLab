#!/usr/bin/env bash
set -euo pipefail
hf download LiAuto-Foundation-Model/ME-U0-RoboDojo \
    --revision 89872d33817bb235aa1b1d33bbf18b4a211636e3 \
    --local-dir "${1}"
