#!/usr/bin/env bash
set -euo pipefail
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$code_root/FULL_TRAIN.sh" smoke
