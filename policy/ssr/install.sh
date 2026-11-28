set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128
