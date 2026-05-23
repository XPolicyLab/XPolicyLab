#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${SCRIPT_DIR}/H_RDT"
pip install -r requirements.txt

cd "${ROOT_DIR}"
pip install -e .
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}/H_RDT"
pip install -r requirements.txt

cd "${SCRIPT_DIR}/../.."
pip install -e .
