#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A1_DIR="${SCRIPT_DIR}/A1"

echo -e "\033[33m[A1 Install] Installing A1 package...\033[0m"
cd "${A1_DIR}"
pip install -e .[all]
pip install -r requirements.txt

echo -e "\033[33m[A1 Install] Installing XPolicyLab package...\033[0m"
cd "${SCRIPT_DIR}/../.."
pip install -e .

echo -e "\033[33m[A1 Install] Installation complete.\033[0m"
