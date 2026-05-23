#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/tinyvla"

pip install -r requirements.txt
pip install -e .

cd policy_heads
pip install -e .

# install llava-pythia
cd ../llava-pythia
pip install -e .