#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -e "${SCRIPT_DIR}/../.."
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install optional accelerators independently so one unavailable CUDA extension
# does not prevent the remaining kernels (or the PyTorch fallback) from working.
if ! python -m pip install ninja packaging psutil; then
    echo "[CogACT2] Optional build helpers could not be installed; native accelerators may need prebuilt wheels." >&2
fi
for package in flash-attn==2.8.3.post1 "flash-attn-4[cu13]==4.0.0b32" flash-linear-attention==0.5.2 causal-conv1d==1.7.0; do
    if python -m pip install --no-build-isolation -c "${SCRIPT_DIR}/requirements.txt" "$package"; then
        if [[ "$package" == flash-attn-4* ]]; then
            # FA2 and FA4 share flash_attn/. Restore FA4 even if it was already
            # installed before FA2; do not reinstall the large Torch dependencies.
            if ! python -m pip install --force-reinstall --no-deps "$package"; then
                echo "[CogACT2] Could not restore FA4 after FA2; check the selected backend at startup." >&2
            fi
        fi
    else
        echo "[CogACT2] Optional accelerator $package could not be installed; runtime will select an available fallback." >&2
    fi
done
