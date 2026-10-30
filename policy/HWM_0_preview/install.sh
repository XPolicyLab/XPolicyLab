#!/bin/bash
set -euo pipefail

# Run with the target Python environment already activated.
python -m pip install 'setuptools>=68' wheel packaging ninja
python -m pip install \
    'torch==2.6.0' 'torchvision==0.21.0' \
    'transformers==5.14.1' 'diffusers==0.39.0' 'accelerate==1.5.2' \
    sentencepiece pillow opencv-python numpy pyyaml h5py \
    'msgpack-numpy==0.4.8' 'websockets>=14.0' 'pydantic>=2.5' safetensors \
    'flash-linear-attention==0.3.2'
python -m pip install 'flash-attn==2.8.1' 'causal-conv1d==1.7.0' --no-build-isolation

# Some prebuilt extension wheels do not match the installed PyTorch C++ ABI.
if ! python -c 'import torch; import flash_attn; import causal_conv1d'; then
    echo 'Rebuilding attention extensions against the active PyTorch installation.'
    FLASH_ATTENTION_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
        python -m pip install --force-reinstall --no-deps --no-cache-dir \
        'flash-attn==2.8.1' 'causal-conv1d==1.7.0' --no-build-isolation
fi
python -c 'import torch; import flash_attn; import causal_conv1d'
