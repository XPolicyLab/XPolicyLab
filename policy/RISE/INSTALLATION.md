# RISE Installation

Upstream: [RISE](https://opendrivelab.com/rise/), https://github.com/OpenDriveLab/RISE.

`install.sh` handles the conda environment and editable installs, but this document is kept because RISE requires Pi0.5 pretrained weights that are intentionally not downloaded by the installer.

## 1. Install the Environment

```bash
cd XPolicyLab/policy/RISE
bash install.sh RISE
conda activate RISE
```

`install.sh` creates or activates the `RISE` conda environment with Python 3.11.14, installs the vendored `RISE/` dependencies, and installs XPolicyLab in editable mode.

## 2. Pi0.5 Pretrained Weights

Training defaults to `policy/RISE/weights/pi05_base_pytorch/`, which must contain `model.safetensors` or `model.pt`.

### Option A: Link Existing PyTorch Weights

```bash
cd XPolicyLab/policy/RISE
bash setup_weights.sh <path/to/pi05_base_pytorch>
```

### Option B: Convert from JAX `pi05_base`

```bash
cd XPolicyLab/policy/RISE
conda activate RISE

OFFLINE_DIR="RISE/policy_and_value/policy_offline_and_value"
cd "${OFFLINE_DIR}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

JAX_CKPT=$(python -c "from openpi_value.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))")
# Or set JAX_CKPT=<path/to/pi05_base>, where the directory contains params/.

python examples/convert_jax_model_to_pytorch.py \
  --config_name Pi05_base_convert \
  --checkpoint_dir "${JAX_CKPT}" \
  --output_path ../../../weights/pi05_base_pytorch \
  --precision bfloat16
```

Use a GPU machine with enough memory for conversion. If you already have PyTorch weights, prefer option A. Override the default training weight path with `RISE_PYTORCH_WEIGHT_PATH`.
