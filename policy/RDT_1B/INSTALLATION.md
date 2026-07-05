# RDT_1B Installation

`install.sh` can install dependencies and optionally fetch/link weights, but this document is kept because RDT_1B has several weight-management modes and external Hugging Face assets.

## 1. One-command Install

```bash
cd XPolicyLab/policy/RDT_1B
bash install.sh
```

Useful modes:

```bash
RDT_SKIP_CONDA_CREATE=1 bash install.sh       # reuse an existing env
RDT_WEIGHTS_SRC=<path_to_weights_root> bash install.sh
RDT_SKIP_WEIGHTS=1 bash install.sh            # deps only
```

## 2. Manual Install Equivalent

```bash
conda create -n rdt_1b python=3.10 -y
conda activate rdt_1b

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging==24.0 ninja
pip install flash-attn==2.7.2.post1 --no-build-isolation

cd XPolicyLab/policy/RDT_1B/rdt
pip install -r requirements.txt

cd ../../..
pip install -e .
```

If the package index cannot resolve `tfds-nightly` or `tensorflow`, install those from the official PyPI index.

## 3. Download Model Weights

```bash
cd XPolicyLab/policy/RDT_1B
mkdir -p weights/RDT
huggingface-cli download google/t5-v1_1-xxl --local-dir weights/RDT/t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir weights/RDT/siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir weights/RDT/rdt-1b
```

## 4. Useful Variables

| Variable or path | Meaning |
| --- | --- |
| `weights/RDT/` | Default local weight root. |
| `RDT_HDF5_DIR` | Training data directory override. |
| `RDT_PRETRAINED_MODEL` | RDT model path or HF id. |
| `TEXT_ENCODER_NAME` | T5 text encoder path or HF id. |
| `VISION_ENCODER_NAME` | SigLIP vision encoder path or HF id. |

## 5. Evaluation Checkpoint Layout

```bash
cd XPolicyLab/policy/RDT_1B
mkdir -p checkpoints
ln -sfn <path_to_trained_ckpt> checkpoints/<ckpt_name>
```

Pass the complete checkpoint directory name as `ckpt_name` during evaluation.
