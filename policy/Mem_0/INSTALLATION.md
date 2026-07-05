# Mem_0 Installation

Conda only (no Docker). Data conversion, training, and evaluation: [README.md](README.md).
This document is kept because Mem_0 uses separate execution, planning-training,
and planning-inference environments plus checkpoint downloads that are not all
covered by the main `install.sh`.

## 1. Execution / inference env (`mem0`)

```bash
cd policy/Mem_0
bash install.sh mem0
```

`install.sh` creates the `mem0` conda env (Python 3.10, PyTorch 2.6 + CUDA 12.4), installs `Mem_0/requirements.txt`, flash-attn, ffmpeg, and editable `XPolicyLab`.

## 2. Planning training env (`llama_factory`, Mn tasks)

```bash
cd policy/Mem_0
bash install_planning.sh
```

Creates `llama_factory` (Python 3.11), clones [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) into `Mem_0/LlamaFactory` when missing, and runs `pip install -e` plus metrics/wandb deps.

Optional before planning train: `wandb login`.

## 3. Planning inference env (`vllm`, Mn eval)

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```

Mn eval serves merged planning weights via vLLM (`eval.sh` optional 11th arg `planning_gpu_ids`, or manual `vllm serve`). Override env name with `CONDA_ENV_VLLM`.

## 4. Backbone checkpoints (not performed by install.sh)

```bash
cd policy/Mem_0/Mem_0/checkpoints
python _download.py
```

| Asset | Local path | Command |
| --- | --- | --- |
| Qwen3-VL-2B-Instruct (execution) | `Mem_0/checkpoints/Qwen3-VL-2B-Instruct` | `python _download.py --model 2b` |
| Qwen3-VL-8B-Instruct (planning) | `Mem_0/checkpoints/Qwen3-VL-8B-Instruct` | `python _download.py --model 8b` |
| Both | above dirs | `python _download.py` (default) |

Requires `huggingface_hub` (`pip install huggingface_hub` if `_download.py` fails).
