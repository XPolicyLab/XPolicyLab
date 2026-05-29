# Mem_0 Installation

Conda only (no Docker). Two environments: `mem0` (execution module, data,
inference) and `llama_factory` (planning module, Mn tasks).

## 1. Execution / inference env

```bash
cd policy/Mem_0
bash install.sh mem0
```

## 2. Backbone checkpoints

```bash
cd Mem_0/checkpoints
python _download.py     # Qwen3-VL-2B-Instruct (execution) + Qwen3-VL-8B-Instruct (planning)
```

## 3. Planning env (Mn tasks only)

```bash
conda create -n llama_factory python=3.11 -y
conda activate llama_factory
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git Mem_0/LlamaFactory
pip install -e Mem_0/LlamaFactory
pip install -r Mem_0/LlamaFactory/requirements/metrics.txt wandb
```

vLLM serving env (planning inference):

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm
```
