# RISE Installation

> 上游：[RISE](https://opendrivelab.com/rise/) · https://github.com/OpenDriveLab/RISE

数据转换、训练与评测见 [README.md](README.md)。

## 1. Conda 环境

```bash
cd XPolicyLab/policy/RISE
bash install.sh RISE
conda activate RISE
```

`install.sh` 会创建/激活 `RISE` 环境（Python 3.11.14）、安装 vendored `RISE/` 上游依赖，并以 editable 方式安装 `XPolicyLab`。

## 2. Pi0.5 预训练权重（install.sh 不包含）

训练默认读取 `policy/RISE/weights/pi05_base_pytorch/`（须含 `model.safetensors` 或 `model.pt`）。

### 方式 A：链接已有 PyTorch 权重（推荐）

```bash
cd XPolicyLab/policy/RISE
bash setup_weights.sh <path/to/pi05_base_pytorch>
```

### 方式 B：从 JAX `pi05_base` 转换

```bash
cd XPolicyLab/policy/RISE
conda activate RISE

OFFLINE_DIR="RISE/policy_and_value/policy_offline_and_value"
cd "${OFFLINE_DIR}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# 下载 JAX pi05_base，或设 JAX_CKPT 为本地目录（须含 params/）
JAX_CKPT=$(python -c "from openpi_value.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))")
# JAX_CKPT=<path/to/pi05_base>

python examples/convert_jax_model_to_pytorch.py \
  --config_name Pi05_base_convert \
  --checkpoint_dir "${JAX_CKPT}" \
  --output_path ../../../weights/pi05_base_pytorch \
  --precision bfloat16
```

说明：
- `--checkpoint_dir` 指向 **含 `params/` 的 JAX 根目录**，不是 `.../pi05_base/params`。
- 转换需 GPU 与足够内存；已有 PyTorch 权重时优先用方式 A。
- `train.sh` 默认从 `weights/pi05_base_pytorch/` 加载；可用 `RISE_PYTORCH_WEIGHT_PATH` 覆盖。
