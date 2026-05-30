# Spirit_v15 环境配置

Spirit_v15 使用 `spirit_v15/` 上游源码，推荐 `uv` 管理环境。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 配置模型环境（uv）

```bash
cd spirit_v15
uv sync --extra train
source .venv/bin/activate
uv pip install -e .
```

### 1b. 配置模型环境（pip，无 uv 时）

```bash
cd spirit_v15
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-train.txt
pip install -e .
```

### 2. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `SPIRIT_PRETRAINED_PATH` | 预训练权重（本地目录或 HuggingFace repo id） |
| `SPIRIT_RAW_DATA_ROOT` | RoboDojo 原始 HDF5 根目录 |
| `XPOLICYLAB_DATA_ROOT` | XPolicyLab 数据根（转换脚本默认 `../../../data`） |
| `SPIRIT_CONVERTED_DATA_ROOT` | 转换后的 Spirit 训练目录 |
| `SPIRIT_PATTERNS_CSV` | 数据匹配 pattern，如 `RoboDojo.stack_bowls.arx_x5` |

## 训练与评测

先 `process_data.sh`，再 `train.sh`。详见 [README.md](README.md)。
