# MolmoAct2 环境配置

MolmoAct2 需要 **两套独立的 uv 虚拟环境**：

| 环境 | 目录 | 用途 |
| --- | --- | --- |
| 推理 Server | `molmoact2/.venv` | FastAPI 官方 server |
| 训练 LeRobot | `molmoact2/lerobot/.venv` | `lerobot_train` / XPolicyLab 集成 |

> `molmoact2/` 不在 Git 中，首次请运行 `bash install.sh` 本地 clone。

## 一键安装

```bash
bash install.sh          # 训练环境 + XPolicyLab（RoboDojo 推荐）
bash install.sh all      # 推理 + 训练
bash install.sh infer    # 仅推理 server
```

## 手动安装

### 0. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 初始化上游源码

```bash
cd molmoact2
git submodule update --init --recursive
# 若 submodule 为空：
# git clone -b molmoact2-policy https://github.com/allenai/lerobot lerobot
```

### 2. 推理环境（可选）

```bash
cd molmoact2
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2
```

### 3. 训练环境

```bash
cd molmoact2/lerobot
UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
```

下载起点权重（可选）：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run huggingface-cli download allenai/MolmoAct2
```

### 4. 安装 XPolicyLab

```bash
cd molmoact2/lerobot
source .venv/bin/activate
cd ../../..
uv pip install -e .
uv pip install h5py opencv-python
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `MOLMOACT2_CHECKPOINT_PATH` | 训练起点（默认 HF `allenai/MolmoAct2`） |
| `MOLMOACT2_DATASET_ROOT` | LeRobot v3.0 数据集根目录 |
| `MOLMOACT2_DATASET_REPO_ID` | 数据集 repo id |
| `MOLMOACT2_OUTPUT_ROOT` | 训练输出根目录 |
| `SKIP_XPOLICYLAB=1` | `install.sh` 时跳过 XPolicyLab |

## 常见错误

| 现象 | 处理 |
| --- | --- |
| `get_policy_class('molmoact2')` 失败 | 使用 `lerobot/.venv`，非 `molmoact2/.venv` |
| transformers 冲突 | 保持两个 venv 分离 |
| `torchcodec` 版本冲突 | 安装时加 `--index-strategy unsafe-best-match` |

训练入口见 [README.md](README.md)。
