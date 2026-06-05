# MolmoAct2 环境配置

MolmoAct2 在 XPolicyLab 中有两类 Python 环境：

| 环境 | 目录 | 用途 |
| --- | --- | --- |
| **XPolicyLab 训练 / 评测（推荐）** | `molmoact2/lerobot/.venv` | `lerobot_train`、`eval.sh`、`model.py` 推理 |
| 上游 FastAPI Server（可选） | `molmoact2/.venv` | 官方 `examples/droid/`、`examples/yam/` server |

**RoboDojo 训练与 XPolicyLab 评测共用 `molmoact2/lerobot/.venv`，无需单独推理 venv。**

> `molmoact2/` 不在 Git 中，首次请运行 `bash install.sh` 本地 clone。

## 一键安装

```bash
bash install.sh          # lerobot 训练环境 + XPolicyLab（RoboDojo 默认）
bash install.sh all      # 上述 + 上游 FastAPI 推理 venv
bash install.sh infer    # 仅上游 FastAPI 推理 venv（非 XPolicyLab eval）
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

### 2. XPolicyLab 训练 / 评测环境（必需）

RoboDojo 训练与 `eval.sh` 推理均使用此 venv：

```bash
cd molmoact2/lerobot
UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
```

### 3. 安装 XPolicyLab

```bash
cd molmoact2/lerobot
source .venv/bin/activate
cd ../../..
pip install -e .    # 若 venv 无 pip，可先 python -m ensurepip
pip install h5py opencv-python
```

### 4. 上游 FastAPI 推理环境（可选，非 XPolicyLab eval）

仅在使用官方 DROID/YAM server 时需要：

```bash
cd molmoact2
uv sync
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2
```

### 5. 训练环境（与第 2 步相同，高级手动步骤）

```bash
cd molmoact2/lerobot
UV_LINK_MODE=copy uv pip install -e ".[molmoact2,training,scipy-dep]" --index-strategy unsafe-best-match
```

下载起点权重（可选）：

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run huggingface-cli download allenai/MolmoAct2
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `MOLMOACT2_CHECKPOINT_PATH` | 训练起点（默认 HF `allenai/MolmoAct2`） |
| `MOLMOACT2_DATASET_ROOT` | LeRobot v3.0 数据集根目录 |
| `MOLMOACT2_DATASET_REPO_ID` | 数据集 repo id |
| `MOLMOACT2_OUTPUT_ROOT` | 训练输出根目录 |
| `MOLMOACT2_LOCAL_CACHE_ROOT` | 本机 HF datasets 缓存（多机训练防 NFS 锁竞争，默认 `/tmp/molmoact2-cache-$(hostname)`） |
| `SKIP_XPOLICYLAB=1` | `install.sh` 时跳过 XPolicyLab |

## 常见错误

| 现象 | 处理 |
| --- | --- |
| `get_policy_class('molmoact2')` 失败 | 使用 `molmoact2/lerobot/.venv`，非 `molmoact2/.venv` |
| `import XPolicyLab` 失败 | 在 `lerobot/.venv` 中 `pip install -e .`（见上文第 3 步） |
| transformers 冲突 | XPolicyLab eval 与训练均用 `lerobot/.venv`；FastAPI server 才用 `molmoact2/.venv` |
| `torchcodec` 版本冲突 | 安装时加 `--index-strategy unsafe-best-match` |
| 多机训练 dataloader 很慢 | 确认 `train.sh` 已设置本机 `HF_DATASETS_CACHE`；可设 `MOLMOACT2_LOCAL_CACHE_ROOT` 到有足够空间的本地盘 |

训练与评测入口见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `uv` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-3500-joint-0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/MolmoACT2/RoboDojo-cotrain-arx_x5-3500-joint-0` |
| 备注 | server 脚本内联代理；首次 HF 下载较慢 |

软链 checkpoint（在 `policy/MolmoACT2/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 0 uv <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-3500-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

