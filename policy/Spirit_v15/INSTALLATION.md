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

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `uv` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo_sim_arx-x5_seed_0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Spirit_v1.5/RoboDojo_sim_arx-x5_seed_0` |
| 备注 | deploy/sitecustomize.py 将 client socket 超时调至 600s |

软链 checkpoint（在 `policy/Spirit_v15/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 3500 joint 0 0 uv <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo_sim_arx-x5_seed_0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

