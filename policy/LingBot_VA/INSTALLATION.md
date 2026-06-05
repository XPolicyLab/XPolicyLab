# LingBot_VA 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n lingbot_va python=3.10.6 -y
conda activate lingbot_va

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### 2. 安装源码与 XPolicyLab

```bash
cd lingbot_va
pip install -e .

cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `LINGBOT_VA_DATASET_PATH` | LeRobot 训练数据完整目录 |
| `LINGBOT_VA_CONFIG_NAME` | 训练配置名（默认 `robotwin30_train`） |
| Wan 权重 | 数据处理脚本 `--model-root` 指向本地 Wan2.2 目录或 HF 缓存 |

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `lingbot_va` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-3500-joint-0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Lingbot_VA/robodojo_sim_arx_x5_v21/checkpoints` |

软链 checkpoint（在 `policy/LingBot_VA/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 0 lingbot_va <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-3500-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

