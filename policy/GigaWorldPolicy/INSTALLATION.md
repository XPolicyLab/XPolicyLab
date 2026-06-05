# GigaWorldPolicy 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n gigaworld-policy python=3.11 -y
conda activate gigaworld-policy
```

### 2. 安装 GigaWorld 依赖

```bash
cd giga_world_policy
pip install -e ./third_party/giga-train
pip install -e ./third_party/giga-models
pip install -e ./third_party/giga-datasets
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `GIGAWORLD_DATA_DIR` | LeRobot 训练数据目录 |
| `GIGAWORLD_NORM_PATH` | `norm_stats_delta.json` 路径 |
| `GIGAWORLD_PRETRAINED_PATH` | Wan2.2 等预训练权重目录或 HF 缓存 |
| `GIGAWORLD_BASE_CONFIG` | 基础训练 JSON 配置 |

## 训练与评测

详见 [README.md](README.md)。内层说明见 [giga_world_policy/Readme.md](giga_world_policy/Readme.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `gigaworld-policy` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo_sim_arx_seed_0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/GigaWorldPolicy/RoboDojo_sim_arx_seed_0/checkpoint_epoch_4_step_100000_old` |

软链 checkpoint（在 `policy/GigaWorldPolicy/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo_sim_arx_seed_0 arx_x5 3500 joint 0 0 gigaworld-policy <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo_sim_arx_seed_0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo_sim_arx_seed_0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

