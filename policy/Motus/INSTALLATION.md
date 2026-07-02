# Motus 环境配置

Motus 上游源码位于 [motus/](motus/)，使用 conda 环境 `motus`。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建 conda 环境

```bash
conda create -n motus python=3.10 -y
conda activate motus

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install packaging psutil ninja wheel
pip install flash-attn --no-build-isolation
```

### 2. 安装 Motus 依赖

```bash
cd motus
pip install -r requirements.txt
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
pip install -e .
```

### 3. 安装 XPolicyLab

```bash
cd ../..
pip install -e .
```

## 模型与数据路径

| 变量 / 参数 | 说明 |
|-------------|------|
| `WAN_PATH` / `--wan_path` | 含 `Wan2.2-TI2V-5B`、`Qwen3-VL-2B-Instruct`、`Motus/` 的模型根目录 |
| `LEROBOT_DATA_ROOT` | LeRobot 数据集父目录（需指定具体子数据集 `root`） |
| LeRobot 直读 | 配置中填写 `repo_id` + `root=${LEROBOT_DATA_ROOT}/<dataset>` |

预训练组件通常包括：`Motus/`（Stage2）、`Wan2.2-TI2V-5B/`、`Qwen3-VL-2B-Instruct/`。

详细 LeRobot 训练与 T5 缓存流程见 [motus/README.md](motus/README.md)。

## 训练与评测

见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `motus` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-3500-joint-0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Motus/checkpoint_step_80000/pytorch_model` |

软链 checkpoint（在 `policy/Motus/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 0 motus <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-3500-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

