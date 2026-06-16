# InternVLA_A1 环境配置

上游详细说明见 [internvla_a1/tutorials/installation.md](internvla_a1/tutorials/installation.md)。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境并安装 InternVLA

```bash
conda create -n internvla_a1 python=3.10 -y
conda activate internvla_a1

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf

cd internvla_a1
pip install -e .

TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/
cp -r src/lerobot/policies/pi0/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models "${TRANSFORMERS_DIR}"
```

### 2. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `PRETRAINED_PATH` | 在 `internvla_a1/launch/internvla_a1_3b_finetune.sh` 中设置，可为 HF id 或本地目录 |
| `INTERNVLA_REPO_ID` | LeRobot 数据集 repo id（`train.sh` 可覆盖） |

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `internvla_a1` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo_sim_seed_0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/InternVLA_A1/RoboDojo_sim_seed_0` |
| 备注 | shared/Qwen3-VL-2B-Instruct、shared/Cosmos-Tokenizer-CI8x8 见 checkpoints/shared/ |

软链 checkpoint（在 `policy/InternVLA_A1/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo_sim_seed_0 arx_x5 3500 joint 0 0 internvla_a1 <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo_sim_seed_0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo_sim_seed_0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

