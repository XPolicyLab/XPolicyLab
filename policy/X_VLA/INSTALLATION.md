# X_VLA 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n XVLA python=3.10 -y
conda activate XVLA
```

### 2. 安装 X-VLA

```bash
cd xvla
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `XVLA_MODEL_PATH` | 预训练权重（HF repo id 或本地目录；`train.sh` 默认） |
| `XVLA_META_PATH` | 训练 metadata JSON（默认 `xvla/meta.json`） |

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `XVLA` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `XVLA_sim_arx-x5` |
| expert_data_num | `3500` |
| action_type | `ee` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/X_VLA/XVLA_sim_arx-x5` |
| 备注 | shared/X-VLA-Pt |

软链 checkpoint（在 `policy/X_VLA/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls XVLA_sim_arx-x5 arx_x5 3500 ee 0 0 XVLA <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls XVLA_sim_arx-x5 arx_x5 ee 0 0 XPolicyLab "ckpt_name=XVLA_sim_arx-x5,action_type=ee" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

