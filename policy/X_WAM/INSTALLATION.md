# X_WAM 环境配置

X-WAM 是统一的 4D 世界动作模型，依赖 Wan2.2-TI2V-5B 视频基座权重。详见 [X-WAM/README.md](X-WAM/README.md)。

## 手动安装

### 1. 创建环境

```bash
conda create -n XWAM python=3.10 -y
conda activate XWAM
```

### 2. 安装 X-WAM

关键依赖版本（已验证）：

- `python>=3.10`
- `torch>=2.4.0`（测试于 2.8.0+cu129）
- `numpy<2`（测试于 1.23.5）
- `diffusers>=0.31.0`（测试于 0.38.0）
- `transformers>=4.49.0,<=4.51.3`（测试于 4.51.3）
- `flash-attn`（测试于 2.8.3）

```bash
cd X-WAM

# 1. 安装 PyTorch >= 2.4.0（示例）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 FlashAttention（需联网）
pip install flash-attn --no-build-isolation
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

X-WAM 的 checkpoint 是实验目录（含 `config.yaml` + `checkpoints/<steps>.ckpt/...`），由 `exp_path` 指向；推理还需 Wan2.2-TI2V-5B 基座权重。

| 变量 | 说明 |
|------|------|
| `XWAM_EXP_PATH` | 实验目录（含 `config.yaml` 与 `checkpoints/`）。未设时由 `XWAM_CKPT_ROOT`/`checkpoints` + 6-tuple 拼出 |
| `XWAM_CKPT_ROOT` | checkpoint 根目录（默认 `policy/X_WAM/checkpoints`） |
| `XWAM_EXP_SETTING` | 实验目录名（默认 `dataset-ckpt-env_cfg-expert_num-action_type-seed` 6-tuple） |
| `XWAM_STEPS` | 选用的 checkpoint step（默认 `last`） |
| `XWAM_WAN_CHECKPOINT_DIR` | Wan2.2-TI2V-5B 基座权重（T5 + VAE + DiT）；未设时回落到 `config.yaml` |
| `XWAM_ALLOW_DUMMY_POLICY` | 设为 `true` 跳过权重加载，仅调试通信协议 |

基座权重见 [官方 Wan2.2 仓库](https://github.com/Wan-Video/Wan2.2)，checkpoint 与数据集见 [X-WAM-checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints)。

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

| 项 | 说明 |
|----|------|
| Server 环境 | `XWAM` |
| Client 环境 | `XPolicyLab`（conda） |
| action_type | `ee` |
| eval_env | `sim`（由 `deploy.yml` 控制，可切 debug / real） |

推理超参在 `deploy.yml` 中：`denoise_steps`（视频，默认 50）/ `action_denoise_steps`（动作，默认 10，异步去噪）、`cfg`、`replan_steps`（`null` = 执行整段预测）。

软链 checkpoint（在 `policy/X_WAM/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <exp_dir> checkpoints/<6-tuple_dir_name>
```

手动评测（`stack_bowls` 为示例任务，`arx_x5` 为 `env_cfg/arx_x5.yml`；`3500` 仅参与拼接 6-tuple 实验目录名，按实际 checkpoint 目录填）：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls <ckpt_name> arx_x5 3500 ee 0 0 XWAM <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls <ckpt_name> arx_x5 ee 0 0 XPolicyLab "ckpt_name=<ckpt_name>,action_type=ee" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。
