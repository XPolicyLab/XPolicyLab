# DreamZero

本目录是 DreamZero 在 XPolicyLab 中的接入层。DreamZero 原项目源码在 `dreamzero/`，训练、评测、数据和环境适配尽量放在当前目录顶层。

## 一键环境

默认环境名是 `dreamzero_robodojo`：

```bash
cd policy/DreamZero
bash install.sh
conda activate dreamzero_robodojo
```

`install.sh` 会安装 DreamZero、`lerobot>=0.4.0` 和 XPolicyLab。`flash-attn`、`transformer_engine` 默认不装，按机器情况打开。

B200 机器建议直接用 CUDA 12.9 对应的 PyTorch 源，并一次性打开 `flash-attn` 和 `transformer_engine`：

```bash
DREAMZERO_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129 \
INSTALL_FLASH_ATTN=1 \
INSTALL_TRANSFORMER_ENGINE=1 \
MAX_JOBS=8 \
bash install.sh dreamzero_robodojo
```

如果后续要跑 TensorRT 加速推理，再额外安装：

```bash
conda activate dreamzero_robodojo
pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 transformer_engine_torch==2.10.0
```

## 默认路径

默认训练数据是多任务 LeRobot v3：

模型、Wan 组件、tokenizer 和训练产物默认都放在当前目录的 `checkpoints/` 下。推荐使用子目录布局；如果历史上把 DreamZero-AgiBot 直接放在 `checkpoints/` 根目录，训练和推理也会兼容。

```bash
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot

pip install "huggingface_hub[cli]"

# You may need to set your HuggingFace token:
# export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

# Download Wan2.1 model weights (~28GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download umt5-xxl tokenizer
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

常用路径都可以覆盖：

```bash
export LEROBOT_DATA_PATH=/path/to/RoboDojo_sim_arx-x5_v30
export DREAMZERO_PRETRAINED_MODEL_PATH=/path/to/DreamZero-AgiBot
export WAN_CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/path/to/umt5-xxl
```

## 多任务训练

常用训练变量：

```bash
export DREAMZERO_NATIVE_DOJO_ACTION=1 #启动这个则用新的MLP训练
export DREAMZERO_MAX_STEPS=100000
export DREAMZERO_SAVE_STEPS=10000
export DREAMZERO_NUM_GPUS=8
export DREAMZERO_PER_DEVICE_BATCH_SIZE=4
export DREAMZERO_REPORT_TO=tensorboard
export WANDB_PROJECT=dreamzero
```

```bash
conda activate dreamzero_robodojo

bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
```

训练产物目录：

```text
policy/DreamZero/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42
```

## 同机评测

`task_name` 只在评测时表示要跑的仿真任务；权重仍用多任务 `ckpt_name=cotrain`：

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 dreamzero_robodojo XPolicyLab
```

指定 checkpoint：

```bash
export MODEL_PATH=/path/to/checkpoint-xxxxx
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 dreamzero_robodojo XPolicyLab
```

`MODEL_PATH` 可以是具体 `checkpoint-*`，也可以是包含 `checkpoint-*` 的训练输出目录。不设置时会按 6 元组自动找 latest，再回退到 `deploy.yml` 里的 `pretrained_model_path`。

## 双机评测

policy 机器启动 server，默认绑定 `0.0.0.0`：

```bash
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 \
  0 dreamzero_robodojo 5000 0.0.0.0
```

环境机器启动 client，把最后一个参数换成 policy 机器 IP：

```bash
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls cotrain arx_x5 joint 42 \
  0 XPolicyLab "ckpt_name=cotrain,action_type=joint" \
  5000 <policy_server_ip>
```

`deploy.yml` 里的 `eval_env` 控制 `sim`、`real`、`debug`，默认是 `sim`。

## 注意事项

- DreamZero 首次推理会 warmup/compile，client timeout 默认扩到 1800 秒，可用 `DREAMZERO_MODEL_CLIENT_TIMEOUT` 覆盖。
- `deploy.yml` 的 `inference_method` 默认是 `lazy_joint_forward_causal`，也可改为 `forward` 或 `lazy_joint_forward`。
- v3 数据当前是 14 维双臂关节，适配层会补成 DreamZero-AgiBot 的 20 维 state 和 22 维 action；缺失的夹爪、head、waist、base velocity 用 0 填充。
