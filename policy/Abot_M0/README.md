# Abot_M0

ABot-M0 在 XPolicyLab 中的封装。完整安装见 [INSTALLATION.md](INSTALLATION.md) 与 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md)。

## 数据准备

```bash
cd abot_m0
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

cp examples/Robotwin/train_files/modality.json \
   "${HF_LEROBOT_HOME}/<repo_id>/meta/modality.json"

python3 examples/RoboDojo/prepare_RoboDojo_abot.py \
  --dataset-dir "${HF_LEROBOT_HOME}/<repo_id>"
```

## 训练

设置权重环境变量后启动（示例）：

```bash
cd abot_m0
conda activate ABot

BASE_VLM=<hf_or_local_qwen_path> \
PRETRAIN_CKPT=<path_to_ABot_M0_Pretrain.pt> \
RELOAD_MODULES=qwen_vl_interface \
bash examples/RoboDojo/train_files/run_robodojo_train.sh
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `BASE_VLM` | VLM 权重 |
| `PRETRAIN_CKPT` | 预训练 checkpoint |
| `RELOAD_MODULES` | 从预训练加载的模块子集 |
| `HF_LEROBOT_HOME` | LeRobot 数据根目录 |

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-sim-arx_x5-100-joint-0 arx_x5 100 joint 0 <policy_gpu> <env_gpu> ABot XPolicyLab
```
