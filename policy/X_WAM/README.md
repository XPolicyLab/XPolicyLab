# X_WAM

X-WAM（Unified 4D World Action Modeling）已封装为 XPolicyLab policy。它以多视角 RGB 观测与当前机器人状态为输入，联合生成未来 4D 观测与机器人未来状态、动作，并通过异步噪声采样（ANS）快速解码动作。上游说明见 [X-WAM/README.md](X-WAM/README.md)，安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据准备

训练数据采用 X-WAM 数据结构（`metadata.json` + `data/` + `video/` + `depth/`），每个 episode 为一个 JSON，含 `instructions` / `observations` / `proprios` / `actions`，本体使用 EE 字段（`*_ee_pos` / `*_ee_rotm` / `*_gripper_pos`）。字段细节见 [X-WAM/README.md](X-WAM/README.md)。

将 RoboDojo（HDF5）数据集转换为该格式可使用：

```bash
python transform_robodojo_to_xwam.py \
    --input-dir  data/RoboDojo \
    --output-dir /path/to/sft_datasets/RoboDojo \
    --workers 16 [--limit N] [--clean]
```

### 关于深度数据

X-WAM 的 4D 建模会用到深度，但 RoboDojo 暂无深度信息（转换脚本中各相机的 `depth_path` 仅指向对应 RGB 视频占位）。有两种处理方式：

- **估计深度**：用 [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) 对 RGB 视频逐帧估计深度，写入 `depth/` 目录并让 `depth_path` 指向真实深度视频，再正常启用深度 loss。
- **禁用深度 loss**：在训练命令中设 `depth_loss_weight=0`（`wan22_5b_sft.yaml` 默认即为 `0.0`），跳过深度监督，先用 RGB + 动作训练。

## 训练

训练在 X-WAM 子目录中进行，读取 `configs/model/wan22_5b_sft.yaml`（模型与超参）与 `configs/data/{dataset}.yaml`（数据路径与归一化统计）：

```bash
cd X-WAM
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr=localhost --master_port=29500 \
    scripts/train_sft.py dataset=robodojo exp_name=<exp_name>
```

本目录当前提供的 dataset config 为 `configs/data/robodojo.yaml`，其 `dataset_path` 指向 `transform_robodojo_to_xwam.py` 的输出目录。`wan22_5b_sft.yaml` 默认 `dataset: robocasa`，本目录无对应 config，故训练时须显式传 `dataset=robodojo`。任意 config 字段均可在命令行用 OmegaConf 点号覆盖（如 `num_training_steps=40000`）。

训练前需准备好两类权重（路径在 `wan22_5b_sft.yaml` 中）：`wan_checkpoint_dir`（Wan2.2-TI2V-5B 基座）与 `pretrained_checkpoint`（X-WAM 预训练 ckpt，默认 `./checkpoints/pretrained/checkpoints/last.ckpt`），可在命令行覆盖这两个路径。详见 [X-WAM/README.md](X-WAM/README.md)。

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 3500 ee 0 <policy_gpu> <env_gpu> XWAM XPolicyLab
```

参数依次为：`bench_name`、`task_name`、`ckpt_name`、`env_cfg_type`、`expert_data_num`、`action_type`、`seed`、`policy_gpu_id`、`env_gpu_id`、`policy_conda_env`、`eval_env_conda_env`。其中 `expert_data_num`（示例 `3500`）仅参与拼接实验目录名，需与实际 checkpoint 目录的 6-tuple 命名一致。

`ckpt_name` 用于解析实验目录（`exp_path`），与 `task_name` 可不同（如 `cotrain`）。推理超参（异步去噪步数、`cfg`、`replan_steps` 等）见 `deploy.yml`。
