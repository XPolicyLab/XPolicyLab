# FocusVLWA

**Contributor:** Focus-VLWA authors | **Paper:** not listed | **arXiv:** not listed | **Original code:** [Focus-VLWA](https://github.com/MachEmbodied/Focus-VLWA)

本目录提供模型适配器、部署循环、逐步历史采集，以及 XPolicyLab 协议所需的服务端和客户端启动脚本。

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

此适配器位于 `XPolicyLab/policy/ME_Brain_1`，通过独立的 `focus_vlwa` 包加载模型，不包含模型源码副本。当前提供推理和评测接入，支持 `env_cfg_type=arx_x5`、`action_type=joint`；其他机器人和动作表示尚未验证。

安装沿用模型仓库的环境配置，因此不重复提供 `install.sh`。设置 `FOCUS_VLWA_SOURCE_REPO` 为模型仓库目录，`XPOLICYLAB_SOURCE` 为本仓库目录，然后在独立策略环境中安装：

```bash
cd "$FOCUS_VLWA_SOURCE_REPO"
uv sync
uv run focus-vlwa install-transformers-patch
uv pip install --python .venv/bin/python -e "$XPOLICYLAB_SOURCE"
```

在 `deploy.yml` 中设置 `model_path` 和 `policy_uv_env_path`。策略环境目录须包含 `.venv`。Focus-VLWA 权重的 `asset_id` 为 `arx_x5_sim`。离线环境可设置 `tokenizer_path` 或 `FOCUS_VLWA_TOKENIZER_PATH`；联网环境可由模型包下载并缓存分词器。命令行 `--ckpt` 的绝对路径优先于 `model_path`。

默认使用已安装的 `focus_vlwa` 包。使用源码时，可在本目录创建名为 `focus-vlwa` 的软链接指向源码仓库，或通过 `FOCUS_VLWA_SOURCE` 环境变量、`focus_vlwa_source` 配置项指定源码的 `src` 目录；配置项同时供服务端和仿真客户端使用。相对路径以本策略目录为基准，显式源码路径优先于软链接。

`history_mode`、`max_token_len` 可覆盖模型配置，环境变量优先；`num_inference_steps` 控制去噪步数；`result_dir` 设置输出位置。

## Data Processing

本适配器只接入评测，不转换数据，因此不提供 `process_data.sh`。独立模型仓库的后训练读取器支持 LeRobot v2.1 和 v3.0；两种版本须分别安装 `.[train-v2]` 和 `.[train]`，并使用原始 25 FPS 回合图像生成头部历史。

XPolicyLab 官方的 `scripts/transform_lerobot_v21_format.py` 和 `scripts/transform_lerobot_v30_format.py` 分别产出对应版本。其基础字段 `observation.state`、`action`、`observation.images.cam_high`、`observation.images.cam_left_wrist`、`observation.images.cam_right_wrist` 与 Focus-VLWA 的字段名一致；本策略要求 `arx_x5` 的关节状态和动作各为 14 维。读取器接受单步动作 `[14]`，或预先计算的 50 步动作块 `[50,14]`，并从头部相机视频按回合构建 `hist_images` 和 `hist_mask`。

**官方转换结果不是完整的后训练数据。** 联合训练和冻结世界模型训练还需要对齐的 `s1`／`s1_mask`（或 `world_state`／`world_state_mask`）以及 `event_action`／`event_action_mask`；官方转换器不生成这些世界模型监督字段。训练时须使用已补齐这些字段的 LeRobot 数据，不能只运行官方转换脚本后直接调用 `focus-vlwa post-train`。模型仓库目前提供读取和训练入口，未提供生成这些额外监督字段的转换脚本；具体数据要求见 [Focus-VLWA 的后训练说明](https://github.com/MachEmbodied/Focus-VLWA#post-training)。

## Training

本目录是评测专用适配器，不提供 `train.sh`；后训练代码、数据读取和两阶段训练命令在独立的 Focus-VLWA 仓库。世界模型监督字段的生成流程不包含在本适配器中。提交评测专用 PR 时须按 XPolicyLab 的贡献规范声明此范围，并提供与发布权重匹配的模型代码版本。

## Evaluation

`focus-vlwa checkpoint` 推理发布包只需 `model.safetensors`、`model_config.json` 和 `assets/arx_x5_sim/norm_stats.json`。不需要把训练用的 `metadata.pt` 或 `optimizer.pt` 加入推理发布包。评测前运行 `focus-vlwa check-checkpoint "$CHECKPOINT"` 检查参数及形状。

客户端和服务端统一使用 `head_history`：20帧完整头部历史，每帧16个视觉令牌；当前三路图像各256个视觉令牌。客户端逐控制步采集，服务端从 `model_config.json` 恢复模型配置，默认文本长度为400。

默认预测50步动作、执行前25步、去噪10步。执行长度通过 `FOCUS_VLWA_REPLAN_STEPS` 配置，去噪步数通过 `deploy.yml` 中的 `num_inference_steps` 配置。历史在每个控制步采集，按环境独立缓存，并在回合开始时重置。

配置 RoboDojo 和 XPolicyLab 后，在 RoboDojo 根目录调用标准评测入口。下面的变量需要按本地环境设置；`CHECKPOINT` 使用检查点目录的绝对路径，`FOCUS_VLWA_ENV` 指向包含 `.venv` 的策略环境目录。

```bash
export TASK=cover_blocks SEED=0 POLICY_GPU=0 ENV_GPU=0
export CHECKPOINT=/lpai/volumes/base-eb-ali-wl/jingxie/openpi-assets/checkpoints/focus-vlwa-release
export FOCUS_VLWA_ENV=/path/to/Focus-VLWA
export CONDA_ENV=RoboDojo
export FOCUS_VLWA_HISTORY_MODE=head_history
export FOCUS_VLWA_MAX_TOKEN_LEN=400
export FOCUS_VLWA_REPLAN_STEPS=25
export EVAL_ENV_TYPE=sim EVAL_NUM=10 ROBODOJO_NUM_ENVS=10 ROBODOJO_RANDOM_ENV_LIMIT=10

bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/ME_Brain_1 \
  --task "$TASK" --ckpt "$CHECKPOINT" \
  --env-cfg arx_x5 --action-type joint --seed "$SEED" \
  --policy-gpu "$POLICY_GPU" --env-gpu "$ENV_GPU" \
  --policy-env "$FOCUS_VLWA_ENV" --eval-env "$CONDA_ENV" \
  --eval-num "$EVAL_NUM"
```

`eval.sh` 由 XPolicyLab 调用，并使用 `setup_eval_policy_server.sh` 和 `setup_eval_env_client.sh` 启动两端进程。也可以从本策略目录直接使用标准十参数入口；下面的示例沿用上面设置的 `CHECKPOINT`、`FOCUS_VLWA_ENV` 和 `CONDA_ENV`：

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>

EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" arx_x5 joint 0 \
  0 0 "$FOCUS_VLWA_ENV" "$CONDA_ENV"
```

## Notes

部署循环在每个控制步采集历史，因此与通用示例的循环不同。输入图像保持 RGB；服务端解码后模型适配器只调整布局，客户端历史采集使用 XPolicyLab 的共享解码函数。

迁移后应在包含 CUDA、模型权重和机器人配置的环境中运行调试闭环。以下命令从本策略目录执行，`FOCUS_VLWA_ENV` 是包含 `.venv` 的策略环境目录；第二次额外设置 `DEBUG_OBS_ENCODED=1` 以检查编码图像传输：

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo "$TASK" "$CHECKPOINT" arx_x5 joint "$SEED" \
  "$POLICY_GPU" "$ENV_GPU" "$FOCUS_VLWA_ENV" "$CONDA_ENV"
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo "$TASK" "$CHECKPOINT" arx_x5 joint "$SEED" \
  "$POLICY_GPU" "$ENV_GPU" "$FOCUS_VLWA_ENV" "$CONDA_ENV"
```
