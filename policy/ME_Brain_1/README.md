# ME-Brain-1.0

**Contributor:** Focus-VLWA authors | **Paper:** not listed | **arXiv:** not listed | **Original code:** [Focus-VLWA](https://github.com/MachEmbodied/Focus-VLWA)

本目录包含 Focus-VLWA 模型源码、数据接入、两阶段后训练及 RoboDojo 评测适配器。支持 `env_cfg_type=arx_x5`、`action_type=joint`；其他机器人和动作表示尚未验证。模型源码位于 [`source/`](source/)，来源版本及集成修改见 [SOURCE.md](source/SOURCE.md)。

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

在本策略目录执行安装。默认使用 `source/.venv`；推理、LeRobot v3.0 训练和 v2.1 训练分别选择以下模式，不要在同一环境中同时安装两种训练读取器：

```bash
bash install.sh inference
# For LeRobot v3.0 training:
bash install.sh train
# For LeRobot v2.1 training, choose train-v2 instead of train:
bash install.sh train-v2
```

安装脚本通过 uv 安装内置模型及 XPolicyLab 依赖，并安装 `transformers==4.53.2` 的兼容补丁。需要支持 CUDA 的 Linux 环境。仿真客户端自动从内置 `source/src` 导入历史采集工具，不需要另外克隆模型仓库；客户端环境仍须按 RoboDojo 的要求安装 NumPy、PyYAML、OpenCV 等依赖。

`deploy.yml` 的 `policy_uv_env_path` 默认为 `source`。使用其他策略环境时，评测参数传入包含 `.venv` 的环境目录。源码默认为内置版本；显式的 `FOCUS_VLWA_SOURCE` 或 `focus_vlwa_source` 可覆盖为其他源码的 `src` 目录，相对路径以本策略目录为基准。

## Data Processing

`process_data.sh` 接入**已补齐世界模型监督字段**的 LeRobot v2.1 或 v3.0 数据，校验版本、25 FPS、动作与状态维度、三路视频及监督字段，然后按标准名称建立分片软链接与清单。它不复制大数据集，不改变图像颜色，也不再次归一化世界模型潜变量。

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  <prepared_dataset_root> [prepared_dataset_root ...]

bash process_data.sh RoboDojo joint arx_x5 joint /data/robodojo-supervised/task_a
```

输出为 `data/RoboDojo-joint-arx_x5-joint/dataset_manifest.json` 和 `shards/0000` 等软链接。同一批分片必须使用相同数据版本；已有输出不会被覆盖。可使用多个路径或带引号的通配符。`FOCUS_VLWA_DATA_PYTHON` 可指定数据处理解释器；此入口只使用 Python 标准库。

基础字段与官方 `scripts/transform_lerobot_v21_format.py`、`scripts/transform_lerobot_v30_format.py` 一致：`observation.state: [14]`、`action: [14]` 或 `[50,14]`，以及 `observation.images.cam_high`、`cam_left_wrist`、`cam_right_wrist` 的原始 25 FPS 回合视频。读取器从回合视频生成 20 帧 `hist_images` 和 `hist_mask`，无需预先保存历史图像列。

**官方转换结果尚不包含完整训练监督。** 联合训练和冻结世界模型训练都需要逐行对齐的 `s1: [270,32]`、`s1_mask: [270]`（也可使用 `world_state`、`world_state_mask`），以及绝对关节目标 `event_action: [14]` 和标量或 `[1]` 的 `event_action_mask`。必须使用已补齐这些字段的数据；普通官方导出会在数据处理时被明确拒绝。本入口不生成世界模型潜变量或事件标注。

校验检查元数据和必要文件是否存在，不逐行扫描全部 parquet 或解码全部视频。实际读取及样本数值由训练读取器检查；数据软链接对应的原始目录须持续可访问。

## Training

`train.sh` 使用标准六参数入口，支持单卡及逗号分隔的多卡列表，并调用内置的 PyTorch 后训练代码。`FOCUS_VLWA_STAGE=joint` 联合训练动作与世界模型；`frozen` 从选定的联合训练检查点开始冻结世界模型。两个阶段使用各自的优化器。

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <gpu_id[,gpu_id...]> [post-train options]

export FOCUS_VLWA_INIT_CHECKPOINT=/checkpoints/posttrain_init
export FOCUS_VLWA_TOKENIZER_PATH=/assets/tokenizer.model
bash train.sh RoboDojo joint arx_x5 joint 0 0 --batch-size 32

FOCUS_VLWA_STAGE=frozen \
FOCUS_VLWA_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-joint-arx_x5-joint-0/5000" \
FOCUS_VLWA_DATASET="$PWD/data/RoboDojo-joint-arx_x5-joint/shards/*" \
bash train.sh RoboDojo frozen arx_x5 joint 0 0,1,2,3 --batch-size 128
```

初始化检查点须包含 `model.safetensors`、`model_config.json` 和 `assets/arx_x5_sim/norm_stats.json`。默认从 `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/shards/*` 读取数据；可通过 `FOCUS_VLWA_DATASET` 指定其他本地目录、分片通配符或逗号分隔的目录列表。已有 `FOCUS_VLWA_LEROBOT_REPOS` 设置的优先级最高；本入口要求对应数据在本机可访问。

`FOCUS_VLWA_PYTHON` 可指定已安装相应训练依赖的 Python 可执行文件；`FOCUS_VLWA_NORM_STATS_DIR` 可覆盖初始化检查点的归一化资产目录。`--num-steps`、`--num-workers`、`--save-interval`、`--parameter-precision float32` 和恢复训练参数可作为额外参数传入。六参数中的 seed 会传给训练配置；全局 batch size 必须能被 GPU 数整除。

输出为 `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/<step>/`。例如联合训练的第 5000 步位于 `checkpoints/RoboDojo-joint-arx_x5-joint-0/5000/`，可将此目录的绝对路径传给评测。恢复同一阶段使用 `--resume-checkpoint`；切换到冻结阶段使用 `FOCUS_VLWA_INIT_CHECKPOINT`。默认训练步数、学习率和冻结策略见内置源码的后训练说明。

设置 `FOCUS_VLWA_DRY_RUN=1` 会检查路径和数据元信息，并打印启动命令，不创建训练任务、不加载模型权重。

## Evaluation

推理发布包需要 `model.safetensors`、`model_config.json` 和 `assets/arx_x5_sim/norm_stats.json`；不需要训练用的优化器文件。`deploy.yml` 中的 `model_path` 设置默认权重目录，命令行检查点的绝对路径优先。权重下载地址仍须由提交者提供。

客户端和服务端使用 `head_history`：20 帧完整头部历史，每帧 16 个视觉令牌；当前三路图像各 256 个视觉令牌。客户端逐控制步采集，按环境独立缓存，并在回合开始时重置。默认文本长度 400，预测 50 步动作、执行前 25 步、去噪 10 步。

`history_mode`、`max_token_len` 可覆盖模型配置，`FOCUS_VLWA_HISTORY_MODE`、`FOCUS_VLWA_MAX_TOKEN_LEN` 的优先级更高；`FOCUS_VLWA_REPLAN_STEPS` 控制执行长度。`num_inference_steps` 控制去噪步数，`result_dir` 指定输出位置，`asset_id` 默认为 `arx_x5_sim`。离线分词器可通过 `tokenizer_path` 或 `FOCUS_VLWA_TOKENIZER_PATH` 指定。

从本策略目录执行标准十参数入口：

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>

export CHECKPOINT="$PWD/checkpoints/RoboDojo-frozen-arx_x5-joint-0/10000"
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" arx_x5 joint 0 \
  0 0 uv RoboDojo
```

## Notes

部署循环逐控制步采集历史，因此与通用示例不同。输入图像保持 RGB；服务端解码后模型适配器只调整布局，客户端使用 XPolicyLab 的共享解码函数。

在具有 CUDA、真实模型权重和机器人配置的环境中验证调试闭环：

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" arx_x5 joint 0 \
  0 0 uv RoboDojo
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks "$CHECKPOINT" arx_x5 joint 0 \
  0 0 uv RoboDojo
```

离线回放需要 XPolicyLab 的 `feat/offline_eval` 共享客户端；仅添加本策略入口不会使旧版共享脚本支持 `EVAL_ENV_TYPE=offline`。训练、调试闭环和仿真结果须在实际 GPU 环境中验证后填写到 PR。
