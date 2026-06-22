# A1

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：

- 数据集子目录命名固定为 5 元组：
  `<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>`
- 训练产物子目录命名固定为 6 元组：
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`
- `task_name` 表示训练任务标识；多任务场景下可由 policy 自行扩展为逗号分隔，或使用一个统一的数据集标识名
- `ckpt_name` 表示 checkpoint 标识；单任务通常与 `task_name` 同名，多任务共训建议显式写成 `cotrain` 或其他固定名称

## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

## 训练

命令（7 个参数，不含 `task_name`）：

```bash
cd /path/to/XPolicyLab/policy/A1
bash train.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

参数语义与总 README 保持一致：

- `dataset_name`: 数据集名称，如 `RoboDojo`
- `ckpt_name`: checkpoint 标识。单任务通常与 `task_name` 相同；多任务共训建议填 `cotrain`
- `env_cfg_type`: 环境配置 / 本体类型，如 `arx_x5`
- `expert_data_num`: 训练轨迹数；如果使用外部 LeRobot 数据集且目录已固定，可将其视为命名占位符，建议填与数据版本一致的固定值
- `action_type`: 动作类型，如 `joint`
- `seed`: 随机种子
- `gpu_id`: GPU 编号列表，如 `0,1,2,3`

### 默认单任务训练

不开 wandb：

```bash
conda activate lqw-a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export ENABLE_WANDB=false
export TASK_NAME=stack_bowls
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 42 1,2,3,4
```

开 wandb：

```bash
conda activate lqw-a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export ENABLE_WANDB=true
export WANDB_PROJECT=a1-xpolicylab
export WANDB_API_KEY=<your_wandb_api_key>
export TASK_NAME=stack_bowls
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 42 0,1,2,3
```

### 默认多任务共训

```bash
bash train.sh RoboDojo cotrain arx_x5 50 joint 42 0,1,2,3
```

### 直接读取外部 LeRobot 多任务数据集训练

当数据已经是现成的 LeRobot 目录时，不再要求其目录名符合 `policy/A1/data/<5元组>` 规则；  
命令行中的 `ckpt_name` 用于统一训练产物命名，多任务共训建议填 `cotrain`。

以共享多任务数据集
`/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21`
为例，推荐：

- `dataset_name=RoboDojo`
- `ckpt_name=cotrain`
- `env_cfg_type=arx_x5`
- `expert_data_num=3500`
- `action_type=joint`

对应训练产物目录将命名为：

```text
policy/A1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42
```

直接运行命令：

```bash
conda activate a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export ENABLE_WANDB=false
export LEROBOT_DATA_PATH=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21
export SEQ_LEN=1536
# export GLOBAL_BATCH_SIZE=128
# export DEVICE_TRAIN_MICROBATCH_SIZE=16
export GLOBAL_BATCH_SIZE=128
export DEVICE_TRAIN_MICROBATCH_SIZE=8
export NUM_WORKERS=4
export MAX_CROPS=3
export ENABLE_WANDB=true
export WANDB_PROJECT=A1
export WANDB_API_KEY=<your_wandb_api_key>

bash train.sh RoboDojo cotrain arx_x5 3500 joint 42 0,1,2,3,4,5,6,7
```

说明：

- 不设置 `LEROBOT_DATA_PATH` 时，训练脚本会优先自动查找共享数据集和 `policy/A1/data/` 下匹配的数据集；若需要本地 HDF5 转换，可设置 `TASK_NAME`
- 设置 `LEROBOT_DATA_PATH` 后，会直接使用该 LeRobot 根目录，不再触发本地 HDF5 转换
- `A1` 的通用 `LeRobotDatasetWrapper` 会从数据集元信息中自动读取相机 / state / action 字段，因此可直接兼容这份共享多任务数据
- 运行时会自动生成 `A1/configs/datasets/xpolicylab_runtime.yaml`，并将共享数据路径写入其中
- 若遇到 `max_sequence_length=1024` 不足导致的报错，可提高 `SEQ_LEN`；当前这份共享数据建议先用 `2048`
- 若遇到图像 crop 数超限报错，可提高 `MAX_CROPS`；但当前 A1 默认 `crop_mode=overlap-and-resize-c2`，实际 crop 数约为 `1 + MAX_CROPS`，因此多任务训练不建议一开始就设太大。对这份共享数据，建议先用 `GLOBAL_BATCH_SIZE=16`、`DEVICE_TRAIN_MICROBATCH_SIZE=1`、`MAX_CROPS=8`
- 若遇到 `DynamicCache` 接口报错或后续 `CUDA illegal memory access` 一类问题，优先检查 `a1` 环境是否偏离 A1 原项目推荐版本；建议使用 `torch==2.6.0`、`torchvision==0.21.0`、`torchaudio==2.6.0`、`transformers<5`
- 若执行 `pip install -e /mnt/xspark-data/lqw/XPolicyLab` 时出现 `setuptools>=61.0` 找不到，通常是当前 pip 走了不可用镜像。可先执行 `export PIP_INDEX_URL=https://pypi.org/simple`，并优先使用 `python -m pip install --no-build-isolation -e /mnt/xspark-data/lqw/XPolicyLab`，避免本地包安装时再次进入隔离构建拉取 build dependencies

## 推理

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

不指定 ckpt：

```bash
conda activate lqw-a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 a1 a1
```

指定 ckpt：

```bash
conda activate a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export MODEL_PATH=/mnt/xspark-data/lqw/XPolicyLab/policy/A1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42/latest-unsharded
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 a1 a1
```

用 `cotrain` 权重评测单任务：

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 a1 a1
```

## 独立部署：A1 RoboDojo joint-delta 策略服务

上面的 `eval.sh` 走的是 RoboTwin/RoboDojo 评测 harness（依赖 `env_cfg/` 基建、按 5/6 元组命名）。
如果你只想**单独起一个 A1 socket server** 供外部仿真器（XPolicyLab `ModelClient` 协议）连接，
用 `serve_a1.sh` 即可，**不需要 `env_cfg/`，也不需要另外 clone A1**。

A1 推理代码（带 delta 还原的 `deploy/infer_vla.py` + 配套的 `a1` 包）已 **vendor 进 `policy/A1/A1`**，
归一化统计 `policy/A1/assets/robodojo_joint_delta.json` 也随仓库提供。所以拉到 XPolicyLab 后，
**你唯一要自备的是 checkpoint**（unsharded 目录，约 34GB，不入库）。

### 起服务

```bash
cd /path/to/XPolicyLab
MODEL_PATH=/abs/path/to/<run>/step88000-unsharded bash policy/A1/serve_a1.sh
```

可用环境变量（中括号内为默认值）：

- `MODEL_PATH`（**必填**）：unsharded checkpoint 目录（内含 `model.pt` + `config.yaml`）
- `PORT` [32180] / `HOST` [0.0.0.0] / `GPU` [0]
- `CONDA_ROOT` [`conda info --base`] / `CONDA_ENV` [a1]（本机这套环境名是 `a1_origin`）
- `NORM_STATS` [`policy/A1/assets/robodojo_joint_delta.json`]：反归一化统计，与该 checkpoint 配套
- `REQUEST_LOG_DIR` [关闭]：设为目录后，会把每个调用方的图像 + 原始 state + 输出 action 落盘
  （对齐 A1 `deploy/api_server.py` 的请求日志），用于排查外部 client 到底发了什么
- `A1_REPO_DIR` [`policy/A1/A1`]：默认用 vendor 的内嵌 A1；要换别的 A1 checkout 时覆盖它

本机实跑示例：

```bash
MODEL_PATH=/x2robot_v2/kayden/A1_cvpr/model/checkpoints/a1_robodojo_joint_delta_unsharded/step88000-unsharded \
PORT=32180 GPU=1 CONDA_ENV=a1_origin \
DATA_DIR=/x2robot_v2/kayden/A1_cvpr/data HF_HOME=/x2robot_v2/kayden/hf_cache \
bash policy/A1/serve_a1.sh
```

checkpoint 约 34GB，加载需几分钟；以**端口进入 LISTEN**（`ss -ltn | grep :32180`）为就绪标志，
不要等日志（stdout 经 `tee` 会块缓冲）。建议放进 tmux 长跑。

### 开环验证

服务起好后，在另一个终端用配套客户端走**真实 socket 协议**做开环对比：

```bash
python policy/A1/test_openloop_client.py \
  --episode /abs/.../RoboDojo/<task>/arx_x5/data/episode_0000007.hdf5 \
  --host 127.0.0.1 --port 32180 --chunk 50 --max_frames 400
```

会打印整体 / 逐维 L1、MSE（关节空间，14 维绝对动作）。当前 step88000 在
`arrange_largest_number` 上整体 L1 ≈ 0.03，各臂关节维 ≈ 0.002–0.01，属正常范围。

### 与训练对齐的关键点（已在 `model.py` / `deploy.yml` 内置，无需手动处理）

- **delta 还原**：模型预测的是关节 delta，server 端按 `delta_mask="6,-1,6,-1"`（双臂 6 维=delta、
  夹爪=绝对）把原始 state 加回，得到绝对关节动作。该步骤在 `model.py` 内完成（不重复传给 `infer_vla`）。
- **归一化**：`bounds_q99`；输入 state 会被 clip 到 [-1,1]，但**输出动作反归一化不 clip**（模型可超出 q99/q01）。
- **图像不 resize**：默认保留调用方原分辨率（与 A1 训练 / HTTP api_server 一致，交给模型自己的
  预处理裁剪缩放）；需要固定尺寸时把 `deploy.yml` 的 `image_resize` 设成 `[w, h]`。
- **相机顺序**：`cam_head` / `cam_left_wrist` / `cam_right_wrist`，三路都要齐，且必须是 **RGB**。
- **指令**：obs 里必须带真实 `instruction`；缺失会退化成占位 prompt，VLA 精度会明显变差。
