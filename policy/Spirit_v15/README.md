## 本地 RoboDojo Raw 到 Spirit 训练流程

如果你现在的原始数据已经不再是 RobotWin，而是 XPolicyLab 体系下的 RoboDojo hdf5 数据，推荐直接走 Spirit_v15 里这条新的转换加训练链路。

这套代码当前实际支持的训练数据格式不是 LeRobot，而是 Spirit 自己的数据目录。训练集目录需要长这样：

```text
<data_root>/
  meta/task_info.json
  data/
    episode_000000/
      meta/episode_meta.json
      states/states.jsonl
      videos/
        head_camera_rgb.mp4
        left_camera_rgb.mp4
        right_camera_rgb.mp4
```

其中：

- meta/task_info.json 记录任务、相机映射、fps、state_encoding
- states/states.jsonl 记录双臂末端位姿和夹爪宽度
- videos 下三路 mp4 会被训练 loader 直接按帧读取

当前这套流程的输入根目录是：

- /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data

例如你现在能看到的一条实际数据路径就是：

- /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data/RoboDojo/stack_bowls/arx_x5

这类数据会按 XPolicyLab 的三级 pattern 来匹配：

```text
<dataset>.<task>.<env_cfg>
```

对应上面的例子就是：

```text
RoboDojo.stack_bowls.arx_x5
```

仓库里现在应该使用的 Spirit 转换器是：

- scripts/convert_xpolicylab_to_spirit.py

它会读取 XPolicyLab.utils.data_loader.load 能解析的 hdf5，并导出成 Spirit 训练目录结构。

你提到的两个参考脚本：

- /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/scripts/transform_aloha_hdf5_format.py
- /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/scripts/transform_lerobot_v21_format.py

它们依然有参考价值，但 Spirit 训练本身不直接吃这两种输出格式。Spirit loader 最终只认上面那种 Spirit 目录。

### 1. 环境准备

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Spirit_v15/spirit_v15

uv sync --extra train
source .venv/bin/activate
```

或者使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-train.txt
```

### 2. 先做数据转换

我在 scripts 下补了一套面向 XPolicyLab/RoboDojo raw 的包装脚本：

- scripts/prepare_xpolicylab_dataset.sh

参数顺序：

```text
prepare_xpolicylab_dataset.sh \
  <raw_data_root> \
  <patterns_csv> \
  <output_root> \
  [task_name] \
  [task_prompt] \
  [fps|auto] \
  [overwrite_flag] \
  [max_episodes_per_target] \
  [robot_type] \
  [data_type] \
  [data_version]
```

含义：

- raw_data_root：XPolicyLab 数据根目录；你当前就是 /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data
- patterns_csv：逗号分隔的 pattern 列表，例如 RoboDojo.stack_bowls.arx_x5 或 RoboDojo.*.arx_x5
- output_root：导出的 Spirit 格式数据目录
- task_name：写入 meta/task_info.json 的任务名
- task_prompt：默认任务提示词；如果单条 episode 自带 instruction，会优先用 episode 自己的 prompt
- fps：传 auto 时沿用原始数据里的 frequency；也可以手工指定固定 fps
- overwrite_flag：1 表示覆盖已存在输出，0 表示不覆盖
- max_episodes_per_target：每个匹配到的 <dataset>/<task>/<env_cfg> 最多转多少条 episode，适合冒烟测试
- robot_type：默认 aloha
- data_type：默认 RoboDojo
- data_version：默认 v1.0

示例：

```bash
bash scripts/prepare_xpolicylab_dataset.sh \
  /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data \
  RoboDojo.*.arx_x5 \
  /vepfs-cnbje63de6fae220/xspark_shared/spirit_datasets/RoboDojo_sim_100/ \
  sim_100 \
  "Do your job." \
  auto \
  1 \
  20 \
  aloha \
  RoboDojo \
  v1.0
```

转换完成后，建议先做一个便宜检查：

```bash
ls /vepfs-cnbje63de6fae220/xspark_shared/spirit_datasets/stack_bowls_arx_x5/meta
ls /vepfs-cnbje63de6fae220/xspark_shared/spirit_datasets/stack_bowls_arx_x5/data | head
```

至少应该能看到：

- meta/task_info.json
- data/episode_xxxxxx

### 3. 启动训练

我同时补了一个从 XPolicyLab raw 开始的一键脚本：

- scripts/train_xpolicylab_from_raw.sh

参数顺序：

```text
train_xpolicylab_from_raw.sh \
  <raw_data_root> \
  <patterns_csv> \
  <converted_data_root> \
  <pretrained_path> \
  <output_dir> \
  [num_gpus] \
  [batch_size] \
  [max_train_steps] \
  [log_interval] \
  [save_steps] \
  [num_workers] \
  [prefetch_factor] \
  [wandb_mode] \
  [task_name] \
  [task_prompt] \
  [fps|auto] \
  [overwrite_flag] \
  [max_episodes_per_target] \
  [robot_type] \
  [data_type] \
  [data_version] \
  [skip_convert] \
  [convert_only]
```

其中前 5 个是必填：

- raw_data_root：XPolicyLab 原始数据根目录
- patterns_csv：逗号分隔 pattern 列表
- converted_data_root：Spirit 格式数据输出目录
- pretrained_path：Spirit-v1.5 基座权重目录，里面必须有 model.safetensors 和 config.json
- output_dir：训练输出目录

示例：

```bash
bash scripts/train_xpolicylab_from_raw.sh \
  /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data \
  RoboDojo.stack_bowls.arx_x5 \
  /vepfs-cnbje63de6fae220/xspark_shared/spirit_datasets/stack_bowls_arx_x5 \
  /vepfs-cnbje63de6fae220/xspark_shared/model_weights/Spirit-v1.5 \
  /vepfs-cnbje63de6fae220/xspark_shared/train_outputs/spirit_stack_bowls_arx_x5 \
  4 \
  16 \
  40000 \
  25 \
  2500 \
  8 \
  4 \
  disabled \
  stack_bowls \
  "stack bowls with both arms" \
  auto \
  1 \
  20 \
  aloha \
  RoboDojo \
  v1.0 \
  0 \
  0
```

如果你已经提前转好了数据，只想跳过转换直接训练，把最后两个参数设成：

```text
skip_convert=1
convert_only=0
```

如果只想先转数据，不启动训练，把最后两个参数设成：

```text
skip_convert=0
convert_only=1
```

### 4. 推荐的最小跑通顺序

建议先按下面顺序做，而不是一上来就全量训练：

1. 先只选 1 到 2 个任务
2. 给 max_episodes_per_target 传一个很小的值，比如 10 或 20
3. 先跑 scripts/prepare_xpolicylab_dataset.sh，确认导出目录没问题
4. 再跑 scripts/train_xpolicylab_from_raw.sh，确认 dataloader 和 pretrained checkpoint 都能正常加载
5. 冒烟测试通过后，再放开 patterns_csv、max_episodes_per_target、batch_size 和 num_gpus

### 5. 和现有脚本的关系

仓库原本已经有两个相关脚本：

- scripts/run_finetune.sh
- scripts/run_robotwin_finetune.sh

它们使用环境变量传参，且 run_robotwin_finetune.sh 仍然是旧的 RobotWin 包装。现在如果你的数据源在 /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/data，应该优先使用：

- scripts/prepare_xpolicylab_dataset.sh
- scripts/train_xpolicylab_from_raw.sh

这两个脚本把同样的 Spirit 训练流程改成了位置参数包装，并把原始数据入口切到了 XPolicyLab/RoboDojo pattern 匹配方式。