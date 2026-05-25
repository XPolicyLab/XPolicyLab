## Motus 训练流程整理

这份说明面向当前这套本地环境，重点是把你给定的两类路径接起来：

- 数据根目录：/vepfs-cnbje63de6fae220/xspark_shared/lerobot
- 基础模型目录：/vepfs-cnbje63de6fae220/xspark_shared/model_weights
- Motus 代码目录：/vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Motus/Motus

先说结论：

- 如果你要用的是 LeRobot 格式数据，Motus 已经支持直接训练，不需要先转成 RobotWin 格式。
- 如果你要用的是 RoboTwin 原始数据，才需要走 data/robotwin2/robotwin_data_convert 这套转换流程。
- 你当前给的数据根目录是 lerobot，所以主流程建议走 LeRobot 直读。

## 1. 环境准备

先进入代码目录并激活环境：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Motus/Motus
conda activate motus
```

如果这个环境还没装 LeRobot 相关依赖，按仓库里的要求补齐：

```bash
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
```

建议先确认基础模型目录里至少有下面这些子目录：

```text
/vepfs-cnbje63de6fae220/xspark_shared/model_weights/
	Motus/
	Wan2.2-TI2V-5B/
	Qwen3-VL-2B-Instruct/
```

其中用途分别是：

- Motus：Stage 2 预训练权重，用来做 Stage 3 SFT 初始化
- Wan2.2-TI2V-5B：WAN 主干和 T5 编码器
- Qwen3-VL-2B-Instruct：VLM 编码器

## 2. 先选定要训练的具体 LeRobot 数据集

当前 /vepfs-cnbje63de6fae220/xspark_shared/lerobot 下面不是单一数据集，而是多个数据集根目录，例如：

- build_LEGO
- pack_backpack
- robodojo_real
- robodojo_real_7tasks
- robodojo_real_8tasks_new
- robodojo_sim
- robodojo_sim_new
- v30

Motus 的 LeRobot loader 读取时，需要你明确指定某一个具体数据集根目录，而不是直接把总目录 /vepfs-cnbje63de6fae220/xspark_shared/lerobot 填进去。

例如，如果你要训练 robodojo_sim，那么配置里应该填：

- repo_id: robodojo_sim
- root: /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim

训练前建议先做一个便宜检查，确认这个目录至少有 meta，通常还会有 data 或 images：

```bash
ls /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim
ls /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim/meta
```

至少应该能看到：

- meta/episodes.jsonl
- meta/info.json

## 3. 处理 T5 文本缓存

这是 LeRobot 训练前最容易漏掉的一步。

Motus 的 LeRobot 数据加载器支持三种语言特征来源：

- parquet 内已经带 language_embedding
- meta/episodes.jsonl 里带 t5_embedding_path，并且磁盘上有对应 pt 文件
- 训练时临时在线编码，并缓存到数据集目录

你当前这批数据目录下没有看到现成的 t5_embedding pt 文件，所以更稳妥的做法是先离线补一遍缓存，再启动训练。这样可以避免训练过程中每个 worker 动态编码文本，导致速度波动或初始化变慢。

示例，以 robodojo_sim 为例：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Motus/Motus

export CUDA_VISIBLE_DEVICES=0

python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
	--repo_id sim_lerobot_v21 \
	--root /xspark-cache/shared/lerobot/sim_lerobot_v21 \
	--wan_path /mnt/xspark-data/xspark_shared/model_weights/ \
	--device cuda \
	--t5_folder_name t5_embedding
```

如果只是想先验证流程，可以先只处理少量 episode：

```bash
python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
	--repo_id robodojo_sim \
	--root /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim \
	--wan_path /vepfs-cnbje63de6fae220/xspark_shared/model_weights \
	--device cuda \
	--t5_folder_name t5_embedding \
	--max_episodes 32
```

处理完后，建议检查两个点：

```bash
ls /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim/t5_embedding | head
grep -n "t5_embedding_path" /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim/meta/episodes.jsonl | head
```

如果这两步都能看到结果，就说明语言缓存已经补齐。

## 4. 新建一份本地训练配置

不要直接改仓库自带的 configs/lerobot.yaml，建议复制一份本地配置，例如：

```bash
cp configs/lerobot.yaml configs/lerobot_xspark_local.yaml
```

然后把里面这几项改成你当前机器上的实际路径。下面给一份可直接参考的配置骨架：

```yaml
common:
	action_dim: 14
	state_dim: 14
	num_video_frames: 8
	video_height: 384
	video_width: 320
	global_downsample_rate: 1
	video_action_freq_ratio: 6

dataset:
	type: "lerobot"
	max_episodes: null
	image_aug: false
	task_mode: "single"
	task_name: null
	params:
		repo_id: "robodojo_sim"
		root: "/vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim"
		embodiment_type: "aloha_agilex_2"
		enable_t5_fallback: false
		t5_wan_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights"
		t5_folder_name: "t5_embedding"
		t5_text_len: 512

model:
	wan:
		config_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B"
		checkpoint_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B"
		vae_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
		precision: "bfloat16"
	vlm:
		checkpoint_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Qwen3-VL-2B-Instruct"
		precision: "bfloat16"
		frozen: true

training:
	batch_size: 8
	gradient_accumulation_steps: 1
	max_steps: 1000000
	learning_rate: 5.0e-5
	weight_decay: 0.01
	scheduler_type: "linear"
	warmup_steps: 200
	cycle_length: 1000000
	f_max: 0.99
	f_min: 0.4
	grad_clip_norm: 0.5
	use_amp: true
	find_unused_parameters: false

system:
	checkpoint_dir: "./checkpoints_lerobot"
	log_level: "INFO"
	log_interval: 1
	save_interval: 5000
	val_interval: 5000
	num_workers: 16
	pin_memory: true

logging:
	report_to: "tensorboard"
	wandb_project: "motus"
	tensorboard_log_dir: "tensorboard_logs"
	run_name: "robodojo_sim_motus"

resume:
	checkpoint_path: null

finetune:
	checkpoint_path: "/vepfs-cnbje63de6fae220/xspark_shared/model_weights/Motus"
```

几点说明：

- dataset.type 必须是 lerobot
- repo_id 填数据集名字
- root 必须指向具体数据集根目录
- enable_t5_fallback 如果已经提前生成缓存，建议设成 false
- finetune.checkpoint_path 指向 /vepfs-cnbje63de6fae220/xspark_shared/model_weights/Motus，用 Stage 2 权重做初始化

如果你要多任务训练，有两种常见方式：

- 单个多任务数据集：如果一个 LeRobot 根目录本身就包含多任务，task_mode 可以保持 single 或按数据实际结构调整
- 多个独立数据集目录：当前这版 loader 不适合直接把 /vepfs-cnbje63de6fae220/xspark_shared/lerobot 下所有目录一次性喂进去，建议先选定一个数据集根跑通，再考虑整理成统一的多任务目录或分别训练

## 5. 启动训练

仓库里没有现成的 lerobot 专用启动脚本，最直接的方式是仿照通用 train.sh 自己执行 torchrun。

单机 4 卡示例：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Motus/Motus

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun \
	--nnodes=1 \
	--nproc_per_node=4 \
	--node_rank=0 \
	--master_addr=127.0.0.1 \
	--master_port=29500 \
	train/train.py \
	--deepspeed configs/zero2_stage2.json \
	--config configs/lerobot_xspark_local.yaml \
	--checkpoint_dir ./checkpoints_lerobot \
	--run_name robodojo_sim_motus \
	--report_to tensorboard
```

如果只想先确认能不能跑起来，建议先把配置里的下面几项调小：

- training.batch_size
- system.num_workers
- dataset.max_episodes

比如先把 max_episodes 设成 16 或 32，做一次 smoke test。

## 6. 恢复训练

如果中断后要恢复，改配置里的 resume：

```yaml
resume:
	checkpoint_path: ./checkpoints_lerobot/checkpoint_step_5000
```

然后重新执行同一条 torchrun 命令即可。

如果是正常的 Stage 3 微调，不需要把 finetune.checkpoint_path 去掉；只有在你想完全从头训练时，才把 finetune.checkpoint_path 和 resume.checkpoint_path 都设为 null。

## 7. 和 RobotWin 转换流程的关系

仓库里现成的这两份内容：

- Motus/TRAINING_XSPARK_ROBOTWIN.md
- Motus/scripts/train_robotwin_xspark.sh

它们是给 RoboTwin 或已经转换成 Motus RobotWin 目录结构的数据用的，默认依赖的数据目录是：

- /vepfs-cnbje63de6fae220/xspark_shared/robotwin_data/motus_processed

这和你当前提供的 /vepfs-cnbje63de6fae220/xspark_shared/lerobot 不是一套数据格式，所以不要直接拿 train_robotwin_xspark.sh 去跑 LeRobot 数据。

只有在下面这种情况下，才走 RobotWin 分支：

- 你手上是 RoboTwin 原始 hdf5 数据
- 你要先把它转换成 clean/randomized/task/videos、qpos、umt5_wan 这种 Motus 原生目录结构

否则按上面的 LeRobot 直读流程处理即可。

## 8. 推荐的最小落地顺序

如果你现在只是想先把整个链路跑通，建议按这个顺序：

1. 从 /vepfs-cnbje63de6fae220/xspark_shared/lerobot 里选一个具体数据集，比如 robodojo_sim
2. 先给这个数据集离线生成 t5_embedding
3. 复制 configs/lerobot.yaml 成本地配置，改成你的绝对路径
4. 把 dataset.max_episodes 先设成一个很小的值做冒烟测试
5. 用 torchrun 启动 1 次训练，确认 dataloader、VLM、WAN、checkpoint 初始化都正常
6. 再放开 max_episodes 和 batch size，进入正式训练

如果后面你希望，我可以继续把上面这份 README 再往前推一步，直接补一份可用的 configs/lerobot_xspark_local.yaml 和一个 scripts/train_lerobot_xspark.sh，这样你就不用手工复制配置了。
