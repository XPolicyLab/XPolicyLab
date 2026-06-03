# FastWAM

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?** 的官方代码仓库。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

本仓库包含 FastWAM 在 LIBERO / RoboTwin 上的训练与评估代码。

## 目录

- [File Structure](#file-structure)
- [环境安装](#环境安装)
- [模型准备](#模型准备)
- [数据集下载](#数据集下载)
- [使用 Release 权重推理](#使用-release-权重推理)
- [训练](#训练)
- [使用自己训练的权重推理](#使用自己训练的权重推理)
- [致谢](#致谢)
- [BibTeX](#bibtex)

## File Structure

```text
FastWAM/
├── configs/
│   ├── data/                 # 数据集配置（LIBERO、RoboTwin 等）
│   ├── model/                # 模型结构与组件配置
│   └── task/                 # 任务级配置（训练 task 名）
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # deepspeed zero1 训练入口
│   ├── preprocess_action_dit_backbone.py  # 训练前预处理 ActionDiT backbone
│   └── precompute_text_embeds.py  # 训练前预计算 T5 文本 embedding cache
├── experiments/
│   ├── libero/
│   │   └── run_libero_manager.py
│   └── robotwin/
│       └── run_robotwin_manager.py
├── src/fastwam/              # 核心代码
├── runs/                     # 训练输出（ckpt、日志）
├── checkpoints/              # 预训练或外部 checkpoint
├── data/                     # data目录
└── evaluate_results/         # 推理/评估结果
```

## 环境安装

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
conda install -c conda-forge "ffmpeg>=6,<7" -y
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

这里把 FFmpeg 安装到 conda 环境中，是为了让 `torchcodec==0.5` 能加载视频解码运行库，例如 `libavutil.so.58`。

## 模型准备

这一步同时是训练和推理的前置项。

第一步，先设置 Wan 模型目录（可选，默认 `./checkpoints`）：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

第二步，预生成 ActionDiT backbone（从Wan22 DiT插值）：

```bash
# uncond (fastwam)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## 数据集下载

### LIBERO

Fast-WAM 使用的 LIBERO 预处理数据已发布到：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

先下载全部压缩包，再全部解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# 下载 4 个 tar.gz 文件后执行
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

解压后目录结构应为：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

Fast-WAM 使用的 RoboTwin 预处理数据已发布到：

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

先下载全部分卷文件，再拼接并解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# 下载全部 robotwin2.0.tar.gz.part-* 文件后执行
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

解压后目录结构应为：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

根目录下如果同时保留：

```text
data/robotwin2.0/dataset_stats.json
```

可直接作为本仓库当前配置使用的统计文件，也可重新计算。

## 使用 Release 权重推理

release 的模型权重以及对应的 dataset stats 已经发布到 [Hugging Face](https://huggingface.co/yuanty/fastwam).

从 Hugging Face 下载 release 权重和 dataset stats：

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

下载后，本地目录应为：

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

`LIBERO` benchmark 评测前，请先按 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 安装环境：
最后一步执行：

```bash
pip install mujoco==3.3.2
```

`mujoco` 环境和 LIBERO 数据版本相关，最好保持一致。

我们已经把 `RoboTwin` 评测相关代码copy到了 `third_party/RoboTwin`。
但仍需按 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 中的教程完成环境安装并下载相关assets：
再创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

一键评测 release 的 LIBERO 权重：

当前 `LIBERO` / `RoboTwin` 的评测 manager 默认使用 `8` 张 GPU
（`configs/sim_libero.yaml` 和 `configs/sim_robotwin.yaml` 中的
`MULTIRUN.num_gpus=8`）。
如果你想用更少的卡，直接在命令行里传更小的值，例如
`MULTIRUN.num_gpus=4`。

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

一键评测 release 的 RoboTwin 权重：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

为了加速 RoboTwin 评测，我们在 [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml) 中打开了 `EVALUATION.skip_get_obs_within_replan=true`。
它会在一次 replan 窗口内连续执行一个 action chunk 时跳过 RGB 渲染，评测更快，但保存下来的视频帧率会低。
如果想保存完整视频，可以把它设为 `false`。

**注意：**我们测试用的是**unseen**指令，这点和Motus对齐。而[Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308)使用的是**seen**，你可以尝试设置`EVALUATION.instruction_type=seen`来使用**seen**指令，理论上会提高一两个点。

## 训练

### 1) 训练前先预计算 T5 embedding cache

使用 `scripts/precompute_text_embeds.py`，按训练 task 预计算：

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

如需多卡可用：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```


### 2) 训练（以 fastwam 为例）

首次跑某个新任务时，请先把对应 `configs/data/*.yaml` 里的 `pretrained_norm_stats` 设为 `null`。
跑完一次训练后，会在当前 run 目录生成 `dataset_stats.json`（例如 `runs/{task_name}/{run_id}/dataset_stats.json`），
后续就可以把 `pretrained_norm_stats` 改成该文件路径。

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

对于LIBERO，我们使用单机8卡训练。对于RoboTwin，我们使用了64卡来加速训练，你可以尝试调小卡数和训练总epoch数。

## 使用自己训练的权重推理

`mujoco` 环境和 LIBERO 数据版本相关，最好保持一致。之后再运行 LIBERO 评测：

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

我们已经把 `RoboTwin` 评测相关代码copy到了 `third_party/RoboTwin`。
但仍需按 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 中的教程完成安装并下载相关assets：
再创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

之后再运行 RoboTwin 评测：

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```


常用 `task_name` 示例：

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## 致谢

本仓库中的 RoboTwin 评测代码基于官方 [RoboTwin 仓库](https://github.com/RoboTwin-Platform/RoboTwin) 适配而来。感谢 RoboTwin 团队公开其代码仓库和相关 assets。

## BibTeX

如果你觉得我们的工作有帮助，欢迎引用：

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
