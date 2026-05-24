# GigaWorld-Policy

> **GigaWorld-Policy: An Efficient Action-Centered World-Action Model**

[![arXiv](https://img.shields.io/badge/arXiv-2603.17240-b31b1b.svg)](https://arxiv.org/abs/2603.17240)
[![Website](https://img.shields.io/badge/Website-Project_Page-blue.svg)](https://gigaai-research.github.io/GigaWorld-Policy/)

## 📖 Overview

World-Action Models (WAM) initialized from pre-trained video generation backbones have demonstrated remarkable potential for robot policy learning. **GigaWorld-Policy** is an action-centered WAM that learns 2D pixel-action dynamics while enabling efficient action decoding, with optional video generation.

### Key Features

- 🚀 **9x faster** inference compared to Motus (leading WAM baseline)
- 📈 **7% higher** task success rate than Motus
- 💪 **95% improvement** over pi-0.5 on RoboTwin 2.0

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n gigaworld-policy python==3.11
conda activate gigaworld-policy
```

### 2. Install Dependencies

```bash
pip install ./third_party/giga-train
pip install ./third_party/giga-models
pip install ./third_party/giga-datasets
```

## 📊 Data Preprocessing

Before training, you need to compute the normalization statistics and pre-compute the T5 text embeddings for your dataset.

### 1. Compute Normalization Statistics

This will generate a `norm_stats_delta.json` file which is required by the policy.

```bash
python -m scripts.compute_norm_stats \
  --data_paths "/path/to/dataset_dir" \
  --output_path "/path/to/norm_stats_delta.json" \
  --embodiment_id {embodiment-id} \
  --delta-mask {delta-mask} \
  --sample-rate 1.0 \
  --action-chunk 48 \
```

*   `--embodiment_id`: Check `compute_norm_stats.py` for the mapping from robot type to ID.
*   `--delta_mask`: A boolean mask indicating which action dimensions are deltas (True) vs. absolute values (False).

### 2. Compute T5 Embeddings

This will pre-compute and save the T5 text embeddings for the language instructions in your dataset.

```bash
python -m scripts.compute_t5_embedding \
  --repo_id "/path/to/dataset_dir" \
  --root "/path/to/dataset_dir" \
  --wan_path "/path/to/Wan2.2-TI2V-5B" \
  --device "cuda" \
  --text_len 512 \
  --t5_folder_name "t5_embedding"
```

## ⚙️ Configuration

After completing the data preprocessing steps, modify the config file `world_action_model/configs/example.py` to point to the generated files and your model weights:

| Parameter | Description |
|-----------|-------------|
| `models.pretrained` | Path to your pretrained model weights |
| `transform.norm_path` | Path to the generated `norm_stats_delta.json` |
| `data_dir` | Path to your dataset |

## 🚀 Training

Once the data is preprocessed and the configuration is set, you can start training:

```bash
python -m scripts.train --config world_action_model.configs.example.config
```

## 🚀 Inference

We provide an inference server and a simple open-loop evaluation client. Open-loop here means we sample observations (images/state) from an offline dataset and run inference, without executing actions in a real environment to collect the next observations.

### 1. Start Server

```bash
python -m scripts.inference_server \
  --model_id "/path/to/huggingface_model_dir_or_id" \
  --transformer_path "/path/to/transformer_checkpoint_dir" \
  --stats_path "/path/to/norm_stats_delta.json" \
  --t5_embedding_pkl "/path/to/t5_embedding.pt"
```

Optionally, add `--return_images` to enable video visualization during inference (videos will be saved under `--vis_dir`):

```bash
python -m scripts.inference_server \
  --model_id "/path/to/huggingface_model_dir_or_id" \
  --transformer_path "/path/to/transformer_checkpoint_dir" \
  --stats_path "/path/to/norm_stats_delta.json" \
  --t5_embedding_pkl "/path/to/t5_embedding.pt" \
  --return_images \
  --vis_dir "./vis"
```

### 2. Run Open-loop Client

```bash
python -m scripts.inference_client \
  --dataset_paths "/path/to/dataset_dir" \
  --save_dir "./vis"
```

## 📅 Roadmap

| Component | Status |
|-----------|--------|
| Inference Code | ✅ |
| Training Code | ✅ |
| Pre-trained Weights | 🔲 |

## 📚 Citation

```bibtex
@article{ye2026gigaworld,
  title={GigaWorld-Policy: An Efficient Action-Centered World-Action Model},
  author={Ye, Angen and Wang, Boyuan and Ni, Chaojun and Huang, Guan and Zhao, Guosheng and Li, Hao and Li, Hengtao and Li, Jie and Lv, Jindi and Liu, Jingyu and Cao, Min and Li, Peng and Deng, Qiuping and Mei, Wenjun and Wang, Xiaofeng and Chen, Xinze and Zhou, Xinyu and Wang, Yang and Chang, Yifan and Li, Yifan and Zhou, Yukun and Ye, Yun and Liu, Zhichao and Zhu, Zheng},
  journal={arXiv preprint arXiv:2603.17240},
  year={2026}
}
```
