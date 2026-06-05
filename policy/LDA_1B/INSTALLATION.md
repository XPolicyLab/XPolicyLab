# LDA_1B Installation

> 上游：[LDA-1B](https://arxiv.org/abs/2602.12215) · https://github.com/jiangranlv/latent-dynamics-action

数据转换、训练与评测见 [README.md](README.md)。

## 1. Conda 环境

```bash
conda create -n LDA_1B python=3.10
conda activate LDA_1B
```

## 2. 安装依赖

```bash
cd XPolicyLab/policy/LDA_1B
bash install.sh LDA_1B
```

`install.sh` 会创建/激活 `LDA_1B` 环境、安装 `LDA-1B/requirements.txt` 与 flash-attn，并以 editable 方式安装上游包与 `XPolicyLab`。

## 3. 权重下载（install.sh 不包含）

在 `policy/LDA_1B` 目录下执行：

```bash
pip install -U "huggingface_hub[cli]"
```

| 资产 | 本地路径 | 命令 |
|---|---|---|
| Qwen3-VL-4B-Instruct | `checkpoints/Qwen3-VL-4B-Instruct` | `huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir checkpoints/Qwen3-VL-4B-Instruct --local-dir-use-symlinks False` |
| DINOv3-ViT-S/16 | `checkpoints/dinov3-vit-s` | `huggingface-cli login`（首次）后 `huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir checkpoints/dinov3-vit-s --local-dir-use-symlinks False` |
| LDA 预训练 | `checkpoints/LDA-pretrain` | `huggingface-cli download Wayer2/LDA-pretrain --local-dir checkpoints/LDA-pretrain --local-dir-use-symlinks False` |

说明：
- DINOv3 需在 HuggingFace 接受 [facebook/dinov3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009) 许可后再下载。
- `train.sh` 默认从 `checkpoints/LDA-pretrain/LDA-pretrain.pt` 加载预训练；可用 `LDA_PRETRAINED_CHECKPOINT` 覆盖。
