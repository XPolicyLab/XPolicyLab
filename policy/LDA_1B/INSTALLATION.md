# LDA_1B Installation

Upstream: [LDA-1B](https://arxiv.org/abs/2602.12215), https://github.com/jiangranlv/latent-dynamics-action.

`install.sh` handles the Python environment and editable installs, but this document is kept because LDA_1B requires separate model downloads and Hugging Face access for DINOv3.

## 1. Install the Environment

```bash
cd XPolicyLab/policy/LDA_1B
bash install.sh LDA_1B
conda activate LDA_1B
```

`install.sh` creates or activates the `LDA_1B` conda environment, installs `LDA-1B/requirements.txt`, installs `flash-attn`, and installs both upstream LDA and XPolicyLab in editable mode.

## 2. Download Weights Not Covered by `install.sh`

```bash
cd XPolicyLab/policy/LDA_1B
pip install -U "huggingface_hub[cli]"
```

| Asset | Local path | Command |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct | `checkpoints/Qwen3-VL-4B-Instruct` | `huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir checkpoints/Qwen3-VL-4B-Instruct --local-dir-use-symlinks False` |
| DINOv3-ViT-S/16 | `checkpoints/dinov3-vit-s` | `huggingface-cli login` first if needed, then `huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir checkpoints/dinov3-vit-s --local-dir-use-symlinks False` |
| LDA pretrained checkpoint | `checkpoints/LDA-pretrain` | `huggingface-cli download Wayer2/LDA-pretrain --local-dir checkpoints/LDA-pretrain --local-dir-use-symlinks False` |

DINOv3 requires accepting the Hugging Face license for the [facebook/dinov3 collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009). `train.sh` defaults to `checkpoints/LDA-pretrain/LDA-pretrain.pt`; override it with `LDA_PRETRAINED_CHECKPOINT`.
