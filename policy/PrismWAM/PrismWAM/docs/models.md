# Models and checkpoints

The PrismWAM checkpoints are on Hugging Face at **[THUSI-Lab/PrismWAM](https://huggingface.co/THUSI-Lab/PrismWAM)**;
PrismWAM\* has its own repository with the same layout, **[THUSI-Lab/PrismWAM-Star](https://huggingface.co/THUSI-Lab/PrismWAM-Star)**.

## Released policies

| Model | Benchmark | Action space | Result | Recipe | Download |
| --- | --- | --- | ---: | --- | --- |
| PrismWAM | RoboTwin 2.0, Clean → Randomized | 14-D absolute joints, dual arm | 70.57 | [`robotwin`](../configs/robotwin) | [🤗 `robotwin/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robotwin) |
| PrismWAM | RoboDojo | 14-D absolute joints, dual arm | 25.10 / 18.55 | [`robodojo`](../configs/robodojo) | [🤗 `robodojo/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo) |
| PrismWAM | LIBERO | 7-D delta end effector + gripper | 99.35 | [`libero`](../configs/libero) | [🤗 `libero/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/libero) |
| PrismWAM\* | RoboTwin 2.0, Clean → Randomized | 14-D absolute joints, dual arm | 83.67 | — | [🤗 PrismWAM-Star `robotwin/`](https://huggingface.co/THUSI-Lab/PrismWAM-Star/tree/main/robotwin) |

Each folder holds one model:

```
robotwin/
  config.yaml          model and inference settings
  policy.pt            weights
  rgb_vae.*, depth_vae.*
  action_stats.pt      (LIBERO: norm_stats.json)
```

### Download and load

```bash
hf download THUSI-Lab/PrismWAM --include "robotwin/*" --local-dir weights   # -> weights/robotwin/
hf download THUSI-Lab/PrismWAM-Star --include "robotwin/*" --local-dir weights_star   # -> weights_star/robotwin/
```

```python
from prismwam import load_pretrained

policy = load_pretrained("robotwin")          # uses weights/robotwin, downloading it if missing
policy = load_pretrained("prismwam_star/robotwin")   # PrismWAM*, weights_star/robotwin
policy = load_pretrained("/path/to/model")    # any exported model folder
```

`scripts/serve.py --model robotwin` takes the same names or paths.

The first load also fetches the T5 text encoder of Wan2.2-TI2V-5B (about 11 GB) to
`weights/Wan-AI/Wan2.2-TI2V-5B`; it encodes the instructions. PrismWAM\* also reads the
current RGB frame with the CLIP image encoder of Wan2.1-I2V-14B-480P (about 5 GB), fetched to
`weights/Wan-AI/Wan2.1-I2V-14B-480P`.

## Training assets

Training needs the pretrained video model and the PrismWAM initializers under `WEIGHTS_ROOT`
(default `./weights`):

```
weights/
  Wan-AI/Wan2.2-TI2V-5B/        pretrained video model: Stage-1 init, RGB layers in Stage 2,
                                text encoder and tokenizer (text caches, inference)
  init/
    depth_vae.pt                depth VAE (RoboTwin, RoboDojo, LIBERO; both stages)
    depth_vae_realworld.pt      depth VAE of the real-robot recipe
    rgb_vae.safetensors         RGB VAE (Stage 2)
    action_initializer.pt       initial weights of the action expert (Stage 2)
```

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan-AI/Wan2.2-TI2V-5B
hf download THUSI-Lab/PrismWAM --include "init/*" --local-dir weights   # -> weights/init/
```

Stage 2 also reads the Stage-1 output from `OUTPUT_ROOT`. Normalization statistics and text
caches are built from the data; see [data preparation](data.md).
