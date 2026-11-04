# FlexWAM


## Install

Python 3.10 or newer. Install a CUDA build of PyTorch before the project dependencies. `deepspeed` compiles during install, so `gcc` must be on `PATH`.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Run commands from the repository root. Configs and scripts are read from the checkout.

## Weights

Save checkpoints under `./checkpoints`. Commands below use `hf download`; `modelscope download` takes the same arguments, with LingBot id `Robbyant/lingbot-video-dense-1.3b`. WAN files are read from `$FLEXWAM_MODEL_BASE_PATH/<repo id>/`. The default source is ModelScope. Set `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` to use Hugging Face. Text precompute downloads only when `FLEXWAM_SKIP_DOWNLOAD=false`.

LingBot Video. The public tokenizer directory is `processor/`; the loader reads `tokenizer/`.

```bash
hf download robbyant/lingbot-video-dense-1.3b --local-dir ./checkpoints/lingbot-video
ln -sfn processor ./checkpoints/lingbot-video/tokenizer
python scripts/preprocess/preprocess_lingbot_action_dit_backbone.py \
  --model-config configs/model/lingbot_video.yaml \
  --lingbot-model-path ./checkpoints/lingbot-video \
  --output ./checkpoints/lingbot-video/LingBotActionDiT_backbone_2048hdim.pt
export FLEXWAM_LINGBOT_PATH="$PWD/checkpoints/lingbot-video"
export FLEXWAM_ACTION_DIT_PATH="$FLEXWAM_LINGBOT_PATH/LingBotActionDiT_backbone_2048hdim.pt"
```

WAN. Wan2.1 supplies the UMT5 tokenizer (`google/umt5-xxl`). Wan2.2 supplies the video DiT. A missing T5 encoder or VAE is fetched on first load.

```bash
export FLEXWAM_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_MODEL_BASE_PATH="$FLEXWAM_MODEL_BASE_PATH"
hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" \
  --local-dir ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.2-TI2V-5B --include "diffusion_pytorch_model*.safetensors" \
  --local-dir ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
python scripts/preprocess/preprocess_action_dit_backbone.py \
  --model-config configs/model/wan22.yaml \
  --output ./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export FLEXWAM_ACTION_DIT_PATH="$PWD/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
```

## Configuration

Hydra configs are split into three groups:

| Group | Path | Role |
|---|---|---|
| `data` | `configs/data/` | Robot dimensions, cameras, action representation, dataset roots |
| `model` | `configs/model/` | `wan22` or `lingbot_video` |
| `task` | `configs/task/` | One training job: which data and model, plus hyperparameters |

A new task for an existing robot is a new file under `configs/task/`. Inspect the resolved config before training:

```bash
python scripts/train.py task=wan/robodojo_piper --cfg job --resolve
```

Included data profiles:

| Config | Action | Cameras | Dataset root |
|---|---|---|---|
| `piper` | 14D dual-arm joint delta; grippers stay absolute | global, left wrist, right wrist | `FLEXWAM_PIPER_3CAM_DATASET_DIR` |
| `flexiv` | 10D pose6D delta | global, global2, left wrist | `FLEXWAM_FLEXIV_3CAM_DATASET_DIR` |
| `robodojo` | 20D bimanual pose6D delta | high, left wrist, right wrist | `FLEXWAM_ROBODOJO_DATASET_DIR` (default `/path/to/robodojo_data`) |
| `robotwin` | 14D joint delta | high, left wrist, right wrist | edit `configs/data/robotwin.yaml` |

Replace `/path/to/...` in the task file or override `data.roots` on the command line.

| Task | Backbone | Data |
|---|---|---|
| `wan/piper_penbox` | WAN2.2 | `piper` |
| `wan/robodojo_piper` | WAN2.2 | `robodojo` |
| `lingbot/L` | LingBot Video | `flexiv` |

`robotwin` has a data profile and no task file.

## Train

`scripts/train/train_wan.sh` precomputes UMT5 embeddings, then launches ZeRO-1 on eight GPUs. Set `DATASET_DIR` to the LeRobot root and `FLEXWAM_MODEL_BASE_PATH` to the WAN weight directory from [Weights](#weights). The default task is `wan/flexiv_cable`. Extra Hydra overrides are forwarded to both the precompute and the trainer. Checkpoints go to `FLEXWAM_OUTPUT_BASE_DIR` (default `./runs`). `wandb.enabled` defaults to false.

```bash
export DATASET_DIR=/path/to/flexiv_cable
export FLEXWAM_MODEL_BASE_PATH=/path/to/wan-base
bash scripts/train/train_wan.sh
```

`scripts/train/train_robodojo.sh` is the same flow for `wan/robodojo_piper`. `scripts/train/train.sh` is the ZeRO-1 launcher; its first argument is the process count. `scripts/train/train_zero2.sh` selects ZeRO-2.

`scripts/train/train_lingbot.sh` is the LingBot counterpart. It precomputes Qwen embeddings with `scripts/preprocess/precompute_lingbot_text_embeds.py`. `TASK` is `lingbot/L` or `lingbot/conveyor`. `FLEXWAM_LINGBOT_PATH` is the LingBot checkout, and `FLEXWAM_ACTION_DIT_PATH` defaults to `LingBotActionDiT_backbone_2048hdim.pt` inside that checkout.

```bash
export DATASET_DIR=/path/to/flexiv/L
export FLEXWAM_LINGBOT_PATH=/path/to/lingbot-video
bash scripts/train/train_lingbot.sh

TASK=lingbot/conveyor DATASET_DIR=/path/to/piper/conveyor \
  bash scripts/train/train_lingbot.sh
```

## Deploy

`deploy/deploy.sh` serves a WAN checkpoint over WebSocket. The server has no authentication. The script binds `0.0.0.0`; change `--host` to `127.0.0.1` unless the port should be reachable on a lab network.

Set `RUN_DIR` to the training run. The script reads `checkpoints/weights/step.pt` and `dataset_stats.json` under that directory unless `CHECKPOINT_PATH` and `DATASET_STATS_PATH` are set. `FLEXWAM_MODEL_BASE_PATH` is the frozen WAN VAE, UMT5 encoder, and tokenizer directory. `--task` in the script is `wan/piper_penbox`; change it and `--robot-type` together. `--robot-type` is `piper`, `flexiv`, or `dual_ee`. `--delta` must match the checkpoint.

```bash
export RUN_DIR=/path/to/run
export FLEXWAM_MODEL_BASE_PATH=/path/to/wan-base
export CUDA_VISIBLE_DEVICES=0
bash deploy/deploy.sh
```

`deploy/deploy_flexiv.sh` is the LingBot template. It reads `FLEXWAM_LINGBOT_PATH` instead of the WAN base directory.

## Evaluate

See `scripts/evaluation/`. `open_loop.py` compares predicted actions with dataset ground truth. `open_loop_deploy.py` runs the deployment policy and writes the action plots. `visualize_critic.py` and `visualize_router.py` render the critic and router videos. Each script prints its arguments with `--help`.

## Layout

```text
configs/     data, model, and task YAML
src/flexwam  model, data, training, inference
deploy/      WebSocket server and robot profiles
scripts/train        ZeRO launchers
scripts/preprocess   text caches, ActionDiT backbone prep, dataset downsample
scripts/evaluation   open-loop and visualization entry points
tests/       unittest modules named test_*.py
```

## License

FlexWAM is released under the [Apache License 2.0](LICENSE). Copyright 2026 The FlexWAM Authors. Files that still carry an upstream copyright header keep that header.

## Acknowledgements

FlexWAM uses code and designs from the projects below.

- [OpenPI](https://github.com/Physical-Intelligence/openpi). `deploy/client/` is the OpenPI WebSocket policy protocol used by the deployment server.
- [FastWAM](https://github.com/yuantianyuan01/FastWAM). The WAN2.2 path — Hydra task configs, ActionDiT backbone preprocessing, text-embedding caches, and open-loop evaluation — follows FastWAM.
- [LingBot-Video](https://github.com/Robbyant/lingbot-video). The LingBot video expert lives in `src/flexwam/models/backbone/lingbot_video/` and loads LingBot-Video weights.
- [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM). Image transforms and color jitter follow OpenWAM.
