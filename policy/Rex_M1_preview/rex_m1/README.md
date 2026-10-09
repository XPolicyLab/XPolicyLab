# Post-Training & Deployment

> Modified by the Rex-M1-preview authors (2026): this file is upstream's, edited only to
> reflect the flattened directory layout. Everything else below is upstream's text.

Post-training and deployment code for **Xiaomi-Robotics-1** — fine-tune the released checkpoint on your own data and deploy the model with the same client–server runtime used for evaluation.

All commands below run from this directory (`policy/Rex_M1_preview/rex_m1/`). Upstream shipped this
tree one level deeper, as `xr1/`; Rex-M1-preview flattened it, so drop any `cd xr1` from older notes.
Rex-M1-preview is an eval-only release: the training entry points below are upstream's and are not
part of it.

---

## Installation

Requires Python ≥ 3.9 and a CUDA GPU. `transformers` must be exactly `4.57.1` (other versions are unverified).

```bash
pip install -e .                 # installs assets/requirements.txt
pip install flash-attn --no-build-isolation
```

`Qwen/Qwen3-VL-4B-Instruct` must be available from Hugging Face or the local cache (the VLM backbone used by the model).

---

## Checkpoint

Download the **[Xiaomi-Robotics-1-5B model](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)** checkpoint from Hugging Face and save it as `pretrained_ckpt/model_states.pt`.

---

## Data

Download the **[XR-1 post-training demo dataset](https://huggingface.co/datasets/XiaomiRobotics/xr1_post_train_demo/tree/main)** from Hugging Face. Each episode is one metadata JSON plus three synchronized videos (ego / wrist-left / wrist-right). See [`docs/data_format.md`](docs/data_format.md) for the full JSON schema.

The default config (`configs/data/load_washer.yaml`) expects five example episodes at `data/json1.json` … `data/json5.json`, with their videos under `data/videos/`. To train on your own data, edit that config:

1. Set `paths` to JSON files or directories (directories are scanned recursively).
2. Replace `mean` and `std` with the per-step mean and standard deviation of the packed relative actions. Both arrays must have shape `(action_length, action_dim)`; the defaults are `action_length = 30` and `action_dim = 60`.
3. Replace `q01` / `q99` with quantiles of the packed robot states. Both arrays must have shape `(state_length, state_dim)`; the defaults are `state_length = 1` and `state_dim = 60`.

Video paths are read from each JSON file. `crop_bbox` uses `(top, left, height, width)` and may be `null`.

---

## Training

Log in to Weights & Biases, then launch training via the launcher script:

```bash
wandb login

RESOURCE_GPU=1 bash scripts/train.sh \
  trainer.project="xiaomi-robotics-1" \
  trainer.exp_name="posttrain" \
  trainer.default_root_dir="outputs" \
  data=load_washer \
  model=posttrain \
  model.params.pretrained="pretrained_ckpt/model_states.pt"
```

`RESOURCE_GPU` is the number of local GPUs. For multi-node, the launcher also reads `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` directly. Asynchronous training is enabled by default; add `model.params.model.async_train=false` for synchronous training.

Checkpoints and the resolved config are written to `outputs/project_<project>/<exp_name>/` (e.g. `outputs/project_xiaomi-robotics-1/posttrain/`).

---

## Deployment

Serve a trained checkpoint with the inference server (requires `tmux`):

```bash
bash scripts/deploy.sh outputs/project_xiaomi-robotics-1/posttrain 1 1
```

The last two arguments are `<num_ports> <num_gpus>`; ports start at `10086`. Inspect the servers with `tmux attach -t model_servers`.

### Client API

`mibot/server/runtime/client.py` accepts three PIL images, an instruction, and the current robot state, and returns `raw_action`, `action_components`, and `action_targets`.

The robot state contains `left/right_arm_joint`, `left/right_gripper_pos`, `left/right_ee_pos`, and `left/right_ee_rotm` (`waist_pos` is optional). For asynchronous execution, pass an `(N, 60)` unnormalized action prefix via `action_prefix=...`; optional camera crops are three `(top, left, height, width)` tuples via `crop_bboxes=...`.

---

## License

Licensed under the [Apache License 2.0](LICENSE).