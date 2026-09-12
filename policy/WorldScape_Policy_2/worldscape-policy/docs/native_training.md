# Native training

Training is configuration-driven through `wsp-train`. All pretrain and
posttrain recipes launch multi-node jobs directly through the shared torchrun
contract in `recipes/common/launcher_common.sh`. The single WAM supports T2VA,
goal-image-to-VA, and video-to-VA samples in one pretraining mixture.
Pretraining and post-training both include `configs/model/wsp2_wan22_5b.yaml`,
the canonical model builder, and `configs/wam/wan22_5b.yaml`, the shared
Wan2.2/DreamZero `2 frame / 24 action / 1 state` noise-kernel geometry.
Distributed defaults and DeepSpeed profiles live in
`configs/pretrain/common_wan22.yaml` and `configs/posttrain/common_wan22.yaml`.

Stage1 is Interactive-only and event-memory-free. It can resume a complete WSP
checkpoint, or bootstrap Wan2.2 TI2V-5B DiT shards, T5/VAE `.pth` files, and
the Wan2.1 CLIP `.pth` directly:

```bash
WAN_CKPT_DIR=/models/Wan2.2-TI2V-5B TOKENIZER_DIR=/models/umt5-xxl \
CLIP_CKPT_DIR=/models/Wan2.1-I2V-14B-480P T2VA_DATA_ROOT=/data/text \
GOAL_IMAGE_DATA_ROOT=/data/goal VIDEO_DATA_ROOT=/data/video \
MLP_WORKER_NUM=2 MLP_ROLE_INDEX=0 MLP_WORKER_0_HOST=10.0.0.1 NUM_GPUS=8 \
./recipes/pretrain/pretrain_mixed_three_mode_stage1.sh
```

To initialize Stage1, set `PRETRAINED_MODEL_PATH` to a complete native WSP
checkpoint bundle. If raw component paths are also supplied, raw tensors are
loaded first and the checkpoint overlays them.

Stage2 constructs the Auto graph and strictly overlays the Stage1 portable
policy artifact. Missing shared WAM, T5, visual-codec, image-encoder, or adapter
weights fail. Only Qwen, event-memory, and new Auto projector/norm keys may be
absent; Qwen uses its existing `from_pretrained` path and the remaining new
modules use random initialization. QFormer exists only when
`VLM_TOKEN_MODE=qformer`.

```bash
PRETRAINED_MODEL_PATH=/checkpoints/stage1/checkpoint-50000 \
Qwen_CKPT_DIR=/models/Qwen3-VL-4B TOKENIZER_DIR=/models/umt5-xxl \
T2VA_DATA_ROOT=/data/text GOAL_IMAGE_DATA_ROOT=/data/goal \
VIDEO_DATA_ROOT=/data/video ./recipes/pretrain/pretrain_mixed_three_mode_stage2.sh
```

The same native checkpoint contract applies to Stage2 and every posttraining
recipe. It controls initialization only; exact/fast trainer resume is selected
by `training.resume` and does not use `PRETRAINED_MODEL_PATH`.

The public action-space switch is `ACTION_MODE=eef`; EEF is currently the only
supported mode. Pretraining defaults to `RELATIVE_ACTION=true`: position and
rotation are expressed relative to each 24-step chunk's anchor state while both
gripper values remain absolute. Relative rotation uses
`R_relative = R_anchor^T R_action`, while position uses the direct delta
`p_relative = p_action - p_anchor`. Inference restores them with
`R_action = R_anchor R_relative` and
`p_action = p_anchor + p_relative`. Posttraining
defaults to `RELATIVE_ACTION=false` and therefore predicts absolute EEF poses.
Native export records the same absolute/relative field semantics for
evaluation-time restoration.

AgileX future-video diffusion uses DreamZero's three-view mosaic before VAE
encoding: head in the upper-left, right wrist in the upper-right, left wrist in
the lower-left, and a black lower-right tile. The mosaic is normalized and
bilinearly resized back to one-view resolution. Qwen receives only the head
view, and goal-image/demo-video persistent context is also encoded from the
head view only.

The final source priority is always **WSP checkpoint > raw component >
random initialization**. The builder logs and attaches an
`initialization_report`. Every new trainer checkpoint is a
`checkpoint-N/` directory. The canonical portable policy uses Hugging Face
`model.safetensors.index.json` plus `model-00001-of-0000N.safetensors` shards.
The default maximum shard size is 5 GB and can be changed with
`CHECKPOINT_MAX_SHARD_SIZE`; models below the threshold use
`model.safetensors`. `trainer_state.pt`, `rank-*.pt`, and the optional
`deepspeed/` directory carry resume-only state. Readers continue to accept
legacy single-file `model.safetensors`, `step-N.pt`, `step-N/`, and `policy.pt`
artifacts.

At launch, `worldscape_policy.cli.train` resolves checkpoint sources in this
order:

1. **Explicit or auto-resume** — `training.resume`, or the latest
   `checkpoint-*` / legacy `step-*` artifact under
   `training.checkpoint_dir` (`OUTPUT_DIR`).
2. **Pretrained policy** — `PRETRAINED_MODEL_PATH` when set and not `none`.
3. **Base components** — WAN/T5/CLIP/VAE/Qwen paths from the launcher env.

When auto-resume finds an output checkpoint, model init uses components and the
trainer reloads model, optimizer, scheduler, scaler, callback, and distributed
state. `RESUME_MODE=fast` is the default: it retains configured DataLoader
workers but starts a fresh data iterator, so sample order is not bitwise
reproducible. `RESUME_MODE=exact` restores sampler/data position and
automatically sets `data_loader.num_workers=0`, because worker RNG and prefetch
queues cannot be serialized. Fresh runs always retain the configured workers.

Each post-training `checkpoint-N/` is also enriched into an evaluation-ready
native bundle by default (`native_export.every_checkpoint=true`). The same
directory contains the portable model index and shards, `config.json`, native
model/generation configuration, normalization, `transform_bundle.json`,
provenance, the manifest expected by `wsp-eval`, and trainer/DeepSpeed state for
resume. The manifest checksums the index and every shard; missing or modified
files fail closed. Training ensures the terminal step has a `checkpoint-N/`; no
duplicate `final/` weights are written.

`configs/posttrain/agilex.yaml` is the only AgileX post-training config.
AgileX task recipes select its `mode`, `visual_prompt`, and `dataset_name`
profiles for **single-task expert** models:

```bash
DATA_ROOT=/data/build-block PRETRAINED_MODEL_PATH=/models/wsp \
./recipes/posttrain/posttrain_agilex_build_block_goal.sh
```

**Generalist** multi-task post-training uses `posttrain_libero.sh` or
`posttrain_robotwin2.sh` with their platform YAMLs.

Configuration overrides remain OmegaConf dot-list arguments after `--config`.
