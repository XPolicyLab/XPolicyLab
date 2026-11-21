# Native training

Training is configuration-driven through `wsp-train`. The bundled post-training
recipes launch through the shared torchrun/DeepSpeed bootstrap in
`recipes/common/launcher_common.sh`. They compose
`configs/posttrain/common_wan22.yaml`, `configs/model/wsp2_wan22_5b.yaml`, and
`configs/wam/wan22_5b.yaml`.

This adapter provides RoboDojo training entry points and retains
upstream AgileX task wrappers. Pretraining configurations and the LIBERO YAML
remain as upstream references, but their launch scripts are not included.
Use the [adapter training guide](../../README.md#training) for runnable
RoboDojo commands, or the [post-training guide](posttraining.md)
for the retained task wrappers.

RoboDojo train 14-D absolute joint actions with z-score
normalization. RoboDojo flash uses H24/C1 and memory uses H24/C24 by default.
The upstream AgileX recipes use absolute EEF actions. Action representation and
normalization are recorded in each exported checkpoint's transform metadata.

`PRETRAINED_MODEL_PATH` selects initialization weights. Continuing a training
run uses `training.resume`, which also restores optimizer and scheduler state.
RoboDojo requires an explicit resume when the output directory already contains
checkpoints and rejects resuming q99 training with z-score statistics.

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

RoboDojo post-training uses `posttrain_robodojo.sh` with `robodojo.yaml`
or `robodojo_memory.yaml`; see the [adapter guide](../../README.md).

Configuration overrides remain OmegaConf dot-list arguments after `--config`.
