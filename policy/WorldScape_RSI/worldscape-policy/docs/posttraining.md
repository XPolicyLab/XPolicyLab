# Post-training

The public RoboDojo entrypoint is the adapter's `train.sh`. It selects
`recipes/posttrain/posttrain_robodojo.sh` and uses the shared torchrun/DeepSpeed
bootstrap in `recipes/common/launcher_common.sh`.

| Variant | Config | Geometry | Normalization |
| --- | --- | --- | --- |
| Flash | `configs/posttrain/robodojo.yaml` | H24/C1 | z-score |
| Memory | `configs/posttrain/robodojo_memory.yaml` | H24/C24 | z-score |

See the [adapter training guide](../../README.md#training)
for complete environment and data setup. Pretrained bimanual adapter row 4 is
copied into local RoboDojo row 0. A native training checkpoint can instead be
resumed with `training.resume`.

YAML `includes` resolve relative to their declaring file, then selector profiles
are applied, then command-line overrides. Shared optimizer and model settings
live in `common_wan22.yaml`. Use `RUN_STEPS` to bound a smoke test while preserving
the original learning-rate schedule.

The [hot-loading guide](robodojo_hotload.md) covers append-only live trajectories.
Both Flash and Memory retain their initial statistics throughout hot loading.

## Freeze-policy ownership

`freeze.config` is the only training-time ownership point for parameter
trainability. Conditioning/model fragments describe structure and checkpoint
paths only; model constructors do not apply their own freeze flags. A QFormer
created with `VLM_TOKEN_MODE=qformer` inherits the parent `vlm: true` rule by
default. Set `FREEZE_QFORMER=false` to add the more-specific trainable override.
In `last` mode QFormer is absent and no QFormer freeze rule is generated.

Each platform posttrain config also owns its evaluation transform metadata.
Successful saves make every `checkpoint-N/` directly evaluable and resumable.
Eval recipes select the highest complete checkpoint by default, while an
explicit task-specific eval model path still takes precedence. No duplicate
`final/` weights are written. Resume uses either fast mode (default, worker
prefetch retained) or exact mode (`RESUME_MODE=exact`, workers forced to zero).
