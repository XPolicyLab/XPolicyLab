# XBrain-v1 inference runtime

This repository is an inference-only runtime snapshot for XBrain-v1 RoboDojo
policies. It intentionally excludes training projects, training configs,
datasets, training launchers, experiment logs, and model weights.

The runtime is required because XBrain-v1 checkpoints use a modified GigaBrain
implementation and must not silently fall back to an arbitrary upstream
`giga-models` installation.

## Installation

Use the Python interpreter of the environment created with the public
GigaBrain installation instructions:

```bash
python -m pip install -e .
```

Then import the stable entrypoint:

```python
from xbrain_v1_runtime import GigaBrain0Pipeline, runtime_info
print(runtime_info())
```

`runtime_manifest.json` records the checkpoint-facing inference contract. The
model checkpoint, normalization statistics, PaliGemma tokenizer, and FAST
tokenizer are separate release resources and are not included here.

The current candidate release is described in `release_manifest.json` and
`RELEASE_0.1.0.md`. A later checkpoint should be published as a new release
after the same compatibility and policy-interface checks pass.

## RoboDojo robot resources

The three real-robot targets use the same 14-D external action contract but
different checkpoint and normalization resources. Set all paths explicitly at
deployment time:

```bash
export XBRAIN_TOKENIZER_PATH=/path/to/paligemma2-3b-pt-224
export XBRAIN_FAST_TOKENIZER_PATH=/path/to/fast

export XBRAIN_PIPERX_CHECKPOINT_PATH=/release/piper_x/model_ema
export XBRAIN_PIPERX_NORM_STATS_PATH=/release/norm/robodojo_piperx_0902.json

export XBRAIN_PIPER_CHECKPOINT_PATH=/release/piper/model_ema
export XBRAIN_PIPER_NORM_STATS_PATH=/release/norm/robodojo_piper_0903.json

export XBRAIN_ARX_X5_CHECKPOINT_PATH=/release/arx_x5/model_ema
export XBRAIN_ARX_X5_NORM_STATS_PATH=/release/norm/robodojo_arx5_0903.json
```

The logical mapping and shared contract are in
`resources/robodojo_models.yaml`. Validate all resource directories before
loading GPU weights:

```bash
python tests/validate_model_resources.py
```

Then validate a selected checkpoint on a CUDA device:

```bash
python tests/test_checkpoint_inference.py \
  --checkpoint "$XBRAIN_PIPERX_CHECKPOINT_PATH" \
  --norm-stats "$XBRAIN_PIPERX_NORM_STATS_PATH" \
  --tokenizer "$XBRAIN_TOKENIZER_PATH" \
  --fast-tokenizer "$XBRAIN_FAST_TOKENIZER_PATH" \
  --embodiment-id 6
```

## Scope

Included:

- model definitions and checkpoint loading required by `GigaBrain0Pipeline`;
- image, state, prompt, FAST-token, normalization, and action transforms;
- action decoding for the supported 14-D dual-arm RoboDojo contracts.

The package pins the inference-facing Transformers/Diffusers/PEFT compatibility
floor. Existing environments may already contain older training packages; run
the package installation so these runtime constraints are resolved together.

Excluded:

- `projects/` training scripts and configs;
- datasets and data conversion code;
- distributed training, optimizer, scheduler, and experiment tooling;
- checkpoints, tokenizer files, credentials, and private paths.
