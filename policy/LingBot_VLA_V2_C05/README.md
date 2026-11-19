# LingBot-VLA V2 C05

Contributor: [tianws](https://github.com/tianws).

C05 is a clean-only, 50-task fine-tune of
[LingBot-VLA V2](https://github.com/Robbyant/lingbot-vla-v2). It consumes RGB,
joint state and language; depth/video teachers are used during training only.
Supported: RoboTwin, `env_cfg_type=aloha_agilex`, `action_type=joint`.
Other robots are not supported merely because they have 14-dimensional state.

Checkpoint/source: [Aaron-Tian/lingbot-vla-v2-c05-robotwin](https://huggingface.co/Aaron-Tian/lingbot-vla-v2-c05-robotwin).
See its `TECHNICAL_REPORT.txt`, `historical_results.json` and `training_reference/`.
Public availability and the immutable download revision are recorded in the PR.

## Installation

Create an isolated Python 3.12 environment and run `bash install.sh` inside it.
Test target: Linux, CUDA 12.8, PyTorch 2.8.0, Transformers 4.57.3, L20 48GB.
Flash Attention satisfies upstream constructor checks; FP32 visual attention uses
eager attention. A compiler/toolkit may be needed to build Flash Attention.
Use a separate RoboTwin simulator environment.

Original FP32 weights are loaded before CUDA placement and checked tensor-for-tensor.
Token embeddings stay on CPU, without dtype conversion. Inference uses fixed-route
FP32 MoE accumulation, ten denoising steps and no compilation. These choices do not
promise bitwise equality across GPUs, BLAS libraries or simulator versions.

## Checkpoint

Run `python download_checkpoint.py --output checkpoints/c05 --revision <release-commit>`.
Use the immutable revision linked in the submission PR. All released files and
six weight shards are checked before the download is marked ready.

```text
checkpoints/c05/
  checkpoint_contract.json
  inference_template.json
  checkpoint/                         # six FP32 shards, config and index
  runtime/lingbot-vla-v2/              # pinned native policy source
  runtime/models/qwen3_vl_tokenizer/
  runtime/data/normalization.json
  runtime/parking_runtime/
  runtime/submission_setup/
  training_reference/
```

The adapter creates a temporary invocation YAML and checkpoint links; it does not
rewrite weights. Upstream revision and selected source hashes are in
`source_provenance.json`.

## Data Processing

Historical data: 2,500 official clean/seen demonstrations, 50 per task, totaling
548,893 frames in LeRobot **v3**. Native keys are `observation.images.cam_high`,
`cam_left_wrist`, `cam_right_wrist`, packed `observation.state`, `action`, and `task`.
They differ from the generic official XPolicyLab converter's per-arm keys; the
original robot mapping and normalizer are included. No finals real-robot data was used.

For new XPolicyLab datasets, start with the official
`scripts/transform_lerobot_v30_format.py`; adapt the robot YAML explicitly and
compute train-only statistics. Stored image bits must use the shared
`decode_image_bit` / `encode_image_bit`. This adapter receives decoded RGB and
does not swap channels or reinterpret JPEGs.

## Training

This submission is **eval-only at the XPolicyLab launcher boundary**. Training
algorithm excerpts, C01/C02/C05 configurations, seen-prompt pool, augmentation
constants, fixed sampling table and SHA provenance are released in
`training_reference/`. They are not a complete portable torchrun launcher.
Private platform launchers and environment archives are excluded.

A portable `process_data.sh` / `train.sh` integration is a follow-up after dataset,
teacher and distributed-resume validation; no untested reproduction command or
completion date is promised. This limitation is disclosed to maintainers in the PR.
The technical report documents the full training lineage and model selection.

## Evaluation

Prepare official RoboTwin metadata next to XPolicyLab. The ten standard arguments
are retained. The policy environment may be a Conda name or an absolute Python
environment directory. Debug clients also accept absolute environment directories.

```bash
cd XPolicyLab/policy/LingBot_VLA_V2_C05
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin stack_bowls_two /absolute/bundle \
  aloha_agilex joint 0 0 0 /absolute/policy-env /absolute/client-env
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug C05_ACTION_CHUNK=20 \
  bash eval.sh RoboTwin stack_bowls_two /absolute/bundle \
  aloha_agilex joint 0 0 0 /absolute/policy-env /absolute/client-env
```

For simulation use RoboTwin's current `scripts/eval_policy.sh` scheduler and the
official unmodified scene code. Set `C05_ACTION_CHUNK=32` for clean and `20` for
randomized before launching the server. Both use the same checkpoint, ten
denoising steps and a native 50-step horizon. No per-task tuning is performed.
`reset()` clears state and seeds the episode stream from the run seed; it does
not recreate the old competition evaluator's seed-selection protocol.
Batched calls use serial inference over the supplied active observations;
`eval_batch` defaults to false for the qualified single-environment path.

The native normalizer converts arm deltas relative to the **same current state**
back to absolute targets; grippers are absolute simulator openings. The adapter
must not accumulate deltas or normalize actions again.

## Results and limitations

Archived competition self-evaluation: clean **81.24%** (4062/5000), randomized
**59.62%** (2981/5000), mean **70.43%**. This is **not an accepted current
leaderboard score**. The historical protocol included mutable scene-boundary
accumulation, bounded expert selection and seen/unseen prompt selection; these can
affect scores. This adapter does not install that scene modification. Please
reproduce with the current official protocol.

Interface debug uses synthetic observations, not simulator successes. Test
receipts and any simulator smoke result are attached to the PR when available.
No single-component augmentation ablation is claimed.
