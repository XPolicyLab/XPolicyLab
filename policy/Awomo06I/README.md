# Awomo-0.6I

**Contributor:** Awomo-WAM Team | **Related report:** [Awomo-0.5 Technical Report](https://github.com/Awomo-WestlakeDI/Awomo-0.5/blob/main/Awomo_0_5.pdf) | **Original code:** [TianhengW/Awomo-0.6I](https://github.com/TianhengW/Awomo-0.6I) | **Model assets:** [Awomo-0.6I RoboTwin checkpoint](https://huggingface.co/QZWang/Awomo-0.6-RoboTwin-AP338)

Awomo-0.6I is an image-conditioned world-action model. This adapter targets RoboTwin dual-arm `joint` control and directly loads the AP338 100k-pretrained, 40k-SFT Power EMA checkpoint. It is an eval-only integration; a training adapter is not included and its release timeline is TBD.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard/).

## Installation

Provide the full historical runtime with its `Automodel/` directory. It is not yet bundled or publicly distributed. The installer creates a conda environment and installs XPolicyLab and the runtime package without resolving its full dependency set; install the remaining dependencies from the matching historical environment before real inference. The historical P6000 run used PyTorch 2.7.1 with CUDA 12.8. A complete portable dependency lock and real-checkpoint startup have not been verified.

```bash
cd /path/to/XPolicyLab/policy/Awomo06I
export AWOMO06I_RUNTIME_ROOT=/path/to/full-historical-runtime
bash install.sh awomo06i
conda activate awomo06i
```

## Model Assets

```bash
bash download_checkpoint.sh
bash download_base_models.sh
```

The checkpoint downloader pins the published model revision and verifies its 73 files against a pinned SHA-256 manifest. The base-model downloader prints `AWOMO06I_FLUX2_BASE` and `AWOMO06I_QWEN_VL` exports. FLUX.2 Klein base 4B, Qwen3-VL-4B-Instruct and RoboTwin assets have separate licenses.

## Data Processing

Not included in this eval-only adapter. XPolicyLab provides decoded RGB observations; the checkpoint contains the RoboTwin normalization statistics.

## Training

Training integration is not included in this eval-only submission; its release timeline is TBD.

## Evaluation

```bash
export AWOMO06I_RUNTIME_ROOT=/path/to/full-historical-runtime
export AWOMO06I_FLUX2_BASE=/path/to/FLUX.2-klein-base-4B
export AWOMO06I_QWEN_VL=/path/to/Qwen3-VL-4B-Instruct
bash eval.sh RoboTwin adjust_bottle Awomo06I arx_x5 joint 0 0 1 awomo06i RoboTwin
```

`AWOMO06I_CHECKPOINT_PATH` optionally overrides standard XPolicyLab checkpoint resolution. The model predicts 16 actions and executes the first 8; inference uses 16 denoising steps, shift 5, DiT Power EMA (`sigma_rel=0.1`) and LoRA EMA. Batch requests are processed serially.

For protocol-only debug, an explicit dummy policy returns zeros and does not load weights or measure model performance:

```bash
AWOMO06I_ALLOW_DUMMY_POLICY=true EVAL_ENV_TYPE=debug \
  bash eval.sh RoboTwin adjust_bottle Awomo06I arx_x5 joint 0 0 0 awomo06i base
AWOMO06I_ALLOW_DUMMY_POLICY=true EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 \
  bash eval.sh RoboTwin adjust_bottle Awomo06I arx_x5 joint 0 0 0 awomo06i base
```

## Reported C2R Result

| Clean | Randomized | Overall | Rollouts |
|---:|---:|---:|---:|
| 88.16% (4408/5000) | 84.78% (4239/5000) | **86.47% (8647/10000)** | 50 tasks × 2 conditions × 100 episodes |

Per-task counts are in `results/robotwin_c2r_100ep.json`; run `python verify_results.py` to recompute the aggregates. The historical evaluation used a 10-case historical prefix and 90 further expert-qualified cases per task-condition, with expert-stage exceptions skipped and recorded. Its equivalence to the official leaderboard protocol and the new adapter's real-checkpoint parity remain unverified.

## Notes

- The full historical runtime and matching dependency environment are required; they are not included in the public repository. The adapter checks 48 captured source-file hashes before loading, but those hashes do not verify the entire runtime.
- The normal and encoded debug loops have been exercised with the explicit dummy policy, not a real checkpoint or simulator. End-to-end model loading and simulator evaluation remain unverified.
