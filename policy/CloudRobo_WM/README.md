# CloudRobo-WM

**Contributor:** cloudrobo | **Paper:** [Technical report](https://github.com/Liumouliu/CloudRobo-WM/blob/main/Cloudrobo_WM_20261016.pdf) | **arXiv:** Not released | **Original code:** [CloudRobo-WM-0](https://github.com/Liumouliu/CloudRobo-WM-0)

CR-WM targets **RoboTwin 2.0 / arx_x5 / joint**, one environment per server, using three RGB cameras and Joint14 state. Model code is included in `source_inference/` and `source_posttrain/`; they use separate environments because both provide the `crwm` package.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

Reported RoboTwin results: [c2c and c2r](#robotwin-results).

## Installation

Place this directory at `XPolicyLab/policy/CloudRobo_WM/`, inside the official RoboTwin workspace. Keep the parent workspace's `env_cfg/` and simulator installation. Inference uses Linux, Python 3.10–3.12 and a compatible BF16 runtime; retain the working inference environment when updating this adapter.

```bash
bash install.sh <inference|posttrain> --python /path/to/framework/python --offline

# Fresh inference environment
bash install.sh inference --python /opt/crwm-inference/bin/python
```

Post-training uses Linux, Python 3.12, PyTorch 2.9.1, torchvision 0.24.1 and a compatible CPU decoder exposing `decord`. Install the framework and its device extension for your chosen runtime first, then install the included training source independently:

```bash
cd source_posttrain
/opt/crwm-training/bin/python -m venv --system-site-packages .venv
.venv/bin/python tools/install_dependencies.py
cd ..
```

Application dependencies remain ordinary Python packages; framework builds and decoder binaries are supplied by the target environment. `source_posttrain/ENVIRONMENT.md` gives the decoder requirements. The installer preserves the selected framework builds. `--offline` does not disable pip network access; reuse an installed environment or configure a local package index when network-free dependency installation is required.

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  --data-root /path/to/robotwin3d --bundle-root /path/to/bundle [--download-assets]

bash process_data.sh RoboTwin crwm arx_x5 joint \
  --data-root /datasets/robotwin3d --bundle-root /datasets/crwm-bundle
```

This runs the native data precheck. Add `--download-assets` to fetch release training resources and metadata into the requested bundle/data roots first. It consumes RobotWin3D's LeRobot **v2.1** export from [flex-pi/robotwin_3d](https://huggingface.co/datasets/flex-pi/robotwin_3d), with parquet, RGB videos, packed lossless depth and camera intrinsics. It also requires the supplied raw-link EEF sidecars. These keys and media differ from XPolicyLab's official LeRobot converters; their output cannot replace this export. XPolicyLab encoded image buffers never enter this native reader.

`source_posttrain/data/clean_index.json` selects 2,500 original episodes: 50 tasks × 50 Clean demonstrations. Download only the original paths listed in `data/clean_files.txt`; keep global episode IDs and metadata unchanged. Raw media/parquet are separate from the asset release. Put release metadata in `DATA_ROOT/meta/` and training resources under the bundle root as shown below.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <device_ids|all> \
  --data-root /path/to/robotwin3d --bundle-root /path/to/bundle \
  [--output-dir /path/to/new_run] [--stop-after-steps N] [--resume /path/to/state] [--download-assets] \
  [-- <torch.distributed.run arguments...>]

bash train.sh RoboTwin crwm arx_x5 joint 42 all \
  --data-root /datasets/robotwin3d --bundle-root /datasets/crwm-bundle \
  --output-dir /outputs/crwm-check --stop-after-steps 5 --download-assets
```

`all` inherits the launch environment's device visibility. A numeric device list requires `CRWM_DEVICE_FILTER` to name that runtime's visibility variable. `CRWM_TRAIN_PYTHON` can select an existing training interpreter. Full training uses 68 ranks, batch 48 per rank and 5,504 updates; supply your own rendezvous arguments after `--` and omit `--stop-after-steps`. Outputs must be outside the bundle; `--resume` requires full state, not a tensor-only weight file.

The release provides `initialization_bf16.pt`, with 1,674 BF16 tensors. Its values are copied into FP32 trainable masters.

New training outputs are flat tensor dictionaries. The unchanged inference loader consumes the published nested `cr_wm.pt` with its contracts; new training outputs require a separately checked export before evaluation.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_python> <eval_env_conda_env>

CRWM_TASK_CONFIG=demo_clean CRWM_INSTRUCTION_TYPE=seen CRWM_TEST_NUM=100 \
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin adjust_bottle /models/crwm arx_x5 joint 42 \
  0 1 auto robotwin
```

`auto` uses `source_inference/.venv/bin/python`; the policy environment argument also accepts `active`, an interpreter/environment path, or a Conda name. Randomized/unseen evaluation uses `CRWM_TASK_CONFIG=demo_randomized CRWM_INSTRUCTION_TYPE=unseen`. The standard joint/gripper action keys and observation-update loop are retained. Batch evaluation is unsupported; auxiliary EEF output is available only through `get_eef_prediction()`.

### RoboTwin Results

Results from the 2026-10-18 **native/inprocess BF16** validation, using the same inference core and checkpoint as this adapter. Each profile contains 50 tasks × 100 frozen trials:

| Protocol | Successes / trials | Task success rate |
| --- | ---: | ---: |
| c2c: Clean-to-Clean / seen | 4445 / 5000 | **88.90%** |
| c2r: Clean-to-Randomized / unseen | 3750 / 5000 | **75.00%** |
| Equal-profile average | 8195 / 10000 | **81.95%** |

Protocol: BF16/eager inference, Euler/SDPA, four denoising steps and 32-action replans, with real observations acquired for every executed action. The audited set combines 479 unchanged earlier results from the same run and 9521 new trials, without overlap.

Separate WebSocket checks completed raw and encoded debug loops (10 episodes / 200 actions each), plus `adjust_bottle` simulator smoke tests: clean/seen 1/1 in 123 actions and randomized/unseen 0/1 at the 400-action limit. All four checks exited 0; full event logs and redaction details accompany the PR. The 10,000 benchmark trials used the native/in-process path.

Per-task results: [Markdown](results/cloudrobo-wm_robotwin_evaluation_result.md) and [Excel](results/cloudrobo-wm_robotwin_evaluation_result.xlsx).

## Model Assets

```bash
bash download_checkpoint.sh --output-dir /models/crwm
python download_release.py training --output-dir /datasets/crwm-bundle
python download_release.py metadata --output-dir /datasets/robotwin3d
```

Each file is SHA256-verified before publication to its destination. Existing mismatches stop the download. Use `--verify-only` to check local files, or `--source-dir /offline/hf-repository` to copy an offline release.

```text
/models/crwm/weights/cr_wm.pt
/models/crwm/weights/third_party/          # full inference backbones/tokenizer
/datasets/crwm-bundle/initialization_bf16.pt
/datasets/crwm-bundle/assets/              # training DINO and encoder-only VAE
/datasets/crwm-bundle/text_cache/
/datasets/crwm-bundle/eef_sidecars/
/datasets/robotwin3d/meta/
/datasets/robotwin3d/data/                  # native parquet, obtained separately
/datasets/robotwin3d/videos/                # native RGB/depth, obtained separately
```

The default inference downloader fetches `cr_wm.pt` and DINO from this public release. The full inference VAE and text encoder/tokenizer are absent from this asset repository; the inference manifests retain their pinned original providers for those six files.

## Configuration

Extra `deploy.yml` keys: `checkpoint_path` (default `null`, explicit asset root), `device` (inference runtime), `num_inference_steps` (`4`), `replan_steps` (`32`), and `request_timeout_s` (`120`). Architecture, normalization and kinematics remain in `source_inference/configs/`. `CRWM_INFERENCE_PYTHON` overrides the inference interpreter; `CRWM_PATH_PYTHON` selects the standard-library path resolver interpreter.

## Acknowledgements

CloudRobo-WM uses Wan2.2 for its video backbone and VAE, DINOv3 for semantic feature extraction, and Depth Anything 3 for depth estimation. The codebase structure draws on Fast-WAM and flex-pi. Details of the datasets used for CloudRobo-WM pre-training are provided in the accompanying paper. We thank the authors and contributors of these projects for making their work publicly available.
