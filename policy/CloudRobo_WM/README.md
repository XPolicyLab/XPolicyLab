# CloudRobo-WM — RoboTwin 2.0 adapter

**Contributor:** cloudrobo | **Paper:** [Technical report](https://github.com/Barry-Liang/CloudRobo-WM/blob/main/Cloudrobo_WM_20261113.pdf) | **arXiv:** Not released | **Original code:** [CloudRobo-WM](https://github.com/Barry-Liang/CloudRobo-WM)

CloudRobo-WM targets **RoboTwin 2.0 / arx_x5 / joint**, using three RGB cameras and Joint14 state. Model code is included in `source_inference/` and `posttrain/src/`;

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

Reported RoboTwin results: [Clean and Random](#results).

## Installation

Place this directory at `XPolicyLab/policy/CloudRobo_WM/`, inside the official RoboTwin workspace. The packaged deployment and interface declarations both use `CloudRobo_WM`; run the static check after installation. The setup scripts derive their import name from the installed directory. Keep the parent workspace's `env_cfg/` and simulator installation. Inference uses Linux, Python 3.10–3.12 and a compatible BF16 runtime; retain the working inference environment when updating this adapter.

```bash
bash install.sh --python /path/to/framework/python [--offline]

# Fresh inference environment
bash install.sh --python /opt/crwm-inference/bin/python
bash smoke_test.sh --static
```

For the separate Ascend post-training environment, see [post-training installation](posttrain/README.md#installation).

Application dependencies remain ordinary Python packages, framework builds and decoder binaries are supplied by the target environment. The inference installer preserves the selected framework builds. `--offline` does not disable pip network access; reuse an installed environment or configure a local package index when network-free dependency installation is required.

## Data Processing

See [post-training data preparation](posttrain/README.md#data-preparation) for dataset download, the Clean index and statistics, required resources, and the standard `download_data.sh` / `process_data.sh` commands. Run those commands from this policy directory.

## Training

`train.sh` is the standard XPolicyLab entrypoint for the included Ascend post-training implementation:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <npu_id> \
  [--mode train|smoke|dry-run]
```

See [training launch](posttrain/README.md#training-launch) for the required environment variables, logical NPU IDs, distributed recipe and save/resume check. [Recovery and logs](posttrain/README.md#recovery-and-logs) describes saved states, exports and diagnostics.

## Evaluation

`eval.sh` uses the official XPolicyLab online client integration: it delegates through XPolicyLab's `utils/setup_env_client.sh`, sets `expert_check=true`, and supports the standard debug mode. Install inside the official RoboTwin workspace and use the current XPolicyLab and RoboTwin environment setup.

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_python> <eval_env_conda_env>

CLOUDROBO_TASK_CONFIG=demo_clean CLOUDROBO_INSTRUCTION_TYPE=seen CLOUDROBO_TEST_NUM=100 \
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin adjust_bottle /models/cloudrobo-wm arx_x5 joint 0 \
  0 1 auto robotwin
```

`auto` selects `source_inference/.venv/bin/python`; the policy environment also accepts `active`, an interpreter/environment path, or a Conda name. Random/unseen uses `CLOUDROBO_TASK_CONFIG=demo_randomized CLOUDROBO_INSTRUCTION_TYPE=unseen`. `ROBOTWIN_EVAL_ARGS_FILE` follows the official client convention; when present it supplies the task profile/count settings instead of those environment variables. One environment per server is supported (`eval_batch=false`). The CLI scene seed is 0; model inference seed is separately fixed to 42. Depth/point-map inputs are zero; RGB remains active. For faster evaluation, we provide the independent saved-input entrypoint called **`eval_fixed.sh`** .
## Results
The completed evaluation contains 50 tasks × 100 frozen trials per profile, with all successes and failures included:

| Profile | Successes / trials | Success rate |
| --- | ---: | ---: |
| Clean / seen | 4764 / 5000 | **95.28%** |
| Random / unseen | 4134 / 5000 | **82.68%** |
| Equal-profile average | 8898 / 10000 | **88.98%** |


Inference used the main BF16 checkpoint below, four denoising steps and 32-action replans. The evidence includes all 100 scored task executions and all 10,000 inputs/outcomes, simulator evaluation logs, policy event logs, native per-task results and a file integrity manifest.
Separate WebSocket checks through [`eval.sh`](eval.sh) completed raw and encoded debug loops (10 episodes / 200 actions each), plus an official online `stack_bowls_two` simulator smoke test: clean/seen 1/1 in 313 actions, with expert checking enabled. All three checks exited with code 0, with no tracebacks. Full commands, logs and exit records are included in [`runtime_validation_20261112`](evidence/runtime_validation_20261112/).

Per-task results: [Markdown](results/evaluation_results.md), [JSON](results/evaluation_results.json). Evidence: [manifest](evidence/PUBLIC_LOG_MANIFEST.json), [archive](evidence/websocket_8898_full_logs.tar.gz), [SHA256](evidence/websocket_8898_full_logs.tar.gz.sha256).


## Model Assets

```bash
bash download_checkpoint.sh --output-dir /models/cloudrobo-wm
bash download_checkpoint.sh --output-dir /models/cloudrobo-wm --source-dir /offline/release
bash download_checkpoint.sh --output-dir /models/cloudrobo-wm --verify-only
```

The main model and initialization downloads are publicly accessible. Each file is SHA256-verified before publication to its destination. Existing mismatches stop the download. Use `--verify-only` to check local files, or `--source-dir /offline/release` to copy an offline release.

 The [pinned download](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/5dfe77637bc7b6636a1190e1709adbd64c157aa1/model.safetensors) for [DINO](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m) is named `model.safetensors` upstream; `download_checkpoint.py` automatically saves it as `weights/third_party/dino.safetensors` using the manifest destination path and verifies its size and SHA256 before publishing the file.

```text
/models/cloudrobo-wm/weights/cloudrobo_wm_bf16.safetensors
/models/cloudrobo-wm/weights/third_party/  # full inference backbones/tokenizer
```

 Public weights are pinned to a verified Hugging Face commit in the manifests:

- [Evaluated post-training model](https://huggingface.co/CloudRobo/CloudRobo-WM-1.0-RoboTwin-20261111/blob/main/robotwin-posttrain/cloudrobo_wm_bf16.safetensors): SHA256 `660bc6bd55d526366ea0f4abccb660d1b5aa1a29c8af1c7bfb1566a4b9e0462d`.
- [Post-training initialization](https://huggingface.co/CloudRobo/CloudRobo-WM-1.0-RoboTwin-20261111/blob/main/pretrain/initialization_bf16.safetensors): 14,133,497,112 bytes, SHA256 `9e105d2f047dfa482dfd172ea912ffbcedd8ae5de35143ffe91c502076160a59`. It is not the evaluated model.

```bash
bash download_initialization.sh --output-dir /models/cloudrobo-initialization
# Offline verification also creates the initialization receipt needed by the training launcher.
bash download_initialization.sh --output-dir /models/cloudrobo-initialization --verify-only
```

The initialization downloader verifies complete-file SHA256 and the 2,509-tensor BF16 header before creating its `.manifest.json` receipt. This verifies the input artifact, not training or save/resume acceptance. The inference manifests retain their pinned original providers for third-party assets.

## Configuration

Extra `deploy.yml` keys: `checkpoint_path` (default `null`, explicit asset root), `device` (`cuda`), `num_inference_steps` (`4`), `replan_steps` (`32`), `inference_seed` (`42`), and `request_timeout_s` (`120`). Configuration files are included in `source_inference/configs/`. `CLOUDROBO_INFERENCE_PYTHON` overrides the inference interpreter.

The task/instruction/trial-count environment variables above apply to direct `eval.sh`; scheduler-provided argument files take precedence.

## Acknowledgements

CloudRobo-WM uses Wan2.2 for its video backbone and VAE, DINO for semantic feature extraction, and Depth Anything 3 for depth estimation. The codebase structure draws on Fast-WAM and flex-pi. Details of the datasets used for CloudRobo-WM pre-training are provided in the accompanying paper. We thank the authors and contributors of these projects for making their work publicly available.
