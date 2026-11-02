# G07-preview

**Contributor:** Galaxea AI | **Paper:** TBD | **arXiv:** TBD | **Original code:** training repository not yet public

**Model assets:** [G07-preview-RoboDojo](https://huggingface.co/yuanty/G07-preview-RoboDojo)

`G07_preview` is an autoregressive world-action model (WAM) built on [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano).
Each 16-action chunk is conditioned on generated future video; a sliding KV cache and the episode's first frame provide
temporal context. This **eval-only** adapter vendors its inference implementation under `src/galaxea_va/` and supports
RoboDojo simulation with `env_cfg_type=arx_x5`, `action_type=joint`, and checkpoint `step_020904`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Run from a RoboDojo workspace containing XPolicyLab and the official `env_cfg/` directory. CUDA 12.8 and a CUDA GPU are
required; `nvcc` must be available for the default compiled inference path. The download script also uses `sha256sum`.

```bash
cd XPolicyLab/policy/G07_preview
bash install.sh [policy_conda_env]

# Example
bash install.sh g07_preview
conda activate g07_preview
bash download_weights.sh
```

## Data Processing

Not included in this eval-only submission. Evaluation uses the robot layout and normalization parameters shipped in
`inference_config.yaml`; no training dataset or data-processing package is needed. `process_data.sh` is omitted.

## Training

Not included in this eval-only submission; `train.sh` is omitted. We commit to publicly release the training code,
evaluated checkpoint, and a technical blog **within three days after this entry is listed on the RoboDojo leaderboard**.
The release through XPolicyLab will also include deployment code, configuration files, and reproduction instructions.

## Evaluation

```bash
cd XPolicyLab/policy/G07_preview
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# RoboDojo simulation; both environments must be installed on this machine.
bash eval.sh RoboDojo press_by_number robodojo arx_x5 joint 0 0 1 g07_preview RoboDojo

# Official protocol checks without a simulator; use the policy environment for both sides.
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls robodojo arx_x5 joint 0 0 0 g07_preview g07_preview
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls robodojo arx_x5 joint 0 0 0 g07_preview g07_preview
```

For separate policy and simulator machines, use `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` as described
in the [Deployment Flow](../../README.md#-deployment-flow). One server holds one episode's state; use `eval_batch: false`
and a separate server for each parallel environment. Batch RPC methods explicitly report that batched execution is unsupported.
The client resizes raw RGB images to model input sizes; encoded images are forwarded unchanged and decoded by XPolicyLab.

## Model Assets

`download_weights.sh` retrieves one pinned revision of the model repository: the evaluated checkpoint with its
frozen configuration, plus unmodified mirrors of the base-model files it loads, each with its license and notices:

```text
checkpoints/
├── Cosmos3-Nano/                         # transformer/, text_tokenizer/, vision_encoder/ (OpenMDW-1.1)
├── DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors   # Apache-2.0
└── robodojo/                             # Apache-2.0
    ├── checkpoints/weights/step_020904.pt
    └── inference_config.yaml
```

The Cosmos3-Nano files are [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) at revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, and the VAE is the
[DiffSynth-Studio conversion](https://modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors) of the
[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) VAE. The G07-preview checkpoint is fine-tuned from Cosmos3-Nano and
released under the Apache License 2.0; its directory keeps the OpenMDW-1.1 copy and notice required by Cosmos3-Nano.

`ckpt_name=robodojo` uses this layout. Other run-directory names and absolute or policy-relative paths are resolved by
`XPolicyLab.utils.checkpoint_resolver`; each selected directory must contain these same two G07-preview files.
All tasks and evaluation seeds use the same checkpoint and deployment settings.

## Configuration

Stable inference defaults live in `deploy.yml`; the setup scripts supply the standard per-launch fields.

| Keys | Meaning |
| --- | --- |
| `device`, `rand_device` | Model runs on CUDA; diffusion noise is sampled on CPU. |
| `action_infer_mode`, `replan_steps` | `idm` inference with 16 actions per model chunk; the adapter requests observations every four actions. |
| `num_action_denoise_steps`, `num_video_denoise_steps` | Ten solver steps for actions and video. |
| `sigma_shift`, `negative_prompt` | `null` uses the frozen scheduler configuration; the negative prompt defaults to empty. |
| `compile_action_infer` | Defaults to `true`. Startup warms all configured memory settings before accepting requests and can take tens of minutes; `false` is useful for debugging. |
| `timing_enabled` | Enables model timing; defaults to `false`. |
| `gripper_latch` | Enables per-arm gripper postprocessing, with full range read from the frozen robot configuration. |
| `infer_sliding_window`, `infer_reset_every`, `infer_reset_keep_chunks` | Attention window, cache reset period, and retained context, measured in 16-action chunks. |
| `memory_context` | Long-memory values of the same three settings used by the policy. |

## Notes

- This package supports the published RoboDojo ARX X5 joint-control checkpoint. Real-world deployment and other embodiments
  are outside this submission's scope.
- The evaluated checkpoint and its `inference_config.yaml` must be kept together.
- Local evaluation evidence and adapter validation are supplied with the submission; official leaderboard scores are
  determined by the RoboDojo evaluation system.

## License

The adapter code and the G07-preview checkpoint are released under the [Apache License 2.0](LICENSE). Five vendored modules are derived from
transformers, diffusers and DiffSynth-Studio; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Downloaded model
files keep the licenses stored beside them under `checkpoints/`.
