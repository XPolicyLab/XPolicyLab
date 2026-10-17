# JAM (Wan) / FastWAM — Piper X

**Contributor:** JAM | **Paper:** [Vision-Language-Action or World-Action? Just Action Models](https://jam-just-action-models.y0uroy.chatgpt.site/assets/JAM_paper.pdf) (public manuscript) | **arXiv:** Not listed | **Original code:** [JAM@0eaec826](https://github.com/y0urOy/JAM/tree/0eaec826faba36d6f2e447a3c7cc6d6db1dd1a71), [FastWAM upstream](https://github.com/yuantianyuan01/FastWAM)

One adapter serves two separately trained, fixed Piper X six-task policies: **JAM-Wan** and the **FastWAM-PiperX baseline**. `runtime/` vendors the implementation used by both candidates, with format-only cleanup recorded in `PROVENANCE.json`. Supported: `bench_name=RoboDojo`, `env_cfg_type=piper_x`, `action_type=joint`. This submission requests a staged Piper X comparison; real-robot trials are **0**, success rate **null/pending**.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, Python 3.10, PyTorch 2.7.1 / torchvision 0.22.1 and CUDA 12.8 wheels. Each policy runs in an isolated venv and separate server process. No inherited Miyabi/FastWAM environment or PBS job is required for installation or serving.

```bash
# From XPolicyLab/policy/JAM. Create only small workspace templates.
python3.10 setup_workspace.py
bash install.sh --venv /opt/venvs/jam-piperx --mode runtime
bash install.sh --venv /opt/venvs/fastwam-piperx --mode runtime
# Select one immutable policy; use fastwam for the other candidate.
/opt/venvs/jam-piperx/bin/python assets.py download-public --method jam
/opt/venvs/jam-piperx/bin/python assets.py verify --method jam
/opt/venvs/jam-piperx/bin/python doctor.py
```

`install.sh --dry-run` prints the install steps. Runtime dependencies are pinned in `requirements/runtime.txt`; data and training use separate profiles. The compiler, Python development headers, FFmpeg and CUDA toolkit may additionally be needed for conversion/training. Fresh platform CUDA installation and hardware latency remain unverified.

`setup_workspace.py` adds `models/{jam,fastwam}` and `env_cfg/` in the **parent** workspace without replacing conflicting files. Piper X dimensions are also registered in the framework's `utils/robot/_robot_info.json`. If an existing platform config differs, retain it and resolve the mismatch with the platform.

## Data Processing

Evaluation of the supplied weights needs no training dataset. Optional reproduction uses `RoboDojo-Benchmark/Robodojo@35efbc7dedfdbeeb6e95fb749bd885d73d483e41`: six tasks, 100 episodes each, per-task 90/10 split with seed 42 (540 train / 60 held-out). Statistics use only training episodes.

The prepared **LeRobot v2.1** export matches official converter keys: `action`, `observation.state`, and `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}`. `runtime/scripts/robodojo_piperx_prepare.py` calls the official `scripts/transform_lerobot_v21_format.py`; source HDF5 image buffers decode only through `decode_image_bit`. The optional encoder produces CPU H.264 CRF18/GOP2 from the converter's RGB PNG frames. PIL reads those intermediate PNGs, never raw XPolicyLab trajectory buffers. All 640×480, 25 Hz frames are retained.

The reproduction wrappers use native named arguments:

```bash
# Template: POLICY_PYTHON=PYTHON bash process_data.sh --workspace NEW_DATA_ROOT [--dry-run]
bash install.sh --venv /opt/venvs/piperx-data --mode data
/opt/venvs/piperx-data/bin/python workflow.py download --workspace /data/piperx-repro
POLICY_PYTHON=/opt/venvs/piperx-data/bin/python bash process_data.sh --workspace /data/piperx-repro
/opt/venvs/piperx-data/bin/python workflow.py stats --workspace /data/piperx-repro
```

The wrapper links the official converter registration and parent `env_cfg/` into the new data workspace. It does not require `PBS_JOBID` or submit jobs. Add `--dry-run` to inspect commands without downloading or converting.

## Training

Full native training code is included; portable retraining was **not rerun** for this PR. The two supplied candidates already completed 28,630 updates. JAM uses action flow matching plus cached VAE-direct future-feature targets (offsets [1,2], layer 18, weight 0.6); FastWAM is the matched native video-denoising baseline. The training-only foresight branch is absent from action inference.

```bash
# Template: POLICY_PYTHON=PYTHON bash train.sh --method jam|fastwam \
#   --workspace NEW_DATA_ROOT --output NEW_RUN_DIR --nodes N --gpus-per-node G [--dry-run]
bash install.sh --venv /opt/venvs/piperx-train --mode training
/opt/venvs/piperx-train/bin/python workflow.py text --workspace /data/piperx-repro
/opt/venvs/piperx-train/bin/python workflow.py cache --workspace /data/piperx-repro --gpus-per-node 8
POLICY_PYTHON=/opt/venvs/piperx-train/bin/python bash train.sh \
  --method jam --workspace /data/piperx-repro --output /data/new-runs/jam-seed42 \
  --gpus-per-node 8 --nodes 1 --dry-run
```

Use `--method fastwam` with a different output directory for the baseline. The fixed recipe is world size 8, batch 16/GPU, global batch 128, seed 42, 10 epochs / 28,630 steps. Multi-node flags are `--node-rank`, `--master-addr`, `--master-port`; eight one-GPU hosts use `--nodes 8 --gpus-per-node 1`. Output must be new; WandB is disabled. `train.sh` invokes Accelerate/DeepSpeed directly and never calls `qsub`; use your site's scheduler for GPU allocation.

## Evaluation

Bundles contain unchanged `dataset_stats.json` and fixed `deployment.json`, with paths relative to the manifest. The shared checkpoint resolver accepts these explicit bundle directories.

```bash
# Standard template:
# bash eval.sh BENCH TASK BUNDLE ROBOT ACTION SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV
WORKSPACE="$(cd ../../.. && pwd)"
# Local raw/encoded interface checks with real weights (requires a policy GPU):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cap_pen "$WORKSPACE/models/jam" \
  piper_x joint 42 0 0 /opt/venvs/jam-piperx unused
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo cap_pen "$WORKSPACE/models/jam" \
  piper_x joint 42 0 0 /opt/venvs/jam-piperx unused
# Baseline: same task, separate checkpoint/environment/process.
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cap_pen "$WORKSPACE/models/fastwam" \
  piper_x joint 42 0 0 /opt/venvs/fastwam-piperx unused
# Policy-server only, loopback until the official remote protocol is supplied:
bash setup_eval_policy_server.sh RoboDojo cap_pen "$WORKSPACE/models/jam" \
  piper_x joint 42 0 /opt/venvs/jam-piperx 9010 127.0.0.1
# Equivalent short form: bash serve.sh jam cap_pen /opt/venvs/jam-piperx 9010 127.0.0.1 0
```

`eval.sh` follows the ten-argument server/wait/client/cleanup flow. It supports the simulator-free debug client only; `EVAL_ENV_TYPE=sim` is rejected because these checkpoints are for real Piper X, not ARX X5 simulation. The trailing eval-env argument is accepted for signature compatibility; the debug client uses the policy venv. `setup_eval_env_client.sh` retains the standard generic launcher but is not an approved remote robot connection. No robot client, tunnel or hardware controller is started here.

## Model Assets

`assets_manifest.json` pins every file's repository commit, size and SHA256. `assets.py download-public --method jam|fastwam` uses Hugging Face's resumable local-directory downloads; `--dry-run` lists paths without network or writes. The selected policy needs about 48.3–48.4 GB including upstream/shared assets; both policies total 60,468,766,082 bytes, plus environments and download metadata. Weights are not committed to Git.

| File | Fixed Hugging Face revision | Bytes | SHA256 |
|---|---|---:|---|
| JAM `step_028630.pt` | [y0urOy/JAM-Wan-PiperX@7318a3e5](https://huggingface.co/y0urOy/JAM-Wan-PiperX/tree/7318a3e575e30920501a45f731a7f6fcb739cd4c) | 12183478777 | `9b8e4bbe1f8093d34886313ad2dd8a355276c660014bec84723c08d6d68c85a6` |
| FastWAM `step_028630.pt` | [y0urOy/FastWAM-PiperX@b1a32213](https://huggingface.co/y0urOy/FastWAM-PiperX/tree/b1a322132d6f4b6c94c952dbf81c3f210accfb9e) | 12041813369 | `383863b3c0f6d387e6a6cafcb34e2a65e52f11ebaf637d4ddf71b25cc9a86511` |
| Shared ActionDiT initialization | [JAM-Wan-PiperX@7318a3e5](https://huggingface.co/y0urOy/JAM-Wan-PiperX/tree/7318a3e575e30920501a45f731a7f6fcb739cd4c) | 2042025883 | `7bd65e1986accaaf2e8dd5e59b5e98ef348d7c2f1d88ab2df010eb954a441445` |

Upstream: `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e` for DiT/VAE/UMT5 and `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` for the tokenizer. Keep the exact supplied ActionDiT initializer; its historical derivation command was not independently recovered. Do not regenerate it when serving.

## Configuration

Both registrations use one immutable checkpoint across all six tasks, seed 42, bf16, 10 inference steps, horizon 32 and replan 24. There is no task-dependent policy/checkpoint routing. `policy_name: JAM` is the shared import name; each registration selects its own manifest before startup.

Extra `deploy.yml` keys: `replan_steps: 24` is checked against the manifest; `request_timeout_s: 120` is the client timeout. Standard host/port and per-run keys follow the reference adapter. `FASTWAM_ROOT` must be unset (or match this vendored runtime); `JAM_DATA_ROOT` changes optional data paths only. Use `workflow.py` for archived script defaults. Source hashes and deliberate portability changes are in `PROVENANCE.json`.

Input: three decoded RGB uint8 HWC cameras, current 14D joint state and nonempty instruction. Output: 24 absolute joint-target dictionaries, left arm 6 / left gripper 1 / right arm 6 / right gripper 1. Dataset rate is 25 Hz. Robot-side units, gripper mapping, control cadence, latency, reset/stop behavior and the official client remain subject to platform confirmation.

## Notes

JAM's methodological contribution is future-feature supervision with a discarded auxiliary branch and no visual generation in its training/inference. The [public manuscript](https://jam-just-action-models.y0uroy.chatgpt.site/assets/JAM_paper.pdf) and [method page](https://jam-just-action-models.y0uroy.chatgpt.site/) describe this rationale and non-RoboDojo experiments. FastWAM is submitted as a paired baseline, not a claim of new FastWAM methodology. Data use the existing public RoboDojo dataset; no new dataset contribution is claimed.

Both Real robot registrations displayed Application received on 2026-10-16 (desktop page verification). Maintainer review, platform acceptance of a six-task/two-policy staged comparison, trial counts/seeds/success definitions, hardware and scheduling are pending. This Piper X-only scope does not satisfy the full verified real leaderboard's three-embodiment coverage, hidden-layout validation and public-video requirements. A special report or separate listing, public result/video scope and publication timing require platform confirmation; no listing/date is promised. The 240 held-out action chunks per model are not robot trials.

Validation commands and dated evidence are in [VALIDATION.md](VALIDATION.md). CPU tests can run without torch, checkpoints or robot access:

```bash
python3.10 setup_workspace.py
python3.10 -m venv /tmp/jam-cpu-check
/tmp/jam-cpu-check/bin/python -m pip install -r requirements/test.txt
/tmp/jam-cpu-check/bin/python tests/test_contracts.py
/tmp/jam-cpu-check/bin/python tests/test_transport.py
```

The transport suite uses an explicit deterministic test double for neural inference; it does not validate the supplied weights. Historical actual-weight GH200 evidence is documented separately. Fresh platform GPU loading and real-robot success remain unverified.
