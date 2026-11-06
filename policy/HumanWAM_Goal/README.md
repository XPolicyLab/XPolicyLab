# HumanWAM-Goal

**Contributor:** Open-HumanWAM team, DAGroup (PKU) | **Paper:** [Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models](https://dagroup-pku.github.io/open-humanwam/) | **arXiv:** [2611.06817](https://arxiv.org/abs/2611.06817) | **Original code:** [Open-HumanWAM / humanwam_goal](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale/humanwam_goal) | **Weights:** [HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin)

HumanWAM-Goal is a goal-image-conditioned world action model: it jointly denoises future video and a 48-step action chunk, conditioned on the current observation, a short observation history (0.5 s and 2 s), the instruction and one goal image of the task's completed state. This adapter serves the released RoboTwin 2.0 Aloha-AgileX checkpoint (joint control, 32-step replanning). Goal images come from the released 10,000-case goal cache; each was generated from the case's **initial head-camera image and instruction only**. The model, geometry and evaluation session live in the released `humanwam_goal` package (import name `goalwam`), which `install.sh` installs; nothing is vendored here.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd policy/HumanWAM_Goal
bash install.sh                 # released code at the pinned revision + policy environment in env/
python download_checkpoint.py --ckpt-name HumanWAM-Goal-RoboTwin
```

`install.sh` clones the Open-HumanWAM code at the revision pinned in `release.json` into `policy/HumanWAM_Goal/OpenHumanWAM` (set `HWGOAL_REPO` to use an existing checkout) and creates the policy environment `env/` with `humanwam_goal/scripts/setup_environment.sh` (Python 3.13, PyTorch 2.10.0+cu128, exact dependency pins; FlashAttention-3 needs glibc ≥ 2.34), then installs XPolicyLab into it. Pass `env` as the policy environment below. The simulator side uses the standard RoboTwin environment.

The environment also contains cuBLASLt 12.8.3.14 (`env/runtime/cublaslt-12.8.3.14/`), taken from the official `nvidia-cublas-cu12==12.8.3.14` wheel and checked against the SHA-256 pinned in `release.json`. The reported results were served with this library; PyTorch 2.10's own cuBLAS 12.8.4.1 selects different BF16 GEMM kernels and changes the actions slightly. The policy server preloads it (`LD_PRELOAD`) and, before loading weights, checks that exactly this library is mapped into the process; the run's `server/runtime.json` records it with the PyTorch and CUDA versions. A missing, different or duplicate library stops startup. Set `HWGOAL_CUBLASLT` to use another verified copy.

`download_checkpoint.py` downloads the checkpoint and the goal cache from [DAGroup-PKU/HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin) at the revision pinned in `release.json`, checks every checkpoint file against its `export.json` (whose digest `release.json` pins) and the goal cache against its pinned manifest digest, and links them under `checkpoints/` and `assets/`.

XPolicyLab must be the checkout inside the RoboTwin workspace (`RoboTwin/XPolicyLab`), so that its `env_cfg` lookup resolves to RoboTwin's robot metadata.

## Data Processing

Not supported yet: this is an **eval-only** submission, so `process_data.sh` is omitted. The released checkpoint was post-trained on the official RoboTwin 2.0 Aloha-AgileX **demo_clean** data only (50 episodes for each of the 50 tasks); the data converter will be released with the training code in Open-HumanWAM ([release plan](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale#release-plan)).

## Training

Not supported yet: this is an **eval-only** submission, so `train.sh` is omitted. Post-training started from a HumanWAM-Goal mid-training checkpoint (iteration 52,000) and the submitted weights are the EMA weights at post-training step 20,000. The post-training code and the initialization checkpoint will be released in Open-HumanWAM ([release plan](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale#release-plan)).

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>

HWGOAL_TASK_CONFIG=demo_clean bash eval.sh RoboTwin open_microwave HumanWAM-Goal-RoboTwin aloha_agilex joint 0 \
  0 1 "$PWD/env" RoboTwin
HWGOAL_TASK_CONFIG=demo_randomized bash eval.sh RoboTwin open_microwave HumanWAM-Goal-RoboTwin aloha_agilex joint 0 \
  0 1 "$PWD/env" RoboTwin
```

With `EVAL_ENV_TYPE=sim` (default) the client is RoboTwin's own `scripts/eval_policy_xpolicylab.py`, whose episode loop (scene setup, observations, action execution, step limit and success check) runs unchanged. The seed mode selects the 100 trials per task:

| `HWGOAL_SEED_MODE` | Trials |
| --- | --- |
| `fixed` (default, reported results) | The task's 100 released cases: each runs one native episode at its released seed with its released instruction (seen for `demo_clean`, unseen for `demo_randomized`), without re-running the expert check. |
| `native` | The evaluator's own selection: expert check over seeds from 100000 until 100 are accepted, instructions generated from the expert episode. |

We report fixed mode because native seed selection is not repeatable: CuRobo's L-BFGS CUDA kernel reads uninitialized shared memory, so the same seed can be accepted in one process and rejected in another, and MotionGen's planning retries stop at a 10 s wall-clock timeout, so acceptance depends on machine load. The released 10,000 cases are the population our expert check accepted (CuRobo's PyTorch L-BFGS path, one fresh process per candidate); fixed mode replays exactly that population. `HWGOAL_FIXED_CASES=start:stop` runs only that slice of a task's 100 released cases (seed order), to split a task across machines or for smoke tests; its result is marked partial.

The adapter only wraps the evaluator's policy-client factory to send the task configuration with each case and to stop the whole run (exit code 2, no result file) on any policy-side error, so a failure can never skip or replace a seed; in fixed mode a released case whose setup fails also stops the run. Results are in RoboTwin's usual result directory (`_result.txt` over the task's 100 trials; per-case native results under `cases/`); HumanWAM-Goal audit logs are in `eval_result/<task>/<task_config>/<time>/`.

`EVAL_ENV_TYPE=debug` runs XPolicyLab's debug client through `deploy.py`. Debug observations correspond to no cached scene, so the current head image serves as the goal and the synthetic `ee_pose` quaternions are normalized (debug mode only). It should end with `[MAIN] eval finished`.

### Goal images

A case is served only when its task configuration, task, seed and exact instruction are in the released cache, and its step-zero RGB, state and head-camera pose equal the cached initial observation; anything else stops the run. In native mode, a seed that another machine's expert check accepts outside the released population therefore stops the run at that case instead of skipping it.

## Model Assets

| Path | Content |
| --- | --- |
| `checkpoints/<ckpt_name>/` | Inference checkpoint: `transformer/`, `vae/`, `text_tokenizer/`, `processor.json`, `sampling.json`, `robot.json`, `export.json` |
| `assets/goal_cache/` | `robotwin_official_goal_cache_v1`: 10,000 head-camera goals with their initial observations, identities and SHA-256 chain |
| `env/runtime/cublaslt-12.8.3.14/` | Pinned `libcublasLt.so.12` preloaded by the policy server (installed by `install.sh`) |

The evaluation profile (joint control, 32 executed steps per query, seed `9000 + query`) is `humanwam_goal/configs/robotwin.yaml` in the installed code.

## Configuration

`deploy.yml` keeps the standard key set. HumanWAM-Goal-specific keys: `checkpoint_path`, `official_goal_cache`, `official_goal_cache_sha256`, `robotwin_profile`, `cublaslt_library`, `output`. Relative paths resolve against this directory. Per-run environment overrides: `HWGOAL_TASK_CONFIG`, `HWGOAL_SEED_MODE`, `HWGOAL_FIXED_CASES`, `HWGOAL_CHECKPOINT`, `HWGOAL_GOAL_CACHE`, `HWGOAL_GOAL_CACHE_SHA256`, `HWGOAL_PROFILE`, `HWGOAL_CUBLASLT`, `HWGOAL_OUTPUT`, `HWGOAL_REPO`, `HWGOAL_ENV` (install only), `ROBOTWIN_ROOT`. `HWGOAL_CHECK_ONLY=1` makes the server and client scripts print the resolved profile and exit; on the server it also checks the pinned cuBLASLt digest.

## Notes

- One environment per policy process (`eval_batch: false`); RoboTwin's multi-worker client is not supported.
- `env_cfg_type` `aloha_agilex` and `arx_x5` give identical wire dimensions; the simulated robot always comes from RoboTwin's task YAMLs (`aloha-agilex`).
- Cached goals were produced by several generation batches of the same initial-only method (9,269 reused from earlier batches whose inputs were re-verified against the current initial observation, 731 newly generated).
