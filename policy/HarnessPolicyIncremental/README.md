# HarnessPolicyIncremental

**Contributor:** HarnessPolicy contributors | **Paper:** Not applicable | **arXiv:** Not applicable | **Original code:** HarnessPolicy and PyRUA lineage (see [Attribution](ATTRIBUTION.md))

An eval-only scripted Agent for RoboDojo's dual-arm ARX-X5 (`joint`, one environment per policy process). It selects a prewritten Python motion program from the evaluator's instruction and uses RGB, own robot state, and an external SAM3 segmentation service. It vendors PyRUA-Lean colour tooling and adapts PyRUA_RSI task programs; this is not a claim of official Agent Track compliance.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Local seed-0 native evidence

Run id: `compliant_desens_s0_native_12w_20261125_2324` (seed 0, `--eval-num native`).

Official-shaped `robodojo.sh summarize` on an isolated staging root (42/126 cells; one seed only):

| Policy | Progress | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: |
| HarnessPolicyIncremental | 33.3% † (42/126) | 68.92 | 75.42 |

`store_laptop_and_headphones` is recorded as **SR=0 / score=0** by operator decision after the native episode hung mid-run (2026-11-26); it is not a completed evaluator sweep for that task. This is local application evidence, not a verified multi-seed leaderboard listing.

## Installation

This adapter uses the upstream HarnessPolicy `harness-policy` Conda recipe; there is no separate `install.sh` or package installation in the eval launchers. Use the upstream HarnessPolicy checkout at `c3deacddd49ec7088143a23da9f1a287919d14ed` for the recipe in `environment/README.md`, `environment/setup.sh` and `environment/constraints.txt`. Use Python 3.11 and record the actual `environment/lock/` files from the provisioned environment. The existing environment was used for this adapter's debug checks; a fresh full provision has not been tested.

1. In the HarnessPolicy workspace, put RoboDojo at `third_party/RoboDojo` (`c3af1ac6ad7302c0c09d17f4b050fd3a5c68c921`, with its submodules). Under that checkout use `XPolicyLab` at `4e5aee2cdd279267faea5d79ad727fcf0f4a02c8` **plus this adapter**; this XPolicyLab revision differs from RoboDojo's gitlink. Keep RoboDojo's `env_cfg/` beside `XPolicyLab/`, even for debug. Do not depend on an unrelated XPolicyLab checkout on `PYTHONPATH`.
2. From the HarnessPolicy root, follow the upstream instructions to provision the `harness-policy` Conda environment (`bash environment/setup.sh`), or use an existing matching environment. For a reproducible SAM3 deployment, follow **Model Assets** below when choosing `SAM3_ROOT`; do not assume the recipe's default SAM3 source matches the verified install record. The provisioner is not a debug quickstart: its full recipe installs simulator dependencies, and benchmark assets are provisioned separately. The debug run below does not start a simulator, download assets, or call SAM3.
3. Run `conda activate harness-policy` (or activate its full prefix). Both local launchers accept the Conda name or full prefix. The policy launcher imports XPolicyLab from RoboDojo's parent layout; no policy checkpoint is required. In split deployment the policy side also needs the ARX-X5 robot metadata for XPolicyLab's shared action-dimension check.

## Data Processing

Not applicable. This adapter does not read or convert a training dataset and has no `process_data.sh`.

## Training

Not applicable. The adapter has no training or fine-tuning pipeline, adapter weights, or training release ETA. Eval-only status and the omission of the training/data scripts are proposed for maintainer approval, not granted. SAM3 is a third-party pretrained component, not a checkpoint trained for this adapter.

## Evaluation

```bash
cd XPolicyLab/policy/HarnessPolicyIncremental  # from the RoboDojo root
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline wiring only, with the named environment already installed:
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 harness-policy harness-policy
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 harness-policy harness-policy
```

`eval-only` is a run label; no checkpoint is selected by it. Debug validates observations, joint-action shapes, bindings and transport using a hold action, not task success. Simulator deployment requires the external segmentation service, the benchmark assets and the matching `deploy.py` on the evaluator side. The paired loop polls while a program is planning, sends one bound action per observation, and returns an acknowledgment only after the evaluator accepts the action call. `trial_end` delivers the evaluator's verdict to the server. A stock loop without bindings/receipts is not compatible. See the [Deployment Flow](../../README.md#-deployment-flow) for launchers on separate machines. Do not use this private-run integration for official evaluation without maintainer approval of the client interface.

## Model Assets

There is no adapter checkpoint or checkpoint download script. Provision the licensed SAM3 `sam3.pt` separately (the development artifact was recorded as ModelScope `facebook/sam3`). Verify the file, source and license at freeze time. The local artifact inspected for this package has SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`; the hash is an identity, not evidence of upstream release provenance.

Pin HarnessPolicy's `services/servers/serve_sam3.py` from the revision above (SHA-256 `71326435f3d1786eff548d0b48848c55ba6d1b05c02d98789448ecb6de207424`). The existing `harness-policy` environment has `sam3` from `RLinf/sam3` commit `c1f6b1e83967c912f675d8975d437ccdc41e17d8` (package install metadata). To reproduce this dependency with the upstream installer, check out that source revision and set `SAM3_ROOT` to its checkout before provisioning. HarnessPolicy's default source manifest lists a different fork; freeze the actual installed revision and dependency lock, not the default manifest alone. The new adapter has not been tested with live SAM3 in this pass.

```bash
# From the pinned HarnessPolicy root, in the provisioned environment:
sha256sum services/servers/serve_sam3.py "$SAM3_CHECKPOINT_PATH"
python -u services/servers/serve_sam3.py --checkpoint "$SAM3_CHECKPOINT_PATH" --host 127.0.0.1 --port 8810
```

Set `SAM3_CHECKPOINT_PATH` to the local file for the command above; the policy launcher does not read it. Use both text and point support (no `--text-only`); record the actual launch flags, GPU, package lock and checkpoint hash. Set `HARNESS_SAM3_URL=http://127.0.0.1:8810` in the policy server's environment (overrides `deploy.yml`'s `sam3_url`; the default is that loopback address). The service has no HTTP health route: wait for its `{"serving": "sam3", ...}` startup line and, for a real deployment gate, validate `/segment` and `/segment_point` with permitted non-evaluation RGB before starting. It serializes model inference. Benchmark the required concurrency separately; do not expose this unauthenticated service to the public network. SAM3 exchanges lossless RGB PNG and binary mask PNG privately; XPolicyLab camera decoding stays on the policy server's shared RGB path.

## Notes

- Instruction priority is the official `extra_info.detailed_instruction`, frozen at the first observation. When that field delegates a concrete instance goal, the short evaluator instruction supplies the binding; it is also the compatibility fallback if the detailed field is absent. Random variants route by wording, not task ID. Confirm the short-text binding and the step/limit/image-lag control signals with maintainers.
- **Official Agent Track blockers:** instruction-selected complete task scripts; native private-run early final-check hook mutates `success` because the pinned runtime has no evaluator-owned stop request; Kong still uses layout as identity after observing which discard fell, and nominal wall/end poses remain. Conveyor uses a guarded live RGB speed estimate with a prior fallback; tube zero offset and mug clearance remain prior calibrations in part. These paths have not been cleared by the Committee.
- `poll_s` controls each action poll and `wrap_up_s` bounds how long the server waits for the result record at trial end. `result_dir` defaults to `./results` relative to the policy server's working directory; `PYRUALEAN_RESULT_DIR` is the legacy override still accepted. Store traces outside source control and review them for privileged data before release.
- The smoke evidence from this patch is debug/offline only. The core motion paths, including Kong and Exp2 mugs, are retained. Compare on an approved, frozen configuration before claiming task-performance parity. See `NOTICE` and `LICENSE-PyRUA-Lean` for the vendored files; keep `results/`, checkpoints, caches and credentials out of the submission.
