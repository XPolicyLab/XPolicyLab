# EmbodiedEvoRepro

**Contributor:** [modaxiansheng](https://github.com/modaxiansheng) | **Paper:** EmbodiedEvo; this adapter makes no new paper claim | **arXiv:** Not supplied | **Original code:** [Einsia/EmbodiedEvo](https://github.com/Einsia/EmbodiedEvo/tree/470278c478bdb5e4b6f17dde1ea0393761f230f0)

This candidate provides an original thin XPolicyLab boundary around EmbodiedEvo's externally installed native coding-agent adapter. It supports **RoboDojo, `arx_x5`, `ee`, `num_envs=1`, `eval_batch=false`**, with only frozen Skills/Lessons carried into fresh Test workers. No upstream implementation is vendored here. This is an attributed reproduction candidate, not an organizer-verified score or a claim to be the original EmbodiedEvo submission.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux and Python 3.11 or later. The adapter accepts a virtualenv directory or a conda environment name for both Python-environment arguments; it does not assume conda exists. XPolicyLab must be inside the parent RoboDojo workspace for simulation. Keep the upstream checkout outside `policy/EmbodiedEvoRepro/`.

```bash
cd XPolicyLab/policy/EmbodiedEvoRepro
bash install.sh <new-venv-directory> <new-upstream-checkout> [upstream-commit]

# Example: install dependencies only; no GPU, model call or experiment starts.
bash install.sh "$HOME/.venvs/embodiedevo-repro" "$HOME/src/EmbodiedEvoRepro-upstream" \
  470278c478bdb5e4b6f17dde1ea0393761f230f0
export EMBODIEDEVO_REPO="$HOME/src/EmbodiedEvoRepro-upstream"
```

The upstream revision is fixed to `470278c478bdb5e4b6f17dde1ea0393761f230f0`. Real mode checks the Git revision, tracked-file cleanliness and imported package location; a different checkout is rejected. `install.sh` preserves existing environment and checkout directories by refusing to overwrite them.

For CPU interface checks alone:

```bash
bash install.sh --interface-only "$HOME/.venvs/embodiedevo-interface"
```

Real evaluation additionally needs the upstream [runtime setup](https://github.com/Einsia/EmbodiedEvo/blob/470278c478bdb5e4b6f17dde1ea0393761f230f0/docs/INSTALL.md), provisioned RoboDojo assets, a compatible simulator environment, an available upstream agent image, an actual frozen Playground source, and an organizer-usable model connection. Install the pinned EmbodiedEvo package in **both** the policy and simulator Python environments: the environment-side deployment loop imports its hooks. No authentication is installed or copied by this adapter. Keep credentials outside source and knowledge bundles.

The pinned public worker uses **Docker** for agent isolation. Historical private native-runtime modifications and the desktop-dependent Pro forwarding arrangement are not bundled or automatically reproduced by this candidate. A frozen configuration specifying another isolation backend is rejected. A separate reviewed implementation is required before claiming native-runtime support.

## Data Processing

No demonstration-data converter or LeRobot training dataset is consumed by this adapter. Its evaluation asset is a completed upstream Playground run containing `protocol.json`, `inputs/`, and `frozen_harness/`.

`process_data.sh` is omitted as part of the requested **eval-only** treatment; the omission is not represented as already approved by maintainers. The PR must declare this scope. There is no separate gradient-training dataset or unpublished trainer for this adapter, so a future trainer release date is not applicable; any required exception remains for maintainer review.

## Training

This adapter does not train neural weights. EmbodiedEvo's Playground produces Skills/Lessons which are frozen before Test; it is outside this evaluation entry point. `train.sh` is omitted under the same requested eval-only treatment. Do not substitute Test outcomes into frozen knowledge, alter the prompt/budget during evaluation, or selectively drop failed scenes.

## Evaluation

The positional interface matches XPolicyLab, with environment-directory support for arguments 9 and 10:

```bash
bash eval.sh <bench_name> <task_name> <frozen-source-path-or-ckpt-name> \
  <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
  <policy-python-env> <simulator-python-env>
```

A runnable CPU wiring example, after the interface-only installation and the parent workspace's `arx_x5` robot configuration are available:

```bash
EVAL_ENV_TYPE=debug EMBODIEDEVO_INTERFACE_PROBE=1 \
  bash eval.sh RoboDojo stack_bowls interface-probe arx_x5 ee 0 0 0 \
  "$HOME/.venvs/embodiedevo-interface" "$HOME/.venvs/embodiedevo-interface"

EVAL_ENV_TYPE=debug EMBODIEDEVO_INTERFACE_PROBE=1 DEBUG_OBS_ENCODED=1 \
  bash eval.sh RoboDojo stack_bowls interface-probe arx_x5 ee 0 0 0 \
  "$HOME/.venvs/embodiedevo-interface" "$HOME/.venvs/embodiedevo-interface"
```

The interface probe holds measured EE poses and verifies transport/action shapes. It calls no LLM, starts no simulator, produces no task success rate, and cannot run in `sim` mode. Encoded observations are decoded only by the shared XPolicyLab server; the adapter receives HWC RGB arrays without channel conversion. The debug example is not evidence of native RoboDojo compatibility.

A real evaluation command template, **only after all required assets/runtime are prepared**, is:

```bash
export EVAL_ENV_TYPE=sim
unset EMBODIEDEVO_INTERFACE_PROBE EMBODIEDEVO_AGENT_RECOVERY
export EMBODIEDEVO_REPO=/absolute/pinned/EmbodiedEvo
export EMBODIEDEVO_SOURCE_RUN=/absolute/completed-frozen-playground
export EMBODIEDEVO_RUN_DIR=/absolute/outputs/YYYYMMDDHHMM_stack_bowls_native
export EMBODIEDEVO_EPISODES=<exact-native-evaluator-episode-count>
bash eval.sh RoboDojo stack_bowls "$EMBODIEDEVO_SOURCE_RUN" arx_x5 ee 1 2 2 \
  /absolute/policy-venv /absolute/simulator-venv
```

Replace the date and angle-bracket value with actual values; the output directory must be new. Both GPU arguments use the same chosen physical card in this example. `EMBODIEDEVO_EPISODES` is a declared count and a maximum-admission guard, **not an override of the official evaluator's task coverage**. It must match the selected native evaluation configuration. The native evaluator controls scene initialization, action limits, reward/success checks, seeds, official reports and videos. This adapter does not patch its scoring or seed plan. The policy polls while the coding agent thinks without advancing physics.

Before calling upstream `integrations.robodojo_policy.run.pin`, the wrapper requires the protocol's nonempty `inputs` hash map to match every tracked frozen input. The new run directory must be outside the source tree. After pinning, it independently checks all copied input hashes and the change receipt, allowing only the four declared native-budget wording substitutions in `test.md` and `primitives/primitives.md`; scripts and submissions remain unchanged. Upstream validates and preserves frozen Skills/Lessons. Test exposes no agent-requested simulator reset; `reset()` only clears policy state between evaluator-owned trials. Each native trial gets a fresh upstream worker. A worker failure is propagated and is not automatically replaced with a new model attempt.

Keep the native evaluator's reports/videos from its own output directory together with `candidate-adapter.json`, `adapter.json`, trial results, action logs and worker receipts under `EMBODIEDEVO_RUN_DIR`. This external XPolicyLab launch does not automatically relocate the evaluator's output into `run/official/`; that directory is created by upstream pinning but can remain empty. Verify requested/completed episode counts, native success, worker completion, frozen hashes and cleanup before reporting any rate. Process exit zero is insufficient. The launch script always waits for its server, propagates server-cleanup failures when the client succeeded, and preserves a nonzero client status when both fail. Preserve interrupted attempts and classify any later reruns separately.

## Model Assets

No frozen knowledge bundle or model weights are included in this candidate. A completed, shareable upstream Playground bundle is still required. Frozen source is resolved with XPolicyLab's shared `resolve_checkpoint_root`: `EMBODIEDEVO_SOURCE_RUN` takes explicit precedence; otherwise `ckpt_name` can be an absolute source-run path or use the shared checkpoint naming layout. It is a knowledge asset, not a neural checkpoint.

A downloader is provided for a **real published** bundle with a known SHA256:

```bash
python download_assets.py --url <public-https-frozen-bundle.tar.gz> \
  --sha256 <published-64-hex-sha256> --output <new-frozen-source-directory>
```

There is intentionally no invented release URL, checksum or example score. The archive must contain `protocol.json`, `inputs/`, `frozen_harness/` at its root. It may not contain authentication files, symlinks, paths outside those roots, or cross-scene Test records. The downloader checks the archive checksum before extraction; upstream performs protocol and frozen-content validation before model startup. Do not make an asset package public before checking its contents and upstream redistribution terms. Runtime configuration paths must be deployable at the destination; this adapter does not silently rewrite model settings or frozen protocol files.

## Configuration

The standard `deploy.yml` keys remain present. Additional keys are `num_envs: 1`, `phase: test`, `run_dir` (normally set by `EMBODIEDEVO_RUN_DIR`), and `episodes` (normally set by `EMBODIEDEVO_EPISODES`). Real mode rejects non-Test phases, diagnostic bypasses, batch mode, different robots/actions, missing frozen source/provision files, changed upstream code, and enabled automatic agent recovery.

`EMBODIEDEVO_DEBUG_EPISODES` optionally sets the CPU probe episode count (default 1). `DEBUG_OBS_ENCODED=1` exercises the shared server decoding path. Neither variable supplies a benchmark score.

## Notes

CPU checks from the XPolicyLab root:

```bash
for script in policy/EmbodiedEvoRepro/*.sh; do bash -n "$script" || exit; done
python -m py_compile policy/EmbodiedEvoRepro/*.py
python -m unittest discover -s policy/EmbodiedEvoRepro/tests -v
git diff --check
```

The focused tests cover refusal of unsupported modes, decoded RGB handling, action shapes, reset restrictions, worker-error propagation and safe asset extraction. They do not validate LLM calls, Docker isolation, Isaac Sim rendering, task success or all-scene completion. Actual raw/encoded WS probe results and any native simulation evidence must be reported separately in the PR. No official evaluation or leaderboard submission is claimed by these tests.
