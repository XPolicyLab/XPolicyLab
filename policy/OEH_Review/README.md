# OEH_Review

**Contributor:** SourceMind Intelligence (Open Embodied Harness) | **Paper:** OEH Study: in-context experience for embodied agents (technical report in preparation) | **arXiv:** pending | **Original code:** https://github.com/SourceMind-Intelligence/OpenEmbodied-Harness (the runtime this adapter runs is vendored under `runtime/` under the Apache-2.0 license in `LICENSE`; `runtime/MANIFEST.json` names its source commit and every file's sha256)

Evaluation-only agent entry: the released OpenWAM-Alpha RoboDojo policy (the official `policy/OpenWAM` adapter, unchanged, weights unchanged) proposes every 32-step EE chunk, and an LLM (`gpt-6-astra`, reasoning effort `xhigh`, OpenAI Responses API) reviews each chunk from the three camera views and the end-effector state, answering with one decision: accept the chunk, a bounded correction, hold, or return home. The Open Embodied Harness runtime (`runtime/oeh_policy`, its Rust kernel under `runtime/kernel`) admits every decision against the workspace bounds and the native step and wall budgets, executes it through XPolicyLab's official OpenWAM episode loop, and records observations, decisions and receipts per episode. `ckpt_name` selects the method configuration: `oeh_review_astra_xhigh` is the entry; `oeh_fast` runs the same OpenWAM proposals through the same gateway without an LLM (the study's direct reference, for wiring checks). There is no training and no checkpoint of its own.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Two Python environments, as for `policy/OpenWAM`: the policy server runs in the OpenWAM policy environment (install it per [policy/OpenWAM/README.md](../OpenWAM/README.md): XPolicyLab, the vendored OpenWAM, torch; it also needs `websockets`), the environment client runs in the RoboDojo environment. Nothing is pip-installed for this adapter: both use the vendored runtime through `PYTHONPATH` (numpy, Pillow and pyyaml, which both environments have). The OEH kernel is built from source with cargo ([rustup.rs](https://rustup.rs); the TLS crates need a C compiler):

```bash
cd XPolicyLab/policy/OEH_Review
bash install.sh --policy-python /path/to/openwam-env/bin/python [--eval-python /path/to/robodojo-env/bin/python]
bash download_checkpoint.sh /path/to/python-with-huggingface_hub   # OpenWAM-Alpha-Sim-RoboDojo, pinned revision
```

`install.sh` verifies `runtime/` against `runtime/MANIFEST.json`, checks the imports of both environments, and builds `runtime/kernel/target/release/oeh` (`--skip-kernel` to build it yourself; `--cargo` to name the cargo binary). The checkpoint goes to `policy/OpenWAM/checkpoints/OpenWAM-Alpha-Sim-RoboDojo/`, where `policy/OpenWAM` resolves it as well (`OPENWAM_CKPT_ROOT` overrides the root; an existing copy elsewhere is used through `OEH_REVIEW_OPENWAM_CKPT_DIR` at evaluation time).

## Data Processing

Unsupported. Eval-only adapter: `process_data.sh` is omitted, no dataset is converted.

## Training

Unsupported. `train.sh` is omitted. The policy weights are OpenWAM-Alpha's released checkpoint; the review prompts and limits are fixed files under `runtime/oeh_policy/teacher/` and `configs/`, hashed into the frozen configuration.

## Evaluation

```bash
cd XPolicyLab/policy/OEH_Review
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# The leaderboard entry
export OEH_REVIEW_OPENWAM_CKPT_DIR=/path/to/OpenWAM-Alpha-Sim-RoboDojo   # only when not under policy/OpenWAM/checkpoints
bash eval.sh RoboDojo classify_objects oeh_review_astra_xhigh arx_x5 ee 0 0 0 /path/to/openwam-env /path/to/robodojo-env

# Wiring check without a simulator or an LLM (OpenWAM through the gateway)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo classify_objects oeh_fast arx_x5 ee 0 0 0 /path/to/openwam-env /path/to/robodojo-env
```

`bench_name` must be `RoboDojo`, `env_cfg_type` `arx_x5`, `action_type` `ee` (OpenWAM's official EE contract; the policy server refuses anything else). `policy_env` and `eval_env` are conda env names or environment directories with `bin/python`. `ckpt_name` is a method configuration from `configs/` (`oeh_review_astra_xhigh`, `oeh_fast`); `task_name` must be a RoboDojo task (`configs/robodojo-step-limits.json` lists them with their native step limits). Batch evaluation is not supported (`eval_batch: false`, one environment per policy server). The review configuration needs the API key file (Configuration).

Before the server starts, `setup_eval_policy_server.sh` freezes the configuration for the task into `state/<ckpt_name>/<task_name>.frozen.json` (`tools/freeze_config.py`): the checkpoint's files, the normalizer, the OpenWAM source, the kernel binary and the vendored runtime are hashed, and the server re-checks them at start and refuses a mismatch. The first freeze hashes the 25 GB checkpoint once; later freezes reuse `state/digest-cache.json`. `EVAL_ENV_TYPE=debug` runs XPolicyLab's offline wiring check against the real policy server (OpenWAM loads its weights; the client sends zero observations); re-run with `DEBUG_OBS_ENCODED=1` for the encoded-color path.

## Model Assets

| Asset | Where | Identity |
| --- | --- | --- |
| OpenWAM-Alpha-Sim-RoboDojo checkpoint | `policy/OpenWAM/checkpoints/OpenWAM-Alpha-Sim-RoboDojo/` (`download_checkpoint.sh`) | `configs/openwam-robodojo-pins.json`: HF revision, every file's size and sha256 |
| OEH runtime and kernel sources | `runtime/` | `runtime/MANIFEST.json` (source commit, sha256 per file); `tools/verify_runtime.py` |
| Review prompts and task notes | `runtime/oeh_policy/teacher/` | hashed into every frozen review configuration |

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `request_timeout_s` | XPolicyLab's per-call transport bound (900 s). The OEH runtime bounds every call itself: a review call ends within `review.call_seconds` + 15 s, the first OpenWAM call adds its warm-up. |

Method configurations (`configs/<ckpt_name>.template.json`, frozen per task by `tools/freeze_config.py`):

| `ckpt_name` | Method |
| --- | --- |
| `oeh_review_astra_xhigh` | OpenWAM-Alpha reviewed by `gpt-6-astra` / `xhigh` through the OpenAI Responses API (the entry). `review.call_seconds` 90, `decision_seconds` 60, `wall_seconds` 21600 per episode, correction limits as in the file. |
| `oeh_fast` | OpenWAM-Alpha through the OEH gateway, no LLM. |

Environment (names only; values are never printed or recorded):

| Variable | Role |
| --- | --- |
| API key file `state/openai-api-key.env` | Read by the review teacher at run time (`NAME=value` or a bare key; `chmod 600`). The frozen configuration names the file, never the key. |
| `OEH_REVIEW_OPENWAM_CKPT_DIR` | The OpenWAM checkpoint directory when it is not under `policy/OpenWAM/checkpoints/`. |
| `OEH_REVIEW_RUNS_ROOT` | Where the per-episode evidence is written (default `runs/`). |
| `OEH_EVAL_OBSERVATION_RETENTION` | Which observation frames the evidence keeps (default: arrays, frames the model read, first and last observation, every 5th head frame). |
| `OPENWAM_ROOT` | The OpenWAM source root (default: the vendored copy under `policy/OpenWAM/OpenWAM`). |

## Notes

- One policy server serves one task; XPolicyLab's clients may run several episodes against it. Each episode starts a fresh OEH runtime (kernel, evidence, model conversation) on the same loaded OpenWAM backend; nothing is carried from one episode to the next.
- Per episode, `runs/<run>/ep<N>/` holds the OEH gateway's evidence (every observation's identity, the frames kept, every model call, every admitted native step with its receipt) and `runs/policy-runtime/<attempt>/<mode>/` the runtime's own log (decisions, kernel events, the review conversation's packets).
- The gateway never calls `is_episode_end` itself and submits exactly the rows of the official loop; RoboDojo's own step limit and scorer decide the episode. A review decision that arrives late is replaced by a one-step hold, so the official loop's call bound is always met.
- Images: observations reach the policy server as XPolicyLab sends them (decoded on the server by XPolicyLab's helpers); the review teacher receives downscaled PNG copies of the decoded RGB views. The adapter decodes no image bits and swaps no channels. The `Image.open` and `np.frombuffer` hits under `runtime/` are in modules of other benchmarks' adapters and in the evidence writer reading back its own PNGs, not on XPolicyLab data.
- Simulation requires a RoboDojo installation with its assets; the LLM requires an OpenAI account serving `gpt-6-astra` on the Responses API.
