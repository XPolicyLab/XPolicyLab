# HarnessTree

**Contributor:** HarnessPolicy contributors | **Paper:** TBD | **arXiv:** TBD | **Adapter:** HarnessTree (see [design inspiration](ATTRIBUTION.md))

HarnessTree is an eval-only RoboDojo agent for dual-arm ARX-X5 joint control, with one environment per policy process. The evaluator's Detail Description selects a prewritten Python skill program that uses RGB observations, robot state, and an external SAM 3 HTTP service. See [ATTRIBUTION.md](ATTRIBUTION.md) for code provenance and [Notes](#notes) for Agent Track limitations.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/HarnessTree
bash install.sh /path/to/python
```

Use the Python interpreter from your policy Conda environment. The script installs XPolicyLab in editable mode and installs Pillow, NumPy, and OpenCV headless. Use a separate RoboDojo evaluation environment with Isaac Sim and cuRobo for simulation. Keep RoboDojo's `env_cfg/` beside `XPolicyLab/` in the parent workspace.

## Data Processing

This eval-only adapter consumes RoboDojo observations directly. It does not convert training datasets and has no `process_data.sh`.

## Training

This eval-only submission has no `train.sh` or training pipeline. **Training release ETA:** with the paper.

## Evaluation

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, and one environment per policy process (`eval_batch: false`). `ckpt_name` is a run label; it does not select weights.

```bash
cd XPolicyLab/policy/HarnessTree
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline wiring (no Isaac Sim or SAM 3 inference):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

Debug mode sends hold actions to check transport and action shapes, not task success. For simulation, leave `EVAL_ENV_TYPE` unset or set it to `sim`, start the SAM 3 service, and point the policy server at it with `HARNESS_SAM3_URL`. For split-machine setup, see [Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

There is no adapter-owned VLA or policy checkpoint. SAM 3 weights are required for simulation, not for debug mode.

| Asset | Setup |
| --- | --- |
| SAM 3 weights (`checkpoints/sam3/sam3.pt`) | Run `bash download_checkpoint.sh [/path/to/python]`. The script tries ModelScope (`facebook/sam3`), falls back to `hf-mirror`, and verifies SHA-256. |
| SAM 3 service environment | Install PyTorch with CUDA and `RLinf/sam3` at `c1f6b1e83967c912f675d8975d437ccdc41e17d8` (or an equivalent build). |
| RoboDojo simulator assets | Provision through RoboDojo; not shipped with this adapter. |

From `XPolicyLab/policy/HarnessTree`, download the weights and start the service:

```bash
bash download_checkpoint.sh /path/to/python
/path/to/sam3-env/bin/python -u scripts/serve_sam3.py \
  --checkpoint checkpoints/sam3/sam3.pt \
  --host 127.0.0.1 --port 8810
```

In the shell that launches evaluation, run `export HARNESS_SAM3_URL=http://127.0.0.1:8810`. The expected checkpoint SHA-256 is `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. The service has no `/health` route; wait for the `{"serving": "sam3", ...}` startup line and use `/segment` with a non-evaluation RGB image to check readiness. Do not expose the service on a public network.

## Configuration

Extra `deploy.yml` keys (beyond the standard set):

| Key | Role |
| --- | --- |
| `robot` | observation/action profile tag (`robodojo_xpl_rgb`) |
| `primitives` | skill stack tag (`no-vla`) |
| `sam3_url` | default SAM3 base URL (overridden by `HARNESS_SAM3_URL`) |
| `poll_s` | max wait while a program plans before the next bound action |
| `wrap_up_s` | max wait for the episode result record at trial end |
| `result_dir` | policy-side trace directory (default `./results`; override with `HARNESS_RESULT_DIR`) |

Environment variables:

| Variable | Notes |
| --- | --- |
| `HARNESS_SAM3_URL` | Overrides `deploy.yml` `sam3_url` for the perception service |
| `EVAL_ENV_TYPE` | `debug` for offline wiring; unset/`sim` for RoboDojo simulation |
| `DEBUG_OBS_ENCODED` | With debug client: send encoded camera colors across the three cameras |

## Notes

### Images / codecs

- XPolicyLab camera observations arrive as plain RGB arrays from the policy server — `model.py` does not decode.
- Private SAM3 RPC uses lossless RGB PNG / mask PNG (`sam3_http_client.py` / `scripts/serve_sam3.py`). That path is **not** XPolicyLab trajectory `decode_image_bit` / `encode_image_bit`.
- Inherited GrabCut helpers in a few programs use an RGB→BGR conversion around OpenCV GrabCut only (`hang_mugs*`, `imitate_sorting_sequence`). Not a trajectory channel-swap.
- Packed bitmaps in a few number/equation programs use `np.frombuffer` on hex constants (not camera JPEG bits).

### Limitations / Agent Track residuals

**Design ceiling (intentional).** HarnessTree is an instruction→complete skill-program agent. Clearing the residuals below would mean replacing that architecture (general test-time tool orchestration, evaluator-owned stop API, visual identity instead of layout priors). We are not planning those redesigns for this submission; an Agent Track `*` / separate display under the [protocol](https://robodojo-benchmark.com/leaderboard/protocol) is acceptable.

- Instruction-selected **complete** task programs (§§2/4): not test-time orchestration of reusable tools; retained rather than stubbed.
- Private deploy may clear evaluator `success` to trigger the native final check on early stop (§6); not an official evaluator-owned stop API (same early-stop pattern as other merged agent adapters; left as-is).
- Kong layout-as-identity, retained calibrations (tubes/mugs), and related priors: not cleared for verified listing.

### Local seed-0 native evidence

Run `compliant_desens_s0_native_12w_20261125_2324` used seed 0 and `--eval-num native`. The table is from `robodojo.sh summarize` on an isolated staging root (42 of 126 cells).

| Policy | Progress | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: |
| HarnessTree | 33.3% (42/126) | 68.92 | 75.42 |

The operator forced SR and score to 0 for `store_laptop_and_headphones` after the native client hung mid-run on November 26, 2026. These are partial local results, not a verified multi-seed leaderboard listing.

### License / attribution

The vendored colour helpers are licensed under Apache-2.0; see [NOTICE](NOTICE) and the [Apache license](LICENSE-Apache-2.0-colour-host.txt). See [Attribution](ATTRIBUTION.md) for design inspiration. Keep `results/`, `checkpoints/`, caches and credentials out of git (see `.gitignore`).
