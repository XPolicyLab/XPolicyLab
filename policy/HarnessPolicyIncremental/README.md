# HarnessPolicyIncremental

**Contributor:** HarnessPolicy contributors | **Paper:** TBD | **arXiv:** TBD | **Original code:** HarnessPolicy / PyRUA lineage (see [Attribution](ATTRIBUTION.md))

Eval-only RoboDojo agent for dual-arm ARX-X5 **joint** actions (one environment per policy process). The official Detail Description selects a prewritten Python skill program; online RGB, own robot state, and an external **SAM 3** HTTP service drive perception/motion. This is a research/integration adapter — see **Notes** before treating it as Agent Track–compliant.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/HarnessPolicyIncremental
bash install.sh /path/to/policy-env/bin/python
# or: bash install.sh python   # if that interpreter is already on PATH
```

`install.sh` editable-installs XPolicyLab and installs Pillow / NumPy / opencv-headless into the **policy** interpreter. The **eval** interpreter (Isaac / cuRobo / RoboDojo client) is whatever you pass as `eval.sh` arg 10 — typically the existing RoboDojo or `harness-policy` Conda env. Keep RoboDojo’s `env_cfg/` beside `XPolicyLab/` so `get_robot_action_dim_info` resolves. Simulator assets come from the RoboDojo provision path (not shipped here).

## Data Processing

Unsupported. No `process_data.sh`; the adapter does not convert training datasets.

## Model Assets

There is **no** adapter-owned VLA/policy checkpoint. `ckpt_name` is only a run label. Leaderboard / sim perception needs the SAM 3 weight and a torch+`sam3` service env:

| Asset | Required for | How |
| --- | --- | --- |
| `checkpoints/sam3/sam3.pt` | sim / official eval (not debug hold) | `bash download_checkpoints.sh [/path/to/python]` |
| `sam3` Python package + torch/CUDA | SAM3 HTTP service | install `RLinf/sam3` @ `c1f6b1e83967c912f675d8975d437ccdc41e17d8` (or equivalent) in the **service** env |
| RoboDojo simulator assets | sim | RoboDojo asset provision |

```bash
cd XPolicyLab/policy/HarnessPolicyIncremental
bash download_checkpoints.sh /path/to/policy-env/bin/python
# ModelScope facebook/sam3 → hf-mirror fallback; sha256-checked
# Expected SHA-256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

export HARNESS_SAM3_URL=http://127.0.0.1:8810
python -u scripts/serve_sam3.py \
  --checkpoint checkpoints/sam3/sam3.pt \
  --host 127.0.0.1 --port 8810
```

The service has no `/health` route — wait for the `{"serving": "sam3", ...}` line; gate with `/segment` on non-eval RGB before load. Do not expose it on a public network.

## Training

Unsupported / **eval-only**. No `train.sh`, no adapter weights, no training pipeline in this PR. **Training release ETA:** with the paper (or earlier if maintainers request a training drop). Notify maintainers when claiming eval-only under [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Evaluation

```bash
cd XPolicyLab/policy/HarnessPolicyIncremental

bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>

# Offline wiring (no Isaac / no SAM3 inference):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 \
  /path/to/policy-env /path/to/eval-env
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls eval-only arx_x5 joint 0 0 0 \
  /path/to/policy-env /path/to/eval-env
```

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. `ckpt_name=eval-only` is a label only. Debug uses a hold action to check transport/shapes — not task success. Simulator runs need SAM3 up (`HARNESS_SAM3_URL` overrides `deploy.yml` `sam3_url`), assets, and the paired deploy loop. See [Deployment Flow](../../README.md#-deployment-flow).

### Local seed-0 native evidence

Run id: `compliant_desens_s0_native_12w_20261125_2324` (seed 0, `--eval-num native`). Official-shaped `robodojo.sh summarize` on an isolated staging root (**42/126** cells; one seed only):

| Policy | Progress | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: |
| HarnessPolicyIncremental | 33.3% † (42/126) | 68.92 | 75.42 |

`store_laptop_and_headphones` is **SR=0 / score=0** by operator decision after the native client hung mid-run (2026-11-26). Local application evidence only — not a verified multi-seed listing.

## Configuration

Extra `deploy.yml` keys (beyond the standard set):

| Key | Role |
| --- | --- |
| `robot` | observation/action profile tag (`robodojo_xpl_rgb`) |
| `primitives` | skill stack tag (`no-vla`) |
| `sam3_url` | default SAM3 base URL (overridden by `HARNESS_SAM3_URL`) |
| `poll_s` | max wait while a program plans before the next bound action |
| `wrap_up_s` | max wait for the episode result record at trial end |
| `result_dir` | policy-side trace directory (default `./results`; legacy `PYRUALEAN_RESULT_DIR`) |

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

- Instruction-selected **complete** task programs (§§2/4): not test-time orchestration of reusable tools; retained rather than stubbed. Committee disposition required for official Agent Track claims.
- Private deploy may clear evaluator `success` to trigger the native final check on early stop (§6); not an official evaluator-owned stop API.
- Kong layout-as-identity, retained calibrations (tubes/mugs), and related priors: see packaging notes / residuals; not cleared for verified listing.
- One-seed local summarize above is application evidence only.

### License / attribution

Vendored colour tooling: `NOTICE`, `LICENSE-PyRUA-Lean`, `ATTRIBUTION.md`. Keep `results/`, `checkpoints/`, caches and credentials out of git (see `.gitignore`).
