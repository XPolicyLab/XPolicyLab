# ActWeave

**Contributor:** ZJU-OmniAI | **Original code:** [ZJU-OmniAI/ActWeave](https://github.com/ZJU-OmniAI/ActWeave/tree/main) (`main`)

Online Agent for RoboDojo `arx_x5 / joint`. Use `eval_batch: false`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). 

## Installation

Requires Linux, Python 3.11.8 or newer in the 3.11 series, Node.js 22.19 or newer, Git, bubblewrap, ffmpeg and curl. Use the official RoboDojo workspace layout, with this checkout at `<RoboDojo>/XPolicyLab` and `env_cfg/` in its parent directory.

Activate the policy Python environment and run from the XPolicyLab checkout:

```bash
export ACTWEAVE_ROOT="$PWD/policy/ActWeave/runtime"
bash policy/ActWeave/install.sh
bash policy/ActWeave/download_checkpoint.sh
```

The installer clones `main` and installs dependencies. Existing checkouts are reused without pulling.

`download_checkpoint.sh` downloads the required demonstration videos.

Install [SAM3 and its weights](https://github.com/facebookresearch/sam3) in a separate CUDA-enabled environment. Activate that environment, then install the service dependencies:

```bash
bash "$ACTWEAVE_ROOT/scripts/install_perception.sh"
```

Prepare GPT 6 Astra and Jev API credentials.

## Data Processing

Not required; `process_data.sh` is unsupported.

## Training

Not required; `train.sh` is unsupported.

## Evaluation

### Configuration

```bash
cp "$ACTWEAVE_ROOT/configs/runtime_template.json" "$ACTWEAVE_ROOT/configs/runtime.local.json"
cp "$ACTWEAVE_ROOT/configs/sandbox_template.json" "$ACTWEAVE_ROOT/configs/sandbox.local.json"
export EVO_CONFIG="$ACTWEAVE_ROOT/configs/runtime.local.json"
export OPENAI_BASE_URL=https://your-astra-endpoint/v1
export OPENAI_API_KEY=your-astra-key
export JEV_API_KEY=your-jev-key
```

- Runtime JSON: set `model`, `jev.model`, `jev.endpoint` (full System One URL), and `policyPython`. Keep `maxRequests: null`.
- Sandbox JSON: set `base`, `python` and NumPy/Pillow mount paths for your environment.
- Relative runtime paths resolve against `ACTWEAVE_ROOT`.

### Run

In the SAM3 environment, start the perception service:

```bash
python "$ACTWEAVE_ROOT/python/perception/server.py" \
  --model /absolute/path/to/sam3.pt --gpu 0 --port 8002 \
  --backend sam3 --runtime-config "$EVO_CONFIG"
```

Keep this service running; its port must match `ports.perception`.

In another shell, export the variables above and run from the XPolicyLab checkout:

```bash
bash policy/ActWeave/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_venv_path> <eval_conda_env>

EVAL_ENV_TYPE=sim bash policy/ActWeave/eval.sh \
  RoboDojo stack_bowls "$EVO_CONFIG" arx_x5 joint 0 0 1 actweave-policy robo-eval
```

`ckpt_name` is the runtime JSON path. Use separate bridge/perception ports and output directories for concurrent instances.
