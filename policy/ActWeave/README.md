# ActWeave

**Contributor:** ZJU-OmniAI | **Original code:** [ZJU-OmniAI/ActWeave](https://github.com/ZJU-OmniAI/ActWeave)

Online Agent adapter for RoboDojo using `arx_x5 / joint` and `eval_batch: false`.

Shared argument and deployment conventions are documented in the [XPolicyLab README](../../README.md).

## Installation

Requires Linux, Python 3.11.8 or newer in the 3.11 series, Node.js 22.19 or newer, Git, and bubblewrap. Place this checkout at `<RoboDojo>/XPolicyLab`, with `env_cfg/` in its parent directory.

Activate the policy Python environment and run from the XPolicyLab checkout:

```bash
export ACTWEAVE_ROOT="$PWD/policy/ActWeave/runtime"
export ACTWEAVE_REVISION="<full ActWeave release commit ID>"
bash policy/ActWeave/install.sh
```

Set `ACTWEAVE_REVISION` to the full release commit ID specified in the evaluation submission. The installer checks out this revision; existing checkouts must match it. Evaluation also requires a clean runtime working tree.

Install [SAM3 and its weights](https://github.com/facebookresearch/sam3) in a separate CUDA-enabled environment, then install the perception service dependencies:

```bash
bash "$ACTWEAVE_ROOT/scripts/install_perception.sh"
```

Prepare Astra and Jev API credentials. This adapter uses external model services and requires no ActWeave checkpoint download.

## Data Processing

No data conversion is required.

## Training

This adapter evaluates deployed models through their APIs.

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

- Runtime JSON: set `model`, `jev.model`, `jev.endpoint` (the full System One URL), and `policyPython`.
- Sandbox JSON: set `base` to the base Python installation, `python` to its interpreter, and `mounts` to the NumPy, SciPy, OpenCV, and Pillow packages and their shared-library directories.
- Relative runtime paths resolve against `ACTWEAVE_ROOT`.

### Run

Start the perception service in the SAM3 environment:

```bash
python "$ACTWEAVE_ROOT/python/perception/server.py" \
  --model /absolute/path/to/sam3.pt --gpu 0 --port 8002 \
  --backend sam3 --runtime-config "$EVO_CONFIG"
```

Keep the perception service running; its port must match `ports.perception`.

In another terminal, export the environment variables above and run from the XPolicyLab checkout:

```bash
bash policy/ActWeave/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_venv_path> <eval_conda_env>

EVAL_ENV_TYPE=sim bash policy/ActWeave/eval.sh \
  RoboDojo stack_bowls "$EVO_CONFIG" arx_x5 joint 0 0 1 actweave-policy robo-eval
```

`ckpt_name` is the runtime JSON path. Each policy server starts its own bridge on an OS-assigned port; concurrent instances can share the same SAM3 service. The RoboDojo client must provide the official `extra_info.detailed_instruction`.
