# JoyAI-VL

**Contributor:** Joy Future Academy · Kanghui Tian | **Paper / arXiv:** — | **Original code:** [JoyAI-VL evaluation release](https://github.com/tiankanghui/JoyVL-Robodojo-Eval), based on [HexaShell Preview](https://github.com/HexaFuture/HexaShell/tree/v1.0)

This is Joy Future Academy's **JoyAI-VL API-driven Agent adapter**.
It uses RGB observations, a shared robot API, SAM3 perception, and joint control.
Every task uses the same system prompt and tool registry; episode context is assembled
from the evaluator's instruction and current robot observations. The retained `roboshell/`,
`vendor/`, `library/`, and `agent/` directories contain the runtime and upstream attributions.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use XPolicyLab inside a working RoboDojo workspace with Isaac Sim 5.1 and cuRobo. The policy server uses that Isaac Python environment. Docker runs the agent and its API proxy; a separate SAM3 service supplies segmentation masks. See [installation details](docs/installation.md) for dependencies and service setup.

Request an API key as described under [Model API access](#model-api-access). From `RoboDojo/XPolicyLab/policy/JoyAI_VL`, configure the endpoint and an API-key file outside the checkout, then build the agent:

```bash
python3 tools/configure.py \
  --robodojo-repo /path/to/RoboDojo \
  --isaac-python /path/to/isaac-env/bin/python \
  --api-base https://modelservice.jdcloud.com/v1 \
  --model joyai_vl_testing \
  --key-file "$HOME/.config/joyai-vl/api-key" \
  --sam3-url http://127.0.0.1:18300
bash install.sh
```

Replace the example paths with your deployment paths. The key file must already exist; keep it private with mode `0600`. The setup writes a private `config.env` without copying key contents. Start the shared segmentation service in another terminal:

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/sam3-env/bin/python services/sam3_server.py \
  --checkpoint /path/to/sam3.pt --port 18300 --cache-images 24
```

The model endpoint supports image inputs, tool calls, reasoning history, and the configured chat-template fields. SAM3 weights are obtained separately.

## Data Processing

This eval-only adapter consumes RoboDojo runtime observations directly; it has no data-conversion stage (`process_data.sh`).

## Training

This eval-only adapter provides deployment code for the JoyAI-VL API. Training and weight release schedule: TBD.

## Evaluation

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, one environment per policy server (`eval_batch: false`).

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <isaac_python_or_env_dir> <eval_env_dir_or_python>

# Example, after configuration and service startup:
bash eval.sh RoboDojo stack_bowls JoyAI-VL arx_x5 joint 2 \
  0 0 /path/to/isaac-env/bin/python /path/to/isaac-env
```

Position 9 supplies the Isaac Python fallback when `ISAAC_PYTHON` is not configured. Position 10 is the benchmark's evaluation environment. `ckpt_name` labels the results; `MODEL` in `config.env` selects the API model. Any evaluator-provided seed can replace `2`. Layouts, hidden splits, episode counts, and scoring remain under RoboDojo's control. See [evaluation details](docs/evaluation.md).

Version history and validation status are recorded in [development history](docs/development.md).

## Model Assets

### Model API access

Request an API key by emailing **either** [qizhangyang.1@jd.com](mailto:qizhangyang.1@jd.com)
or [tiankanghui.1@jd.com](mailto:tiankanghui.1@jd.com).

| Parameter | Value |
| --- | --- |
| Base URL for `tools/configure.py` | `https://modelservice.jdcloud.com/v1` |
| Chat Completions URL | `https://modelservice.jdcloud.com/v1/chat/completions` |
| Serving model ID | `joyai_vl_testing` |
| Authorization | `Bearer {{API_KEY}}` |
| Content type | `application/json` |
| Request tracing | `Trace-Id: {{TRACE_ID}}` (a fresh UUID for each HTTP request) |
| Streaming | `false` |

Minimal connectivity example; replace the two placeholders before running:

```bash
curl -X POST "https://modelservice.jdcloud.com/v1/chat/completions" \
  -H "Authorization: Bearer {{API_KEY}}" \
  -H "Content-Type: application/json" \
  -H "Trace-Id: {{TRACE_ID}}" \
  -d '{
    "model": "joyai_vl_testing",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

For evaluation, the egress proxy reads the external key file and automatically adds
the Bearer token and a new `Trace-Id` to each model request, including retries.
Trace IDs are recorded in the proxy log for request lookup. Evaluation requests use
the inference settings in `configs/model_api.example.json`.

### Additional assets

SAM3 checkpoints and RoboDojo assets are obtained under their respective access and
license terms. Runtime identities and hashes are in [runtime references](metadata/runtime_reference.json).

## Configuration

`policy_name: JoyAI_VL` matches the import directory; the display name is **JoyAI-VL**.

| Setting | Purpose / default |
| --- | --- |
| `command_budget` | `0`: native episode step limit applies |
| `agent_wall_s` | `5400`: agent wall-clock budget in seconds |
| `request_timeout_s` | `120`: bridge RPC timeout, separate from model HTTP requests |
| `config.env` | Private machine paths, endpoint, external key-file path and service URL |
| `configs/model_api.example.json` | API settings: 65,536 tokens, temperature 0, non-streaming, maximum reasoning effort, retained reasoning history, 100 turns, 900-second HTTP timeout |

The egress proxy reads the external API key. Machine-specific settings stay in the local `config.env`.

## Notes

[Agent documentation](docs/agent_submission.md) describes context assembly and shared tools.
[Integration](docs/integration.md) covers the observation/action interface and internal
image transport. [Reproducibility](docs/reproducibility.md) records source hashes and
runtime dependencies. Licenses and attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

CPU checks from this directory, using the policy Python environment:

```bash
python3 tools/verify_release.py
python3 -m unittest discover -s tests -v
```

The suite uses synthetic responses and mocked endpoints to check source integrity,
configuration, API requests, RGB observations and the evaluation entry points.
The full process debug checks and their logs are described in [validation](docs/validation.md).
