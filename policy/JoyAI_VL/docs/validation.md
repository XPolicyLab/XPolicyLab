# Adapter validation

These checks cover the XPolicyLab interface, image transport and process startup.
They use synthetic observations and a local model API fixture. They do not measure
JoyAI-VL task success rates.

## CPU checks

Activate the installed policy Python environment, then run from `policy/JoyAI_VL`:

```bash
python3 tools/verify_release.py
python3 -B -m unittest discover -s tests -v
```

The tests cover official legacy and marked JPEG input, RGB camera and tool PNGs,
SAM3 mask pixels, generic tool context, API request parameters, action polling,
final-verdict forwarding, fallback network gateways and checkout import order.
The internal PNG decoder rejects official JPEG buffers: the official policy
server decodes those through XPolicyLab's shared helper before calling the model.

## Full process debug checks

Both observation modes use the official `eval.sh`, policy server, WebSocket
client and `utils/debug_env_client.py`, with the actual JoyAI-VL adapter, Docker
agent and SAM3 service. The only model substitute is `tests/mock_model_api.py`,
which reads the image requests and returns fixed generic observation/wait calls.
It is a test fixture, outside the agent's runtime mounts and tool registry.

Start SAM3 as described in the installation guide. In the policy directory, start
the API fixture in another terminal:

```bash
python3 tests/mock_model_api.py --port 19400
```

Configure a local `config.env` using `tools/configure.py`: set `MODEL=interface-check`,
use the fixture's host address reachable from the Docker egress container as the
API origin, `/v1` as the base path, and a private file containing a dummy test key.
For this check set `HEXASHELL_MAX_TURNS=2` and `HEXASHELL_STREAM=0` in that local
configuration. Keep real model credentials in the normal evaluation configuration.
The API fixture listens on all interfaces for the container to reach it.

With the policy and SAM3 dependencies installed, run:

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 bash eval.sh \
  RoboDojo stack_bowls interface-check arx_x5 joint 0 0 0 \
  /path/to/policy-env/bin/python <client-conda-env>

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh \
  RoboDojo stack_bowls interface-check arx_x5 joint 0 0 0 \
  /path/to/policy-env/bin/python <client-conda-env>
```

`stack_bowls` selects the standard debug command; the official debug client
supplies artificial images and the instruction `language instruction`. Each mode
runs its default ten episodes. The fixture exercises SAM3, tool pictures and
joint-action forwarding. The two-turn test cap exercises the agent's done path
and the official final-verdict callback.

## Recorded environment

- XPolicyLab base: `30b2d4a2e7c421c8f0d853755ec1f8393ec85ec4`.
- Policy/client Python: 3.11; WebSockets: 17.4.
- Local debug container: Python 3.11.2, NumPy 1.24.2, Pillow 11.3.0.
- Hardware: NVIDIA GeForce RTX 5090; actual cuRobo planner and SAM3 service.
- The local container was assembled from the available reference runtime and
  Pillow. The public Dockerfile separately pins NumPy 1.26.0 and Pillow 11.3.0;
  rebuilding that recipe is not claimed by this debug check.

## Results — 2026-11-19

| Check | Result | Evidence |
| --- | --- | --- |
| CPU suite | 22 tests passed | [CPU log](validation/cpu.log) |
| Plain RGB debug loop | 10/10 episodes completed, exit code 0 | [Process log](validation/plain.log) |
| Encoded-image debug loop | 10/10 episodes completed, exit code 0 | [Process log](validation/encoded.log) |
| Model API fixture | 40 requests, including camera and tool pictures | [API log](validation/model-api.log) |
| SAM3 service | 40 segmentation requests returned HTTP 200 and masks | [Service log](validation/sam3.log) |

Both process logs reach `[MAIN] eval finished` without tracebacks. Each episode
executes six action steps, ends through the agent's done path and records the
official client's final verdict. All 20 agent processes exit with code 0.
[Machine-readable results](validation/results.json) list the individual episodes.
Local filesystem roots are replaced with placeholders and ANSI colour codes are
removed from the published logs.

The debug client uses synthetic observations and reports `success_official=false`;
this is a transport/startup check, with no simulator task score. Simulator
task-score evaluation for this adapter revision has not been run.
