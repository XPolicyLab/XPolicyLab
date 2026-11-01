# Extending The Harness

Every extension is a file in a known place; nothing needs a change in the runtime.
After any of them, `python -m unittest discover -s tests` checks that every configuration
still resolves and that every prompt pack describes exactly the API its kernel exposes.

## A skill

Step by step: [Write a skill](../tutorials/write-a-skill.md).

```text
skills/<name>/
├── skill.toml
└── __init__.py
```

```toml
name = "pointing"
summary = "Open-vocabulary pointing and 3D localization from one or more cameras."
layer = "base"                      # base | reactive | handlers | learned
mixin = "skills.pointing:Pointing"
exports = ["point_to", "localize", "localize_3d"]
needs_any = ["pointer", "vlm"]      # service kinds; or needs = [...] (all), requires = [skills]
order = 30                          # where it sits in the session's method resolution
```

The mixin's methods run inside the kernel's session (the episode core of
`runtime/episode.py`, for a robot `RobotSession` of `runtime/session.py`) and may use
`self.env` (the worker), `self.state`, `self._execute(...)` for bounded motions, and
other skills' methods. A skill runs on robots unless its `embodiments` names other kinds
(`embodiments = ["desktop"]`). Document the exports in each prompt pack inside
`<!-- if: <name> --> ... <!-- end -->` so the prompt drops them with the skill; prompt
text may also depend on what the robot connection reports (`<!-- if: measured_gripper
-->`, `<!-- if: step_limit -->`, see `runtime/prompts.py`). Turn it on in the agent:
`[skills] names = [...]`, or by its layer.

## An MCP tool server

Any server that speaks the Model Context Protocol over streamable HTTP becomes a set of
tools programs call: a manifest of kind `mcp` (`services/mcp-units.toml` is an example,
with its server in `services/servers/serve_mcp_example.py`), the deploy places it like any
service, and the agent names it and turns the skill on:

```toml
[services]
mcp = ["mcp-units"]

[skills]
enable = ["mcp"]
```

Programs then see `mcp_tools()` and `mcp_call(tool, **arguments)` (skills/mcp). Programs
run isolated and cannot start processes, so a server that only speaks stdio runs behind an
HTTP bridge as a service.

An MCP server runs outside the program's sandbox: whatever its tools reach, programs reach
through `mcp_call`. A filesystem or shell server would hand them the files and processes
the isolation hides (the benchmark checkout, task files, keys), so give an agent only
servers whose tools are safe for any program to call. Robot commands still go through
the worker and its shield.

## A service, and a VLA as a learned skill

Step by step: [Add a VLA](../tutorials/add-a-vla.md).

`services/<name>.toml`:

```toml
kind = "policy"                     # policy | world_model | segmenter | pointer | vlm | llm | kinematics
summary = "..."
protocol = "xpolicylab"             # http | openai_chat | xpolicylab
port = 19008
vram_gb = 14.0                      # measured; placement and swapping use it
load_seconds = 180
client = { url = "{url}", description = "what the model is told about it" }
needs_files = ["{weights}/simate/model.safetensors"]

[xpolicylab]                        # an XPolicyLab policy: no server code of our own
policy = "Simate_beta"
ckpt_name = "{weights}/simate"
action_type = "joint"

[[weights]]                         # `harness fetch` sources (ModelScope, hf-mirror)
dir = "simate"
modelscope_dataset = "RoboDojo-Benchmark/RoboDojo"
root = "ckpt/RoboDojo/Simate-beta/RoboDojo-sim-arx_x5-joint-0"
```

Any of XPolicyLab's policies becomes a learned skill this way: `run_policy` speaks its
protocol and hands it the observation it was trained on. A service we serve ourselves
has `command = ["{python}", "-u", "services/servers/<server>.py", "--port", "{port}", ...]`
instead of `[xpolicylab]`. Then place it in a deploy and name it in the agent
(`[services] policies = ["simate", "g05"]`; `start` hands the task to the first unless
the program names another), and give it a description from its own measured scores
([Add a VLA](../tutorials/add-a-vla.md#3-its-description)).

## An ablation

`configs/ablations/<name>.toml`, a few lines on top of the agent:

```toml
description = "without the verifier"
include = ["../default.toml"]
[runtime]
verifier = false
```

The agent names the prompt pack, the skills, the services the skills use, runtime
settings and the budget. Model roles live in `configs/models/` so the agent runs with
any model. `harness config <experiment or agent file> --deploy <machine>` shows what it
resolves to; `harness lock` writes its identity. A change to what the agent does goes
into `configs/default.toml` itself, with a new `AGENT_VERSION` and a changelog entry.

## A machine

`configs/deploy/<machine>.toml`: GPUs, weights, state directory, `[sim]` workers, and
per service `gpu = N | "auto"`, `mode = "resident" | "on_demand" | "external" | "off"`,
`swap = "<group>"` for on-demand services that share a GPU. Examples for four, two and
one GPU and for no GPU are in `configs/deploy/`; a machine's own file there is
git-ignored. `harness doctor --deploy <machine>` checks it.

## A bench or a robot

A bench is a worker that serves the robot API (`profile`, `reset`, `tick`, `state`,
`observe`, `fk`, `ik`, `plan`, outcome) through `harness serve --factory
module:create_environment`; see `benches/robodojo/runtime.py` and
`benches/robotwin/runtime.py`. A robot's description (joint groups, limits, cameras,
calibration, gripper, mount poses) goes to `embodiments/<name>/embodiment.toml`; a
prompt pack goes to `prompts/<bench>/`.

## An evaluator

When someone else runs the policy (XPolicyLab), three pieces keep the runtime
untouched: an observation contract (`core/contract.py`) so development runs see what
the evaluator hands a policy, a connector (`connectors/`) that presents the harness
in the evaluator's interface, and an export (`export/`) that writes the submission.
[Evaluation through XPolicyLab](../concepts/xpolicylab.md) describes the XPolicyLab ones.
