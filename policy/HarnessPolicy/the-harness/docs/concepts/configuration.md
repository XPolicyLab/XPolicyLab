# Configuration and identity

There is one agent. What a run evaluates is that agent (or one of its ablations) with
a model, on a bench, resolved against a machine's **deploy**. The files layer the way
other agent frameworks layer theirs (SWE-agent's default config, Codex's and Claude
Code's settings files): later layers win.

| Layer | Where | What it sets |
|---|---|---|
| code defaults | the code | the value of every key a file leaves out |
| the agent | `configs/default.toml` | prompt pack, skills, the services they use, runtime settings, budget, the default model |
| a model | `configs/models/<name>.toml` | model roles: main (writes programs), vlm, pointer, verifier |
| an experiment | `configs/experiments/<name>.toml` | the bench, tasks, seeds, layouts, budgets, worker options, overrides |
| the command line | `--model NAME`, `--agent FILE`, `--set key=value`, `--tasks`, `--seeds` | anything above, for one invocation |

```text
configs/
├── default.toml   the agent
├── ablations/     the paper's ablations: each includes default.toml and switches parts off
├── models/        model roles: hosted models or services, plus the runtime settings a model needs
├── experiments/   the agent with a model on a bench: tasks, seeds, layouts, budgets, overrides
├── deploy/        per machine: GPUs, weights, where each service runs
├── workers/       options for starting a worker by hand (harness serve --options ...)
└── scripts/       scripted model replies for pipeline checks
```

Where services run is the deploy's business, never the agent's; keys live in
environment variables, never in these files.

## The agent

```toml
--8<-- "configs/default.toml"
```

The agent's code version is `AGENT_VERSION` in `src/harness_policy/__init__.py`. Every key is in the
[configuration reference](../reference/configuration.md).

## Ablations

An ablation includes the agent and changes only what the paper's ablation changes:

```toml
--8<-- "configs/ablations/no-verifier.toml"
```

Files `include` others (merged in order; the file's own keys win). Paths are relative
to the file that names them. The interface ladder R0-R3 and the Tab. 5 ablations are
listed in [The interface ladder](ladder.md).

## Models

The models are separate from the agent, so the agent runs with any model:

```toml
--8<-- "configs/models/grok-low.toml"
```

```toml
--8<-- "configs/models/qwen3-vl-32b.toml"
```

A role is a hosted model (`backend = "openrouter"`, the default, or `openai_chat` for
any OpenAI-compatible server) or a service of the deploy (`service = "molmo2"`), with
its own settings on top. Keys are read from the environment variable the role names
(`api_key_env`, default `OPENROUTER_API_KEY`), never from a file. A self-hosted
server's URL is the same: `base_url_env` (the default model is DeepSeek, key `DEEPSEEK_API_KEY`).
`--model NAME` (or an
experiment's `model = "NAME"`) picks `configs/models/<NAME>.toml`.

## An experiment

```toml
--8<-- "configs/experiments/dev.toml"
```

`--set key.path=value` changes an experiment for one invocation; `--set agent.a.b=v`
and `--set model.a.b=v` change the agent and the model roles:

```bash
harness run configs/experiments/subset8.toml --model grok-medium --output runs/medium
harness run configs/experiments/dev.toml --set agent.runtime.keep_images=1 --set budget.max_cost_usd=2 --output runs/k1
harness run --tasks stack_bowls --seeds 0-2 --agent configs/ablations/r2.toml --output runs/r2
```

Without an experiment file, `harness run` runs the agent on RoboDojo's development
layouts for the tasks and seeds the command line gives.

## The lock and the identity

Resolving the agent against a deploy gives the driver's configuration and a **lock**:
everything that decides what the run computes and nothing about where it runs.

| In the lock | Not in it |
|---|---|
| the agent's version and configuration name | URLs, ports, GPUs |
| the prompt's sha256 and sources | whether a service is resident, on demand or external |
| the enabled skills, and the dropped ones with their reasons | API keys |
| the model roles (a service-backed role by its service name) | the machine |
| the services the skills use, and their manifests' sha256 | |
| the runtime settings and the budget | |

The lock's hash is the configuration's **identity** (16 hex characters). Every episode
records the lock, the identity, the agent version and the git commit in its
`manifest.json`; two runs with the same identity ran the same configuration. A paper
table, a leaderboard entry and an exported policy each name one.

```bash
harness lock --out agent.lock                          # the lock, the identity, a code fingerprint
harness config configs/experiments/dev.toml --json     # everything the experiment resolves to
```

## Strict resolution

The agent can lose skills on a machine that lacks their services, and a smaller
configuration has another identity. Evaluation runs (`layouts = "eval"`) therefore
resolve **strictly**: a service the agent names that the machine does not run stops
the run instead of running a smaller configuration, as does a skill named in
`skills.names` that cannot run. Skills an ablation leaves out by design (R0 has no
vision roles) drop on every machine and are not an error. `--strict` / `--no-strict`
override the default.
