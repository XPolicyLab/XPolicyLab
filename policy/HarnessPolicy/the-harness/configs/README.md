# Configs

There is one agent, `default.toml`. A run is the agent (or one of its ablations) with a
**model** on a **bench**, resolved against a machine's **deploy**; later layers win:

```text
configs/
├── default.toml   the agent: prompt pack, skills, services, runtime, budget, its default model
├── ablations/     the paper's ablations: each includes default.toml and switches parts off
├── models/        model roles: main, vlm, pointer, verifier (hosted models or services)
├── experiments/   the agent with a model on a bench: tasks, seeds, layouts, budgets, overrides
├── deploy/        per machine: GPUs, weights, where each service runs (examples; yours is git-ignored)
├── workers/       options for starting a worker by hand (harness serve --options ...)
└── scripts/       scripted model replies for pipeline checks
```

```bash
harness config configs/experiments/dev.toml --deploy mine    # skills, services, identity
harness run configs/experiments/dev.toml --deploy mine --output runs/dev
harness run --tasks stack_bowls --model grok-low --deploy mine --output runs/x
harness lock --out agent.lock                                 # the agent's identity
```

- The agent, models, experiments, locks and identity: [docs/concepts/configuration.md](../docs/concepts/configuration.md)
- `--model`, `--set`, includes, tasks, seeds, budgets: [docs/how-to/configure-runs.md](../docs/how-to/configure-runs.md)
- Every key: the documentation site's *Reference → Configuration keys* (generated)
