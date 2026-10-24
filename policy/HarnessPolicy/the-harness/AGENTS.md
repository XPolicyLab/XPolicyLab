# Development conventions

## One agent

- There is one agent: `configs/default.toml` with the code it runs. Its version is
  `AGENT_VERSION` (`src/harness_policy/__init__.py`): raise the minor version for a change
  in what it does (prompt, skills, defaults, high-level loop), the patch version for a fix.
- No named harnesses. Code, configs and directories carry no harness or iteration names
  (z1, z2, r4, v03, ...). A variant is an ablation (`configs/ablations/<what it changes>.toml`,
  including `../default.toml`) or an experiment override, never a copy. A retired harness
  goes to an archive tag or branch (`archive/pre-default-agent`, `archive/z2`,
  `archive/step-loop`), not into main.
- Capabilities are plugins: a skill package (`skills/<name>/`: `skill.toml` + code), a
  service manifest (`services/<name>.toml`, what the model is told about it in
  `client.description`), a prompt pack section. A skill's calls are bounded executions
  that end with an event; it keeps no state across episodes (experience goes through the
  library, frozen for evaluation).

## Configuration

- Layers, later ones winning: code defaults < `configs/default.toml` < `configs/models/<name>.toml`
  (`--model`) < `configs/experiments/<name>.toml` < `--agent`, `--set`. Where services run
  is the deploy's (`configs/deploy/`). Keys live in environment variables, never in files.
- A configuration key the code reads is documented in the dictionary next to its validator.
- Every result records the lock and identity (agent version, prompt, skills, models,
  services, runtime), the git commit and whether the tree was dirty.

## Evidence

- The agent sees what the evaluator hands a policy (`runtime.contract`); more is a
  development diagnostic or a separately reported setting.
- Develop on development layouts and evaluation layouts alike; a run on evaluation
  layouts measures one frozen configuration (`layouts = "eval"` resolves strictly).
  Compare the agent with the learned policy it wraps on the same layouts.
- One hypothesis per change. A change enters the default when development runs show it
  beats the noise: paired comparisons (same tasks, seeds, budget, model), the spread of a
  repeated run next to the difference, calls and cost per episode next to the score.
  Failed cases stay in the denominator.
- The paper is a reference, not a specification: when results contradict it, change the
  code, then the paper text, and keep `docs/concepts/method-conformance.md` true.

## Code

- Keep the core (`src/harness_policy`) standard-library Python 3.11 at import: Isaac,
  CUDA, torch and benchmark code stay in worker, kernel and service processes. Run
  `python -m unittest discover -s tests` for every change.
- One environment: the `harness-policy` conda environment defined in `environment/`.
  Install packages only through `environment/setup.sh` under `environment/constraints.txt`,
  and commit a refreshed `environment/lock/` with any dependency change.
- `scripts/setup_*.sh` print checkout plans only. No installs, asset downloads or paid API
  calls in quickstarts or tests.
- Preserve original observations, action receipts and errors in traces. A recorded
  observation replay is not a new closed-loop evaluation, and a mock is never a result.
- Motion requests are tied to an episode and observation step. Never silently retry a
  possibly executed motion.
- Never commit credentials, simulator assets, checkpoints, environments or run artifacts.
- One topic per pull request; a change of several steps is one pull request with one
  commit per step.
