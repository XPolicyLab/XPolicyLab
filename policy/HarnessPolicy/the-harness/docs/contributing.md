# Contributing

## Conventions

These are the repository's rules (`AGENTS.md`), with the reasons behind them:

- **One agent, versioned.** `configs/default.toml` and the code it runs are the agent;
  `AGENT_VERSION` goes into every lock and result. Variants are ablations (`configs/ablations/`) or
  experiment overrides, never copies; retired harnesses go to archive tags, and no code,
  config or directory is named after a harness or an iteration.
- **Capabilities are plugins**: skills, service manifests and prompt sections, turned on
  by configuration. A skill's calls are bounded executions that end with an event.
- **Configuration layers like other agent frameworks'**: code defaults, the agent, a model
  file, an experiment, the command line ([Configuration and identity](concepts/configuration.md)).
- **Evidence decides.** Develop under the evaluator's observation contract, on
  development layouts and evaluation layouts alike; a change enters the default when
  paired runs show a difference larger than a repeated run's spread, with calls and
  cost per episode reported too. Measure the agent against the learned policy it wraps
  on the same layouts.
- **The core stays standard-library Python 3.11.** `src/harness_policy` imports no
  Isaac, CUDA, torch or benchmark code at module level; those stay in worker, kernel and
  service processes. Configuration, resolution, locks and reports then work on any
  machine.
- **One environment.** The RoboDojo path uses the `harness-policy` conda environment.
  A dependency goes into a layer of `environment/` (`requirements-*.txt`), installs
  through `environment/setup.sh` under `environment/constraints.txt`, and its change is
  committed together with a refreshed `environment/lock/`.
- **No installs or paid calls in quickstarts and tests.** `scripts/install_plan.py`
  prints checkout plans; the tests start no simulator and call no model.
- **Traces keep everything.** Original observations, action receipts and errors are
  kept; a recorded replay is not a new closed-loop evaluation, and a mock is never a
  benchmark result.
- **Motions are tied to an episode and a step.** A possibly executed motion is never
  retried silently.
- **Comparisons are paired.** Environment configuration, task, seed and budget must
  match; failed cases stay in the denominator.
- **Nothing generated or secret is committed**: credentials, simulator assets,
  checkpoints, environments, run artifacts.

Pull requests: one topic per pull request; a change of several steps is one pull
request with one commit per step.

## Tests

```bash
python -m unittest discover -s tests
```

They cover the shield (property and fault-injection tests), the kernel and its
isolation, every skill against a fake robot, the agent's and every ablation's resolution and prompt
against every combination of optional skills, services and the broker, the suite, the
XPolicyLab connector and export against a fake host, and the documentation (below).
Tests that need Linux user namespaces, numpy/cv2 or a git checkout skip without them.

## Documentation

The site is MkDocs with the Material theme; its sources are `docs/` and `mkdocs.yml`.

```bash
mkdocs serve -a 127.0.0.1:8000     # live preview; on a remote machine: ssh -L 8000:127.0.0.1:8000 <host>
mkdocs build --strict              # what CI runs: a broken link or reference fails it
```

- **Reference pages are generated** at build time by `scripts/gen_docs_reference.py`
  from the code: the CLI from its argument parser, configuration keys from the
  dictionaries the validators use (`AGENT_KEYS`, `RUNTIME_KEYS`, `MODEL_KEYS`,
  `EXPERIMENT_KEYS`, `DEPLOY_KEYS`, `PLACEMENT_KEYS`, `MANIFEST_KEYS`, `SKILL_KEYS`), skills from their
  manifests and docstrings, services from their manifests, the agent by resolving it,
  the Python API from docstrings. Change the code, not the generated page.
- **A new configuration key needs a description**: add it to the dictionary next to
  the validator. `tests/test_docs.py` fails while a key the code reads is undocumented.
- **Commands in the docs are checked**: every `harness ...` command in `docs/` and
  `README.md` must parse with the CLI, and every repository path they mention must
  exist.
- **The tables in `README.md`** between `<!-- generated: ... -->` markers come from the
  same generator: `python scripts/gen_docs_reference.py --readme` rewrites them, and the
  tests fail while they are out of date.
- Real files can be shown with `--8<-- "path"` (snippets), so examples cannot drift
  from the configuration they show.

Each push and pull request builds the site in CI (`.github/workflows/docs.yml`) and
keeps it as a downloadable artifact. Publishing to GitHub Pages is prepared and
switched off; see the workflow file.
