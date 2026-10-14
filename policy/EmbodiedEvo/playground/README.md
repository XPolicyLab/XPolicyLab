# EmbodiedEvo Playground source

This directory bundles the Playground implementation from the supplied
EmbodiedEvo source checkout, previously named `EmbodiedRSI`. It runs independently
of that checkout. The [source manifest](SOURCE_MANIFEST.json) records the upstream
commit and hashes of imported files; [third-party notices](THIRD_PARTY_NOTICES.md)
cover the unchanged public recipes and their license.

Start with the [policy's Playground instructions](../README.md#playground).
`../playground.sh` uses the policy Python environment, preserves caller-relative
paths, and launches this source in a separate process so its `orchestrator`,
`simulators` and `environments` packages do not enter the Test server's namespace.

```bash
# From this directory, after simulator provisioning and provider configuration:
PLAYGROUND_RUN="$PWD/runs/$(date +%Y%m%d%H%M)_stack_bowls"
bash ../playground.sh explore --task RoboDojo --env-task stack_bowls \
  --run-dir "$PLAYGROUND_RUN" --gpu 0
bash ../playground.sh export --run-dir "$PLAYGROUND_RUN" \
  --output "${PLAYGROUND_RUN}_workspace"
```

The export is a workspace checkpoint accepted by the outer policy's `eval.sh`.
It copies only verified frozen `skills/` and `lessons/`, grouped by task, and
supplies fresh Test templates. It preserves the Playground recipe condition.
Programs, observations, private ledgers, credentials and simulator files do not
transfer. Existing exports are never overwritten.

| Path | Purpose |
| --- | --- |
| `orchestrator/` | Explore, persistent sessions, freeze, isolated agents and direct Test |
| `environments/`, `simulators/` | Python execution, native actions, Playground reset and raw feedback |
| `tasks/`, `tasks_baseline/` | Handwritten phase instructions, primitive manuals and recipes |
| `integrations/robodojo_policy/` | Upstream optional native-evaluator transport, including Playground reset |
| `scripts/`, `docker/`, `configs/` | Provisioning, diagnostics and configuration examples |
| `entrypoint.py`, `export_workspace.py` | XPolicyLab launcher and frozen-workspace export |
| `tests/` | CPU checks; native transport tests need a provisioned RoboDojo/XPolicyLab |

The optional native transport uses a separate subprocess and selects its own
bundled `EmbodiedEvo` adapter; normal Test deployment uses the outer adapter.
The upstream direct `test`/`run` commands remain available. See the
[execution protocol](docs/PROTOCOL.md), [installation](docs/INSTALL.md) and
[running guide](docs/RUNNING.md). These upstream guides describe commands from
this directory; their optional standalone `.venv` is separate from `../.venv`.
The default agent image tag is aligned with the outer policy:
`embodiedevo-agent:xpolicy`.

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

CPU checks do not establish real simulator or model performance. Simulator
assets, virtualenvs, credentials, local runtime profiles and historical results
are excluded from this source bundle.
