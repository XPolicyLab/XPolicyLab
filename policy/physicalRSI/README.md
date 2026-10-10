# PhysicalRSI

**Contributor:** HKU MMLAB | **Paper:** TBD | **arXiv:** TBD | **Original code:** [yanming03/PhysicalRSI](https://github.com/yanming03/PhysicalRSI)

[PhysicalRSI](https://mmlab.hk/research/PhysicalRSI) combines an API-backed agent, frozen task-aware memory and an execution skill library for RoboDojo ARX X5. At episode start, the agent selects a registered skill composition that remains active until reset. The library contains pi05, pi05-sparse-memory and code-policy programs, with their internal weight identities recorded separately. Runtime source is included under `runtime/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

From `policy/physicalRSI`:

```bash
python download_assets.py --output skill-assets
uv sync --project skill-assets/implementations/openpi --python 3.11 --frozen --no-dev
policy_python="$PWD/skill-assets/implementations/openpi/.venv/bin/python"
bash install.sh "$policy_python"

PYTHONPATH=runtime "$policy_python" -m PhysicalRSI_baselines.robodojo.prepare_code_programs \
  "$PWD/skill-assets/implementations/code-skills"
PYTHONPATH=runtime "$policy_python" -m PhysicalRSI_baselines.robodojo.configure_skills \
  --assets "$PWD/skill-assets" \
  --framework "$(cd ../.. && pwd)" \
  --python "$policy_python" \
  --output "$PWD/skills.json" --evidence "$PWD/results" \
  --endpoint https://YOUR_API_HOST/v1/chat/completions --model YOUR_VISION_MODEL
```

The downloader first fetches the pinned release manifest, then verifies the source and weight archives against it. The manifest is downloaded data, not a task inventory embedded in this adapter. Its exact bytes and provenance are saved as `skill-assets/assets.json` and `skill-assets/assets.provenance.json`. To use another published skill library, pass `--manifest URL_OR_LOCAL_PATH --manifest-sha256 EXPECTED_SHA256` and a fresh `--output` directory; offline installation accepts a local manifest. This changes the asset release without editing the adapter. Benchmark observation and action adapters still need to implement the target benchmark contract. Code programs include their perception and motion-planning dependencies. Configuration generation registers the installed programs and both neural skills; it refuses to overwrite an existing configuration.

## Data Processing

Not required for this eval-only adapter. `process_data.sh` is omitted.

## Training

Not supported by this integration. `train.sh` is omitted.

## Evaluation

```bash
export PHYSICALRSI_SKILL_CONFIG="$PWD/skills.json"
export PHYSICALRSI_AGENT_API_KEY=YOUR_API_KEY
export PHYSICALRSI_EVAL_OUTPUT="$PWD/results"
PYTHONPATH=runtime "$policy_python" -m PhysicalRSI_baselines.robodojo.skill_preflight "$PHYSICALRSI_SKILL_CONFIG"

EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo general_pickup skill-library arx_x5 joint 0 0 0 \
  "$PWD/skill-assets/implementations/openpi/.venv" "$policy_python"
```

Use `EVAL_ENV_TYPE=sim` and the simulator's conda environment for simulation. The API endpoint must support image input and chat completions; `OPENAI_API_KEY` or `ARK_API_KEY` may also supply credentials.

- Neural skills use joint actions; code programs retain their declared joint or EEF contract. Use the matching evaluation action type.
- Code skills run in owned local processes over inherited pipes. They share the installed XPolicyLab source; there is no nested policy server or copied framework. Declared perception sidecars receive private loopback endpoints.
- Argument 9 accepts a Python executable, virtualenv, or uv project. `uv` uses `policy_uv_env_path` from `deploy.yml`.
- `result_dir` sets the default evidence directory, `obs_transform_pipeline` selects the observation transform, and `request_timeout_s` is the public client request budget. Set it in the deployment configuration before constructing the client. The adapter never patches transport internals. `deploy.py` uses the public demo-policy observation/action sequence.

Checkpoint-backed skills load during model initialization, before server readiness.
Reset clears episode state and reuses those loaded weights. GPU memory must fit
all configured checkpoint-backed skills; initialization failure stops startup.
The agent still chooses a composition from the first observation. An unknown
composition or an empty operation-library selection is rejected without changing
the agent decision. Code-program construction remains observation-dependent and
uses the explicit public request budget.

`runtime/` is an installable Python package installed by `install.sh`. The model
entrypoint subclasses only `ModelTemplate` and delegates to its owned runtime;
entrypoints do not change the interpreter import path.

The public client and private code worker default to a 1,800-second request budget.
The worker budget can be set explicitly with `request_timeout_s` in its program
configuration; startup has a separate `startup_timeout_s` setting.

The code library removes unused simulator-state readers and terminal evaluator
metadata helpers from its source. Source files are available under
`skill-assets/implementations/code-skills/implementations/` after download.
Programs remain task-specialized and may use calibrated camera/workspace planes
and object geometry priors; the library does not claim depth-based localization
throughout or invariance to arbitrary layouts. Native smoke results establish
execution, while full task comparisons and further localization changes are
validated separately.
