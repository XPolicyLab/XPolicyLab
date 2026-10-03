# PhysicalRSI

**Contributor:** HKU MMLAB | **Project:** [PhysicalRSI](https://mmlab.hk/research/PhysicalRSI) | **Original code:** [yanming03/PhysicalRSI](https://github.com/yanming03/PhysicalRSI)

PhysicalRSI combines an API-backed agent, frozen task-aware memory and an execution skill library for RoboDojo ARX X5. At episode start, the agent selects a registered skill composition that remains active until reset. The library contains pi05, pi05-sparse-memory and code-policy programs, with their internal weight identities recorded separately. Runtime source is included under `runtime/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

From `policy/physicalRSI`:

```bash
python download_assets.py --output skill-assets
uv sync --project skill-assets/implementations/openpi --python 3.11 --frozen --no-dev
bash install.sh "$PWD/skill-assets/implementations/openpi/.venv/bin/python"

PYTHONPATH=runtime python -m PhysicalRSI_baselines.robodojo.prepare_code_programs \
  "$PWD/skill-assets/implementations/code-skills"
PYTHONPATH=runtime python -m PhysicalRSI_baselines.robodojo.configure_skills \
  --assets "$PWD/skill-assets" \
  --python "$PWD/skill-assets/implementations/openpi/.venv/bin/python" \
  --output "$PWD/skills.json" --evidence "$PWD/results" \
  --endpoint https://YOUR_API_HOST/v1/chat/completions --model YOUR_VISION_MODEL
```

The downloader verifies the source and weight archives against `assets.json`. Code programs include their perception and motion-planning dependencies. Configuration generation registers the installed programs and both neural skills; it refuses to overwrite an existing configuration.

## Data Processing

Not required for this eval-only adapter. `process_data.sh` is omitted.

## Training

Not supported by this integration. `train.sh` is omitted.

## Evaluation

```bash
export PHYSICALRSI_SKILL_CONFIG="$PWD/skills.json"
export PHYSICALRSI_AGENT_API_KEY=YOUR_API_KEY
export PHYSICALRSI_EVAL_OUTPUT="$PWD/results"
PYTHONPATH=runtime python -m PhysicalRSI_baselines.robodojo.skill_preflight "$PHYSICALRSI_SKILL_CONFIG"

EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo general_pickup skill-library arx_x5 joint 0 0 0 \
  "$PWD/skill-assets/implementations/openpi/.venv" base
```

Use `EVAL_ENV_TYPE=sim` and the simulator's conda environment for simulation. The API endpoint must support image input and chat completions; `OPENAI_API_KEY` or `ARK_API_KEY` may also supply credentials.

- Neural skills use joint actions; code programs retain their declared joint or EEF contract. Use the matching evaluation action type.
- Run one code-program service per host or isolated container because its sidecars use fixed ports.
- Argument 9 accepts a Python executable, virtualenv, or uv project. `uv` uses `policy_uv_env_path` from `deploy.yml`.
- `result_dir` sets the default evidence directory, `obs_transform_pipeline` selects the observation transform, and `physicalrsi_startup_timeout_s` controls server startup timeout. `deploy.py` uses the bundled PhysicalRSI evaluation loop.
