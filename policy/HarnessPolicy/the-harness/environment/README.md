# Project environment

One conda environment, `harness-policy`, runs everything on the RoboDojo path:
the framework CLI, the RoboDojo simulator worker, the kernel that executes
model-written programs, and learned-skill servers. They run as separate
processes on the same interpreter.

| Layer | Source | Pinned by |
|---|---|---|
| Python 3.11, PyTorch 2.7.0 (CUDA 12.8) | conda, download.pytorch.org | `constraints.txt` |
| Isaac Sim 5.1, Isaac Lab, cuRobo | pypi.nvidia.com, RoboDojo submodules | `constraints.txt`, `third_party/sources.toml` |
| RoboDojo runtime, XPolicyLab client | RoboDojo checkout | `constraints.txt` |
| harness-policy | this repository (editable) | — |
| Learned skills and perception | `requirements-skills.txt` | `constraints.txt` |
| Documentation site (MkDocs) | `requirements-docs.txt` | `constraints.txt` |

Every `pip install` runs with `--constraint constraints.txt`, so no project
dependency can move torch, numpy, Isaac Sim, or websockets. If a package
cannot be installed under the constraints, the install fails instead of
upgrading the core.

## Use

```bash
source environment/activate.sh
python environment/check.py
```

`activate.sh` sets `HARNESS_ROBODOJO_ROOT`, accepts the Omniverse EULA, and
sources `~/.harness_env` for machine-specific settings such as
`OPENROUTER_API_KEY`, `HF_ENDPOINT`, or `TMPDIR`. Keep that file private
(`chmod 600`) and out of Git.

## Create

The RoboDojo checkout (with submodules) must exist first; its revisions are in
`third_party/sources.toml`. Assets are downloaded separately with RoboDojo's
`<RoboDojo>/scripts/init_assets.sh`.

```bash
bash environment/setup.sh                   # all steps
bash environment/setup.sh --from project    # resume
bash environment/setup.sh --only lock       # refresh the lock files
```

Steps: `conda torch isaacsim isaaclab curobo robodojo project skills docs verify lock`.
A different environment name can be chosen with `HARNESS_ENV`.

## Lock files

`lock/conda-linux-64.txt` (`conda list --explicit`) and `lock/pip-freeze.txt`
record the exact environment that passed `check.py` and the RoboDojo smoke run.
Editable installs (Isaac Lab, cuRobo, XPolicyLab, this repository) are listed
as comments because they point at local checkouts.

## Known deviations from upstream

- `isaacsim-kernel` declares `websockets==12.0`, which only Kit's asset-search
  extensions use; headless simulation does not load them. XPolicyLab policy
  servers need the websockets 13+ asyncio API, so the environment pins 17.1.
  `pip check` reports this one conflict; the RoboDojo smoke run covers it.
- Xiaomi-Robotics-1 upstream uses torch 2.8, transformers 4.57.1, numpy 2.1,
  and flash-attn. Here it runs on the pinned core with SDPA attention.
- `mmengine` is installed with `--no-deps` so it does not add `opencv-python`
  next to `opencv-python-headless`.

## Adding a dependency

Add it to `requirements-skills.txt` (or a new layer), run
`bash environment/setup.sh --from skills`, then a RoboDojo smoke run, and
commit the refreshed `lock/` together with the change.
