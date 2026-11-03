# Runtime installation

Linux x86_64, Python 3.11, NVIDIA CUDA 12 and a compatible driver are required.

## Python environment

Run from this policy directory. Choose writable storage for the environment and caches:

```bash
export WORK=/shared/rsi-runtime
mkdir -p "$WORK/tmp" "$WORK/cache/pip"
export TMPDIR="$WORK/tmp" PIP_CACHE_DIR="$WORK/cache/pip"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch" CUDA_CACHE_PATH="$WORK/cache/cuda"
export WARP_CACHE_PATH="$WORK/cache/warp" NUMBA_CACHE_DIR="$WORK/cache/numba"
export TORCH_EXTENSIONS_DIR="$WORK/cache/torch_extensions"
export MPLCONFIGDIR="$WORK/cache/matplotlib" PYTHONDONTWRITEBYTECODE=1
python3.11 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
bash install.sh
python -m pip check
python -B preflight.py --check-runtime
```

Model dependencies are in `worldscape-policy/pyproject.toml`; geometry and CUDA dependencies are in `requirements-runtime.txt`. CuRobo source is bundled and selected by `register_vendor()`. Warp is pinned to 1.17.0 for its CUDA 12 build.

The official evaluation shell scripts accept conda environment names. A conda environment or prefix with the same Python dependencies can be used for those entrypoints. The native outer worker accepts an interpreter path such as Isaac Sim's `python.sh`.

## Robot assets

Set `ROBODOJO_ROOT` to an installed RoboDojo checkout containing:
- `Assets/Robots/x5/X5A.urdf`, `curobo.yml` and referenced meshes;
- `env_cfg/arx_x5.yml`, `env_cfg/sim/sim_config.yml`;
- robot controller source.

Use the official robot configuration in the XPolicyLab parent workspace's `env_cfg/`. For training, the official checkout also provides `utils/robot/_robot_info.json`.

## Provider and weights

Configure `R2SI_API_BASE`, `R2SI_API_KEY` and, when supported by the provider, `R2SI_SESSION_RELEASE_PATH`. Download and bind Flash/Memory using the [policy guide](README.md#model-assets).

```bash
python -B preflight.py --check-runtime --check-cuda --check-robot-assets --check-checkpoints
```

GPU indices are relative to `CUDA_VISIBLE_DEVICES`. The native planner uses the first visible GPU. Include its memory requirements when placing WAMs on that device.

## Deployment modes

- **Official inner Agent:** `eval.sh` or the server/client setup scripts, one environment per Agent server.
- **Direct WAM:** `serve_robodojo.sh`, using `wam_deploy_robodojo.yml`.
- **Autonomous RSI:** the full repository's `scripts/iterate.py` and `scripts/local_worker.py`.
- **Remote WAM:** set `WRSI_POLICY_CLIENT_CONFIG` for a matching lease-aware service.

## Version maintenance

After editing a candidate copy's frozen assets:

```bash
python -B maintain_manifest.py --write-new-release
python -B maintain_manifest.py
```

The manifest binds runtime source, generic Skills, configuration and checkpoint hashes. `WRSI_INTEGRITY_MODE=full` selects full content verification between actions.

For outer-loop workspaces, use storage with atomic directory rename, fsync and SQLite writer locks. Set a short `R2SI_TEMP_ROOT` beneath `R2SI_STORAGE_ROOT` for deeply nested campaign paths.

