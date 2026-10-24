#!/usr/bin/env bash
# Validate the selected YAML and checkpoint before starting distributed workers.
# Source after setting ROOT and config.

# Explicit opt-in: ignore an inherited q99 ROBODOJO_JOINT_STATS value.
export ROBODOJO_JOINT_STATS="${ROBODOJO_ZSCORE_STATS:?Set ROBODOJO_ZSCORE_STATS to the z-score JSON artifact}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
: "${RUN_NAME:?Set a new RUN_NAME for z-score training}"
: "${OUTPUT_DIR:?Set a separate OUTPUT_DIR for z-score training}"

# Validate before the distributed launcher. A q99 optimizer checkpoint cannot
# resume in z-score coordinates; genuine z-score resumes remain supported.
if [[ -n "${WSP_PYTHON:-}" ]]; then
    zs_python="$WSP_PYTHON"
elif [[ -n "${CONDA_ENV:-${CONDA_PREFIX:-}}" ]]; then
    zs_python="${CONDA_ENV:-$CONDA_PREFIX}/bin/python"
else
    zs_python="$(command -v python)"
fi
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$zs_python" - "$config" "$@" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
from worldscape_policy.cli.config_composition import load_composed_config
from worldscape_policy.cli.config_profiles import resolve_config_profiles
from worldscape_policy.robodojo_joint import load_joint_statistics
# Match the training CLI: the final occurrence of a dot-list override wins.
overrides = OmegaConf.from_dotlist(sys.argv[2:])
config = resolve_config_profiles(load_composed_config(sys.argv[1]), overrides=overrides)
loader = config.data_loader
load_joint_statistics(loader.dataset_kwargs.stats_path, normalization="zscore")
if config.batch_adapter.diffusion_view_layout != 'mosaic_2x2':
    raise SystemExit('RoboDojo training supports only mosaic_2x2')
if not loader.dataset_kwargs.cache_video_decode and not loader.augment_video:
    raise SystemExit('mosaic with CACHE_VIDEO_DECODE=false requires AUGMENT_VIDEO=true to resize frames to 160x320')
resume = config.training.resume
checkpoints = list(Path(config.training.checkpoint_dir).glob('checkpoint-*'))
for checkpoint in checkpoints:
    path = checkpoint / 'transform_bundle.json'
    if not path.is_file():
        raise SystemExit('Use a separate OUTPUT_DIR: existing checkpoint has no z-score contract.')
    payload = json.loads(path.read_text())
    fields = payload.get('embodiments', {}).get('arx_x5_joint', {})
    if not fields or any(f['normalization'] != 'mean_std' for k in ('state_fields', 'action_fields') for f in fields[k]):
        raise SystemExit('Use a separate OUTPUT_DIR: existing checkpoints are not RoboDojo z-score.')
if resume is None:
    if checkpoints:
        raise SystemExit('OUTPUT_DIR already contains checkpoints: choose a new RUN_NAME/OUTPUT_DIR '
                         'or explicitly set training.resume to a compatible checkpoint.')
else:
    path = Path(resume) / 'transform_bundle.json'
    if not path.is_file():
        raise SystemExit('z-score resume requires a checkpoint with transform_bundle.json')
    payload = json.loads(path.read_text())
    fields = payload['embodiments']['arx_x5_joint']
    if any(f['normalization'] != 'mean_std' for k in ('state_fields', 'action_fields') for f in fields[k]):
        raise SystemExit('Cannot resume a q99 run with z-score. Start a new run from pretrained weights.')
PY
