"""What the adapter needs from a checkpoint directory to build a SANA-Robot deployment.

A checkpoint directory is the run directory of ``train.sh`` or a released checkpoint in the same layout::

    <checkpoint>/
    ├── config.yaml                          the training config: the policy's arguments and the data settings
    ├── meta.json                            the description of the training dataset
    ├── normalization.json                   the normalization of the robot rows
    └── checkpoints/latest/model_ema.pth     the weights (or one ``*.pth`` file in the directory)

Every piece can be given by a ``deploy.yml`` key instead; see the policy README.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_SANA_ROOT = POLICY_DIR / "Sana"
SANA_ROOT_ENV = "SANA_ROBOT_ROOT"
VAE_ENV = "SANA_ROBOT_VAE"
TEXT_ENCODER_ENV = "SANA_ROBOT_TEXT_ENCODER"

TRAIN_CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
INDEX_FILE = "episodes.jsonl"
NORMALIZATION_FILE = "normalization.json"
WEIGHTS_FILE = "model_ema.pth"
CHECKPOINT_FOLDER = "checkpoints"
LATEST_LINK = "latest"
RECIPES = {"ar": "configs/sana_robot/robodojo_ar.yaml", "base": "configs/sana_robot/robodojo_base.yaml"}

# deploy.yml keys that are fields of ``sana_robot``'s ``DeployConfig``.
DEPLOY_FIELDS = (
    "steps",
    "video_flow_shift",
    "action_flow_shift",
    "async_delay_steps",
    "n_action_steps",
    "optimized",
    "precision",
    "device",
)
# What a release of the package before it took these names calls three of them.
EARLIER_NAMES = {"chunk_size": "chunk_rows", "n_action_steps": "period_rows", "async_delay_steps": "delay_rows"}


def package_name(owner, name: str) -> str:
    """Return the name under which ``owner``, a config or a controller of the package, holds a setting of the adapter."""
    return name if hasattr(owner, name) else EARLIER_NAMES.get(name, name)


def setting(owner, name: str):
    """Return a setting of the adapter from a config or a controller of the package."""
    return getattr(owner, package_name(owner, name))


@dataclass(frozen=True)
class RunFiles:
    """The files of a checkpoint directory and the settings its policy was trained with."""

    root: Path
    weights: Path
    normalization: Path
    meta: Path
    vae: Path
    text_encoder: Path
    policy: dict
    view_sizes: Optional[list]
    chunk_size: Optional[int]
    video_frame_stride: Optional[int]


def sana_root(model_cfg: Mapping[str, Any]) -> Path:
    """Return the checkout of the SANA repository: the ``sana_robot_root`` key, ``$SANA_ROBOT_ROOT``, or ``Sana/`` in the policy directory."""
    root = model_cfg.get("sana_robot_root") or os.environ.get(SANA_ROOT_ENV) or DEFAULT_SANA_ROOT
    return Path(os.path.expanduser(str(root))).resolve()


def import_sana_robot(root: Path) -> None:
    """Make the ``sana_robot`` package importable: as installed, else from the checkout at ``root``."""
    try:
        importlib.import_module("sana_robot")
        return
    except ImportError:
        pass
    if not (root / "sana_robot").is_dir():
        raise ImportError(
            f"the sana_robot package is not installed and {root} holds no checkout of it; run install.sh, or set "
            f"sana_robot_root in deploy.yml or ${SANA_ROOT_ENV}"
        )
    sys.path.insert(0, str(root))
    importlib.import_module("sana_robot")


def _existing(candidates, what: str) -> Path:
    """Return the first existing path of ``candidates``, which may hold None."""
    paths = [Path(os.path.expanduser(str(path))) for path in candidates if path]
    for path in paths:
        if path.exists():
            return path.resolve()
    checked = "\n  ".join(str(path) for path in paths) or "<nothing to check>"
    raise FileNotFoundError(f"SANA_Robot found no {what}. Checked:\n  {checked}")


def _under(base: Path, path) -> Optional[Path]:
    """Return ``path`` under ``base`` when it is relative; None for an empty path."""
    if not path:
        return None
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else base / path


def find_weights(root: Path, model_cfg: Mapping[str, Any]) -> Path:
    """Return the weights file of a checkpoint directory: ``checkpoint_file``, the step ``checkpoint_step`` of a run, its latest step, or the directory's only ``*.pth`` file."""
    if root.is_file():
        return root
    if model_cfg.get("checkpoint_file"):
        return _existing([_under(root, model_cfg["checkpoint_file"])], "weights file (checkpoint_file)")
    steps = root / CHECKPOINT_FOLDER
    if model_cfg.get("checkpoint_step") is not None:
        step = steps / f"step_{int(model_cfg['checkpoint_step'])}" / WEIGHTS_FILE
        return _existing([step], "weights file (checkpoint_step)")
    if (steps / LATEST_LINK / WEIGHTS_FILE).exists():
        return (steps / LATEST_LINK / WEIGHTS_FILE).resolve()
    files = sorted(root.glob("*.pth"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"SANA_Robot expects {steps / LATEST_LINK / WEIGHTS_FILE} or exactly one *.pth file in {root}, found "
            f"{[file.name for file in files]}; name the file with checkpoint_file"
        )
    return files[0].resolve()


def resolve_run(model_cfg: Mapping[str, Any], checkpoint: Path, repository: Path) -> RunFiles:
    """Return the files and the training settings of a checkpoint.

    A ``deploy.yml`` key comes first, then what the checkpoint directory holds, then what its training config
    names; paths of the training config are relative to the repository, where training runs. A directory without
    a training config reads the RoboDojo recipe of the repository for its ``policy_type``.

    Args:
        model_cfg: The deploy config.
        checkpoint: The checkpoint directory, or a weights file whose directory holds the other files.
        repository: The checkout of the SANA repository.

    Returns:
        The files and the settings.
    """
    checkpoint = Path(checkpoint)
    root = checkpoint.parent if checkpoint.is_file() else checkpoint
    recipe = repository / RECIPES[model_cfg.get("policy_type") or "ar"]
    config_file = root / TRAIN_CONFIG_FILE if (root / TRAIN_CONFIG_FILE).exists() else recipe
    train_config = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    data = train_config.get("data") or {}
    policy = model_cfg.get("policy") or train_config.get("policy")
    if policy is None:
        raise FileNotFoundError(
            f"SANA_Robot found the policy's arguments neither in {root / TRAIN_CONFIG_FILE} nor in {recipe}; "
            "set the policy key in deploy.yml"
        )
    dataset = _under(repository, data.get("path"))
    return RunFiles(
        root=root,
        weights=find_weights(checkpoint, model_cfg),
        normalization=_existing(
            [
                model_cfg.get("normalization_path"),
                root / NORMALIZATION_FILE,
                _under(repository, data.get("normalization")),
            ],
            "normalization file (normalization_path)",
        ),
        meta=_existing(
            [model_cfg.get("dataset_meta_path"), root / META_FILE, dataset / META_FILE if dataset else None],
            "dataset description meta.json (dataset_meta_path)",
        ),
        vae=_existing(
            [model_cfg.get("vae_path"), os.environ.get(VAE_ENV), _under(repository, train_config.get("vae"))],
            "LTX-2 video VAE directory (vae_path)",
        ),
        text_encoder=_existing(
            [
                model_cfg.get("text_encoder_path"),
                os.environ.get(TEXT_ENCODER_ENV),
                _under(repository, train_config.get("text_encoder")),
            ],
            "Gemma 2 2B directory (text_encoder_path)",
        ),
        policy=dict(policy),
        view_sizes=model_cfg.get("view_sizes") or data.get("view_sizes"),
        chunk_size=model_cfg.get("chunk_size") or data.get("chunk_size") or data.get(EARLIER_NAMES["chunk_size"]),
        video_frame_stride=model_cfg.get("video_frame_stride") or data.get("video_frame_stride"),
    )


def description_dataset(meta: Path) -> tempfile.TemporaryDirectory:
    """Return a dataset directory that holds the description of the training dataset and no episode.

    A ``sana_robot`` deployment reads the control frequency, the cameras, the prompt fields and the robot slots
    from the directory of its training dataset. Evaluation has the description alone, so the deployment gets a
    directory with that file and an empty episode index. The caller keeps the returned object for as long as the
    directory is needed.
    """
    directory = tempfile.TemporaryDirectory(prefix="sana_robot_dataset_")
    shutil.copyfile(meta, Path(directory.name) / META_FILE)
    (Path(directory.name) / INDEX_FILE).write_text("")
    return directory


def build_deployment(model_cfg: Mapping[str, Any], run: RunFiles, dataset: str):
    """Load the models of a checkpoint as a ``sana_robot`` ``Deployment``.

    Args:
        model_cfg: The deploy config; its keys of ``DEPLOY_FIELDS`` that are set replace the defaults of
            ``DeployConfig``.
        run: The checkpoint's files and settings.
        dataset: The dataset directory of ``description_dataset``.

    Returns:
        The deployment, its video stream the one of ``use_causal_vae``.
    """
    from sana_robot.deploy.runtime.deployment import DeployConfig, Deployment

    settings = {
        "policy_type": model_cfg.get("policy_type") or "ar",
        "checkpoint": str(run.weights),
        "vae": str(run.vae),
        "text_encoder": str(run.text_encoder),
        "policy": run.policy,
        "dataset": str(dataset),
        "normalization": str(run.normalization),
        "view_sizes": run.view_sizes,
        package_name(DeployConfig, "chunk_size"): run.chunk_size,
        "video_frame_stride": run.video_frame_stride,
    }
    settings.update({package_name(DeployConfig, name): model_cfg.get(name) for name in DEPLOY_FIELDS})
    deployment = Deployment(DeployConfig(**{name: value for name, value in settings.items() if value is not None}))
    use_causal_vae(deployment)
    return deployment


def use_causal_vae(deployment) -> None:
    """Give a deployment the video stream of SANA's causal LTX-2 VAE, the encoder of the checkpoints' training latents; see ``video_stream``."""
    from .video_stream import CausalViewStream, load_causal_encoder

    encoder, normalize = load_causal_encoder(deployment.config.vae, deployment.device)
    deployment.stream = CausalViewStream(encoder, normalize)
