"""Training configs.

Each config lives in its own file under `external_config/` and is registered by `load_external_configs`.
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import glob
import importlib.util
import logging
import pathlib
import sys
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.robodojo_policy as robodojo_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Extra frames fed to the model as additional views, on top of the current observation.
    # Each entry is a frame offset: >= 0 anchors at the episode's first frame (0 being that
    # frame itself), < 0 is relative to the current frame (-25 = 25 frames back). Empty
    # disables the mechanism. Every entry costs 256 prefix tokens per camera listed below.
    extra_frames: tuple[int, ...] = ()
    # LeRobot keys of the cameras every extra frame is taken from.
    extra_frame_keys: tuple[str, ...] = ("observation.images.cam_high",)


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Overrides the model's default normalization mode when set.
    # None means model's default normalization mode. True/False means quantile/z-score normalization, respectively.
    use_quantile_norm: bool | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        base_config = self.base_config or DataConfig()
        norm_stats = base_config.norm_stats
        if norm_stats is None:
            norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        return dataclasses.replace(
            base_config,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=norm_stats,
            use_quantile_norm=(
                model_config.model_type != ModelType.PI0
                if self.use_quantile_norm is None
                else self.use_quantile_norm
            ),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        data_assets_dir = str(assets_dir / asset_id)
        try:
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class MixedDataConfig(DataConfig):

    components: Sequence[DataConfig] = ()
    weights: Sequence[float] = ()


@dataclasses.dataclass(frozen=True)
class MixedDataSourceConfig:
    """One source inside a mixed dataset config."""

    name: str
    weight: float
    data_factory: tyro.conf.Suppress[DataConfigFactory]


@dataclasses.dataclass(frozen=True)
class MixedLeRobotDataConfig(DataConfigFactory):

    repo_id: str = "mixed"
    sources: Sequence[MixedDataSourceConfig] = ()

    # If set, overrides the normalization mode for the mixed config and every component.
    # If None, each component keeps the normalization mode selected by its own factory.
    use_quantile_norm: bool | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MixedDataConfig:
        shared_config = self.create_base_config(assets_dirs, model_config)
        components: list[DataConfig] = []
        weights: list[float] = []

        for source in self.sources:
            if source.weight <= 0:
                raise ValueError(f"Mixed data source '{source.name}' must have weight > 0, got {source.weight}.")

            for source_dataset in glob_mixed_data_source(source):
                component_factory = dataclasses.replace(
                    source_dataset.data_factory,
                    assets=AssetsConfig(
                        assets_dir=self.assets.assets_dir,
                        asset_id=shared_config.asset_id,
                    ),
                    base_config=dataclasses.replace(
                        source_dataset.data_factory.base_config or DataConfig(),
                        norm_stats=shared_config.norm_stats,
                    ),
                )
                component = component_factory.create(assets_dirs, model_config)
                component_use_quantile_norm = (
                    component.use_quantile_norm
                    if self.use_quantile_norm is None
                    else self.use_quantile_norm
                )
                components.append(
                    dataclasses.replace(
                        component,
                        asset_id=shared_config.asset_id,
                        norm_stats=shared_config.norm_stats,
                        use_quantile_norm=component_use_quantile_norm,
                    )
                )
                weights.append(source_dataset.weight)

        mixed_use_quantile_norm = (
            shared_config.use_quantile_norm
            if self.use_quantile_norm is None
            else self.use_quantile_norm
        )
        return MixedDataConfig(
            repo_id=shared_config.repo_id,
            asset_id=shared_config.asset_id,
            norm_stats=shared_config.norm_stats,
            use_quantile_norm=mixed_use_quantile_norm,
            components=components,
            weights=weights,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobodojoDataConfig(DataConfigFactory):
    use_delta_actions: bool = True
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = dataclasses.field(kw_only=True)
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "task_index": "task_index",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.use_eef_action:
            raise NotImplementedError("Joint action representation for Robodojo is not implemented.")

        data_transforms = _transforms.Group(
            inputs=[robodojo_policy.RobodojoInputs()],
            outputs=[robodojo_policy.RobodojoOutputs()],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobodojoRealDataConfig(DataConfigFactory):
    use_delta_actions: bool = True
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = dataclasses.field(kw_only=True)
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # The common robot-frame origin is the midpoint between the parallel arm bases.
    arm_base_distance: float = 0.600
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "task_index": "task_index",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.use_eef_action:
            raise NotImplementedError("Joint action representation for Robodojo is not implemented.")

        data_transforms = _transforms.Group(
            inputs=[robodojo_policy.RobodojoInputs(arm_base_distance=self.arm_base_distance)],
            outputs=[robodojo_policy.RobodojoOutputs(arm_base_distance=self.arm_base_distance)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "SLA_Memory"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    # Probability of hiding the robot state from each training sample.
    state_dropout_prob: float = 0.0

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of micro-batches whose gradients are averaged before each optimizer update.
    gradient_accumulation_steps: int = 1
    # Number of data loader workers per training process. Increasing this number will speed up data loading but
    # will increase memory and CPU usage on every rank.
    num_workers: int = 2
    # Number of optimizer update steps to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def effective_batch_size(self) -> int:
        """Global batch size represented by one optimizer update."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")
        if not 0.0 <= self.state_dropout_prob <= 1.0:
            raise ValueError("state_dropout_prob must be between 0 and 1.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS: list["TrainConfig"] = []


def glob_mixed_data_source(source: MixedDataSourceConfig) -> Sequence[MixedDataSourceConfig]:
    """
    repo_id can be wildcard, like ./data/intern/data/*
    Expanded datasets inherit the parent source weight; the mixed sampler combines
    that weight with each dataset's sample count.
    """
    repo_id = source.data_factory.repo_id
    if not repo_id.startswith(("/", "./", "../", "~")) or not glob.has_magic(repo_id):
        return (source,)

    expanded_pattern = str(pathlib.Path(repo_id).expanduser())
    matches = sorted(pathlib.Path(match) for match in glob.glob(expanded_pattern))
    dataset_dirs = [path for path in matches if path.is_dir() and (path / "meta").is_dir()]  # LerobotDataset format
    if not dataset_dirs:
        raise ValueError(
            f"Mixed data source '{source.name}' pattern '{repo_id}' did not match any valid LeRobot dataset directories."
        )

    return tuple(
        dataclasses.replace(
            source,
            name=f"{source.name}:{dataset_dir.name}",
            data_factory=dataclasses.replace(source.data_factory, repo_id=str(dataset_dir)),
        )
        for dataset_dir in dataset_dirs
    )


def load_external_configs(directory_path: str) -> list[TrainConfig]:
    external_configs = []
    path = pathlib.Path(directory_path)

    if not path.exists():
        logging.warning(f"External config path '{path}' does not exist. Skipping.")
        return []

    print(f"Loading external configs from: {path}")

    for file_path in path.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        try:
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                found_in_file = 0
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    if isinstance(attr, TrainConfig):
                        external_configs.append(attr)
                        found_in_file += 1

                    elif isinstance(attr, list) and attr:
                        if all(isinstance(item, TrainConfig) for item in attr):
                            external_configs.extend(attr)
                            found_in_file += len(attr)

                logging.info(f"Loaded {found_in_file} configs from {file_path.name}")

        except Exception as e:
            logging.error(f"Failed to load config from {file_path}: {e}")
            import traceback
            traceback.print_exc()

    return external_configs

# Found next to this file, so every checkout (or vendored copy) reads its own external configs.
EXTERNAL_CONFIG_DIR = str(pathlib.Path(__file__).parent.parent / "external_config")
_CONFIGS.extend(load_external_configs(EXTERNAL_CONFIG_DIR))

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
