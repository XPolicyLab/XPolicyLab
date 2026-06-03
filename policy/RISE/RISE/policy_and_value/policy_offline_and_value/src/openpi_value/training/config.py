"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
from enum import Enum
from enum import auto

import openpi_value.models.model as _model
import openpi_value.models.pi0_config as pi0_config
import openpi_value.models.tokenizer as _tokenizer
import openpi_value.shared.download as _download
import openpi_value.shared.normalize as _normalize
import openpi_value.training.optimizer as _optimizer
import openpi_value.training.weight_loaders as weight_loaders
import openpi_value.transforms as _transforms
import openpi_value.policies.custom_agilex_policy as custom_agilex_policy


ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter



class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


def _xpolicylab_repo_id(default):
    dataset = os.environ.get("RISE_XPOLICYLAB_DATASET")
    if not dataset:
        return default
    return [str(pathlib.Path(dataset).expanduser().resolve())]


def _xpolicylab_asset_id(default):
    dataset = os.environ.get("RISE_XPOLICYLAB_DATASET")
    return pathlib.Path(dataset).expanduser().resolve().name if dataset else default


_RISE_CAMERA_LAYOUTS: dict[str, dict[str, str]] = {
    "rise": {
        "top_head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    },
    "robodojo": {
        "top_head": "observation.images.cam_high",
        "hand_left": "observation.images.cam_left_wrist",
        "hand_right": "observation.images.cam_right_wrist",
    },
}


def _xpolicylab_lerobot_layout() -> str:
    layout = os.environ.get("RISE_LEROBOT_LAYOUT", "").strip().lower()
    if layout in _RISE_CAMERA_LAYOUTS:
        return layout

    dataset = os.environ.get("RISE_XPOLICYLAB_DATASET")
    if dataset:
        info_path = pathlib.Path(dataset).expanduser().resolve() / "meta" / "info.json"
        if info_path.is_file():
            import json

            features = json.loads(info_path.read_text()).get("features", {})
            if "observation.images.cam_high" in features:
                return "robodojo"

    return "rise"


def _xpolicylab_camera_keys() -> dict[str, str]:
    return _RISE_CAMERA_LAYOUTS[_xpolicylab_lerobot_layout()]


def _xpolicylab_image_repack(*, include_fut: bool = False) -> dict[str, str]:
    images = dict(_xpolicylab_camera_keys())
    if include_fut:
        for logical_name, dataset_key in _xpolicylab_camera_keys().items():
            images[f"fut_1_{logical_name}"] = f"fut_1_{dataset_key}"
    return images


def _xpolicylab_repack_group(
    *,
    with_advantage: bool = False,
    with_value_fields: bool = False,
) -> _transforms.Group:
    repack: dict[str, Any] = {
        "images": _xpolicylab_image_repack(include_fut=with_value_fields),
        "state": "observation.state",
        "actions": "action",
    }
    if with_advantage:
        repack["action_advantage"] = "action_advantage"
    if with_value_fields:
        repack.update(
            {
                "prompt": "prompt",
                "episode_length": "episode_length",
                "frame_index": "frame_index",
                "frame_index_progress": "frame_index_progress",
                "is_failure_data": "is_failure_data",
                "episode_index": "episode_index",
            }
        )
    return _transforms.Group(inputs=[_transforms.RepackTransform(repack)])


def _xpolicylab_default_prompt(default: str) -> str:
    return os.environ.get("RISE_DEFAULT_PROMPT", default)


def _xpolicylab_default_weights_dir() -> str:
    # policy/RISE/weights/pi05_base_pytorch
    policy_dir = pathlib.Path(__file__).resolve().parents[6]
    return str(policy_dir / "weights" / "pi05_base_pytorch")


def _xpolicylab_pytorch_weight_path(default: str | None = None) -> str | None:
    """PyTorch checkpoint dir (model.safetensors or model.pt). Override via RISE_PYTORCH_WEIGHT_PATH."""
    path = os.environ.get("RISE_PYTORCH_WEIGHT_PATH", "").strip()
    return path or default or _xpolicylab_default_weights_dir()


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

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


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
                assert isinstance(model_config, pi0_config.Pi0Config) or isinstance(model_config, pi0_config.Pi0Config_Custom)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            advantage_bins=model_config.advantage_bins,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING

    # repo_ids: list = tyro.MISSING
    
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        assets_dir = self.assets.assets_dir or str(assets_dirs)


        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(assets_dir), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            if isinstance(asset_id, list) and len(asset_id) == 1:
                asset_id = asset_id[0]

            else:
                assert self.assets.assets_dir is not None and \
                    self.assets.asset_id is not None, \
                    "Need to specify norm path when using multiple datasets"
            
            data_assets_dir = str(assets_dir / asset_id)

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




# * Our in-house aloha
@dataclasses.dataclass(frozen=True)
class LerobotCustomAgilexDataConfig(DataConfigFactory):
    """
    Configuration for the CustomAgilex robot dataset.
    This config handles the data transforms for the CustomAgilex robot's multi-camera setup and state/action space.
    """

    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    use_delta_joint_actions: bool = True

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # Repack transforms to match the dataset keys to the expected format
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=_xpolicylab_repack_group
    )

    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Create data transforms for inputs and outputs
        data_transforms = _transforms.Group(
            inputs=[
                custom_agilex_policy.CustomAgilexInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[custom_agilex_policy.CustomAgilexOutputs()],
        )

        # Apply delta action transform if enabled
        if self.use_delta_joint_actions:
            # Assuming first 13 dimensions are joints and last dimension is gripper
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)  # index 6, 13 is gripper
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Create model transforms
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        data_config = self.create_base_config(assets_dirs, model_config)
        if _xpolicylab_lerobot_layout() == "robodojo":
            data_config = dataclasses.replace(data_config, prompt_from_task=True)

        return dataclasses.replace(
            data_config,
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
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    
    # Global batch size.
    batch_size: int = 32 # * DEFAULT, total batch size for all gpus
    # batch_size: int = 48   # * CUSTOM

    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    
    # num_workers: int = 2  # * DEFAULT, slow
    num_workers: int = 12  # * Testing -- Faster


    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    
    
    save_interval: int = 2500  # * DEFAULT

    is_train: bool = True  # * Only use partial data in training
    
    split: str = 'all'  # * Use all data for training
    
    n_history: int = 0  # Number of history frames to use. If 0, no history will be used.
    n_future: int = 0   

    with_episode_start: bool = False  # If true, will use the episode start frame as the first frame in the history.
    preceding_skipping_ratio: float = 0.
    trailing_skipping_ratio: float = 0.
    
    use_suboptimal_progress: bool = False
    
    suboptimal_progress_multiplier: float = -1.0
    suboptimal_progress_offset: float = 0.0
    
    drop_last: bool = True  # If true, will drop the last incomplete batch.
    
    grad_accu_steps: int = 1  # Gradient accumulation steps.
    

    skip_norm_stats: bool = False


    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 2500

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
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    
    # * JAX -> PyTorch conversion helper (pi0.5 base)
    TrainConfig(
        name="Pi05_base_convert",
        model=pi0_config.Pi0Config(pi05=True),
        data=FakeDataConfig(),
    ),

    # * Compute_norm
    TrainConfig(
        name = "Compute_norm",
        model = pi0_config.Pi0Config(),


        data = LerobotCustomAgilexDataConfig(
            repo_id = _xpolicylab_repo_id([
                "data/sample_dataset",
            ]),
            default_prompt="Insert the memory stick.",
            use_delta_joint_actions=False,


            assets=AssetsConfig(
                assets_dir="data/norms",
                asset_id=_xpolicylab_asset_id("sample_dataset"),
            ),
        ),


        # * From scratch here
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_ckpt/pi0_base/params"),

        num_train_steps=100_000,
        keep_period=5000,
        save_interval=2000,

        num_workers=8,

        batch_size=64,  # * 8 gpus
    ),
    

    # * policy: pi05 style 
    TrainConfig(

        name="Pi05_style_training",

        model=pi0_config.Pi0Config_Custom(
            pi05=True,
            
            advantage_bins=10,
            
            apply_blur_visual_aug=True,
        ),
        
        data = LerobotCustomAgilexDataConfig(
            repo_id = _xpolicylab_repo_id([
                "data/sample_dataset",
            ]),
            
            assets=AssetsConfig(
                assets_dir="data/norms",
                asset_id=_xpolicylab_asset_id("sample_dataset"),
            ),
            
            default_prompt=_xpolicylab_default_prompt("fold the box."),
            use_delta_joint_actions=False,

            repack_transforms=_xpolicylab_repack_group(),
        ),
        
        pytorch_weight_path="path/to/ckpt",

        num_train_steps=100_000,
        keep_period=20000,
        save_interval=10000,

        num_workers=8,

        batch_size=64,  # * 8 gpus
    ),


    # * policy: recap style (add action advantage into policy input)
    TrainConfig(

        name="Policy_offline_release",

        model=pi0_config.Pi0Config_Custom(
            pi05=True,
            
            advantage_bins=10,
            
            apply_blur_visual_aug=True,
        ),
        
        data = LerobotCustomAgilexDataConfig(
            repo_id = _xpolicylab_repo_id([
                "data/sample_dataset",
            ]),
            
            assets=AssetsConfig(
                assets_dir="data/norms",
                asset_id=_xpolicylab_asset_id("sample_dataset"),
            ),
            
            default_prompt=_xpolicylab_default_prompt("fold the box."),
            use_delta_joint_actions=False,

            repack_transforms=_xpolicylab_repack_group(with_advantage=True),
        ),
        
        pytorch_weight_path=_xpolicylab_pytorch_weight_path(),

        num_train_steps=100_000,
        keep_period=20000,
        save_interval=10000,

        num_workers=8,

        batch_size=64,  # * 8 gpus
    ),


    # * value model: value_release
    TrainConfig(
        name="value_release",

        model=pi0_config.Pi0Config_Custom(
            pi05=True,
            with_value_head=True,
            loss_value_weight=1.,

            loss_value_use_bce=False,  # * [-1, 1]
            
            loss_action_weight=0.,
            p_mask_ego_state=1.,

            discrete_state_input=False,
            apply_blur_visual_aug=True,  # * Add custom visual aug

            p_with_progress_loss=1.,  # * Two losses at the same time.

            # * TD learning
            exist_negative_progress=True,
            value_TD_learning=True,
            
            value_TD_TAU=0.01,
            value_gamma=0.995,
            value_terminal_window=10,
            value_failure_reward=-0.6,
            
        ),
        
        data=LerobotCustomAgilexDataConfig(

            repo_id=_xpolicylab_repo_id([
                "data/sample_dataset",
            ]),
            
            assets=AssetsConfig(
                assets_dir="data/norms",
                asset_id=_xpolicylab_asset_id("sample_dataset"),
            ),
            
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            repack_transforms=_xpolicylab_repack_group(with_value_fields=True),
        ),

        pytorch_weight_path="path/to/ckpt",

        num_train_steps=100_000,
        keep_period=10000,
        save_interval=10000,

        num_workers=8,
        batch_size=64,  # * 8 gpus

        n_future=1,
    ),



    # * value model: visualization
    TrainConfig(
        name="vis_value_release_joint_T",

        model=pi0_config.Pi0Config_Custom(
            pi05=True,
            with_value_head=True,
            loss_value_weight=1.,

            loss_value_use_bce=False,  # * [-1, 1]
            
            loss_action_weight=0.,
            p_mask_ego_state=1.,

            discrete_state_input=False,
            apply_blur_visual_aug=True,  # * Add custom visual aug

            p_with_progress_loss=1.,  # * Two losses at the same time.

            # * TD learning
            exist_negative_progress=True,
            value_TD_learning=True,
            
            value_TD_TAU=0.01,  # * TOO Small
            
            value_gamma=0.995,
            value_terminal_window=10,
            value_failure_reward=-0.6,
            
        ),
        
        data=LerobotCustomAgilexDataConfig(

            repo_id=_xpolicylab_repo_id([
                "data/sample_dataset",
            ]),
            
            assets=AssetsConfig(
                assets_dir="data/norms",
                asset_id=_xpolicylab_asset_id("sample_dataset"),
            ),
            
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            repack_transforms=_xpolicylab_repack_group(with_value_fields=True),
        ),

        pytorch_weight_path="path/to/ckpt",

        num_train_steps=50_000,
        keep_period=10000,
        save_interval=10000,

        num_workers=8,
        batch_size=64,  # * 8 gpus

        n_future=1,
    ),


]




if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    # * print outliners
    for config in _CONFIGS:
        names = [c.name for c in _CONFIGS if c.name == config.name]
        if len(names) > 1:
            print(f"Duplicate config name found: {config.name}")
            
    for config in _CONFIGS:
        name_counts = sum(1 for c in _CONFIGS if c.name == config.name)
        if name_counts > 1:
            print(f"Config name '{config.name}' appears {name_counts} times.")
    
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
