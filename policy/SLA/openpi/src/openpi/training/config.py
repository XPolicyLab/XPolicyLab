"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import glob
import importlib.util
import logging
import pathlib
import sys
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.droid_dreamzero_policy as droid_dreamzero_policy
import openpi.policies.molmoact_policy as molmoact_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.franka_hkust as franka_hkust
import openpi.policies.liberoplus_policy as liberoplus_policy
import openpi.policies.agilex_policy as agilex_policy
import openpi.policies.robotwin_policy as robotwin_policy
import openpi.policies.robocasa_no_movement_policy as robocasa_no_movement_policy
import openpi.policies.interndata_policy as interndata_policy
import openpi.policies.robocoin_policy as robocoin_policy
import openpi.policies.robodojo_policy as robodojo_policy
import openpi.policies.agibotbeta_policy as agibotbeta_policy
import openpi.policies.piper_policy as piper_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
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

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Pad the camera parameters (extrinsics and intrinsics) with 
    # zeros to match the expected input shape of the model.
    pad_camera_params: bool = False


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
    fast_local_loading: bool = False


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
    pad_camera_params: bool = False
    use_eef_action: bool = True

    # Optimize initialization of large mixtures of immutable local LeRobot datasets.
    # Note: setting this to True will disable the dataset validation process.
    # So load each component manually to check its completeness before setting this to True.
    fast_local_loading: bool = False

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
            pad_camera_params=self.pad_camera_params,
            fast_local_loading=self.fast_local_loading,
        )


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAgilexDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = False
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[agilex_policy.AgilexInputs(adapt_to_pi=self.adapt_to_pi, use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
            outputs=[agilex_policy.AgilexOutputs(adapt_to_pi=self.adapt_to_pi, use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = False
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[robotwin_policy.RobotwinInputs(adapt_to_pi=self.adapt_to_pi, use_eef_action=self.use_eef_action)],
            outputs=[robotwin_policy.RobotwinOutputs(adapt_to_pi=self.adapt_to_pi, use_eef_action=self.use_eef_action)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotInterndataSplitAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = False
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "images.rgb.head",
                        "cam_left_wrist": "images.rgb.hand_left",
                        "cam_right_wrist": "images.rgb.hand_right",
                    },
                    "state_left_joint": "states.left_joint.position",
                    "state_left_gripper":  "states.left_gripper.position",
                    "state_right_joint": "states.right_joint.position",
                    "state_right_gripper":  "states.right_gripper.position",
                    "actions_left_joint": "actions.left_joint.position",
                    "actions_left_gripper":  "actions.left_gripper.position",
                    "actions_right_joint": "actions.right_joint.position",
                    "actions_right_gripper":  "actions.right_gripper.position",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = (
        "actions.left_joint.position",
        "actions.left_gripper.position",
        "actions.right_joint.position",
        "actions.right_gripper.position",
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[
                interndata_policy.InterndataSplitAlohaInputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_eef_action=self.use_eef_action,
                    unified_gripper=self.unified_gripper,
                )
            ],
            outputs=[
                interndata_policy.InterndataSplitAlohaOutputs(
                    adapt_to_pi=self.adapt_to_pi,
                    use_eef_action=self.use_eef_action,
                    unified_gripper=self.unified_gripper,
                )
            ],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotInterndataLift2DataConfig(DataConfigFactory):
    """Lift2 InterData config using robot-frame EEF poses."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    use_so3_rpy: bool = False
    unified_gripper: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "images.rgb.head",
                            "cam_left_wrist": "images.rgb.hand_left",
                            "cam_right_wrist": "images.rgb.hand_right",
                        },
                        "state_left_eef": "states.left_ee_to_robot_pose",
                        "state_left_gripper": "states.left_gripper.position",
                        "state_right_eef": "states.right_ee_to_robot_pose",
                        "state_right_gripper": "states.right_gripper.position",
                        "actions_left_eef": "actions.left_ee_to_robot_pose",
                        "actions_left_gripper": "actions.left_gripper.position",
                        "actions_right_eef": "actions.right_ee_to_robot_pose",
                        "actions_right_gripper": "actions.right_gripper.position",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = (
        "actions.left_ee_to_robot_pose",
        "actions.left_gripper.position",
        "actions.right_ee_to_robot_pose",
        "actions.right_gripper.position",
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[interndata_policy.InterndataLift2Inputs(unified_gripper=self.unified_gripper)],
            outputs=[interndata_policy.InterndataLift2Outputs(unified_gripper=self.unified_gripper)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_joint_actions:
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotInterndataFrankaDataConfig(DataConfigFactory):
    """Single-arm Franka InterData config using a robot-frame EEF pose."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    use_so3_rpy: bool = False
    unified_gripper: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "images.rgb.head",
                            "cam_hand": "images.rgb.hand",
                        },
                        "state_eef": "states.ee_to_robot_pose",
                        "state_gripper": "states.gripper.position",
                        "actions_eef": "actions.ee_to_robot_pose",
                        "actions_gripper": "actions.gripper.position",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = (
        "actions.ee_to_robot_pose",
        "actions.gripper.position",
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[interndata_policy.InterndataFrankaInputs(unified_gripper=self.unified_gripper)],
            outputs=[interndata_policy.InterndataFrankaOutputs(unified_gripper=self.unified_gripper)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_joint_actions:
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # The repack transform is *only* applied to the data coming from the dataset,
    # and *not* during inference. We can use it to make inputs from the dataset look
    # as close as possible to those coming from the inference environment (e.g. match the keys).
    # Below, we match the keys in the dataset (which we defined in the data conversion script) to
    # the keys we use in our inference pipeline (defined in the inference script for libero).
    # For your own dataset, first figure out what keys your environment passes to the policy server
    # and then modify the mappings below so your dataset's keys get matched to those target keys.
    # The repack transform simply remaps key names here.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "index": "index",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type, unified_gripper=self.unified_gripper)],
            outputs=[libero_policy.LiberoOutputs(unified_gripper=self.unified_gripper)],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoPlusDataConfig(DataConfigFactory):
    """LIBERO Plus data config, which is modified from LeRobotLiberoDataConfig."""

    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "index": "index",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[liberoplus_policy.LiberoPlusInputs(unified_gripper=self.unified_gripper)],
            outputs=[liberoplus_policy.LiberoPlusOutputs(unified_gripper=self.unified_gripper)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaHKUSTDataConfig(DataConfigFactory):
    """Uses recorded EEF states as pose targets and the original gripper action labels."""

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.exterior_image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "index": "index",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("observation.state", "action")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        rotation_mask = _transforms.make_bool_mask(-3, 3, -1)
        data_transforms = _transforms.Group(
            inputs=[franka_hkust.FrankaHkustInputs()],
            outputs=[franka_hkust.FrankaHkustOutputs()],
        ).push(
            # Interpret rx/ry/rz as xyz Euler angles in radians; use rotvecs inside the model.
            inputs=[_transforms.EulerToRotvec(rotation_mask)],
            outputs=[_transforms.RotvecToEuler(rotation_mask)],
        ).push(
            # Use SO(3) deltas during training and restore absolute EEF targets at inference.
            inputs=[_transforms.DeltaActions(delta_action_mask, rotation_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask, rotation_mask)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasaNoMovementDataConfig(DataConfigFactory):

    # Convert gripper actions to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "observation/agentview_left": "observation.images.robot0_agentview_left",
                        "observation/agentview_right": "observation.images.robot0_agentview_right",
                        "observation/eye_in_hand": "observation.images.robot0_eye_in_hand",
                        "actions": "action",
                        "prompt": "prompt",
                        "index": "index",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[robocasa_no_movement_policy.RoboCasaInputs(unified_gripper=self.unified_gripper)],
            outputs=[robocasa_no_movement_policy.RoboCasaOutputs(unified_gripper=self.unified_gripper)],
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
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotOmniDROIDDataConfig(DataConfigFactory):
    """
    This dataclass is based on LeRobotDROIDDataConfig.
    We additionally add camera extrinsics and intrinsics to the repack transform,
    which are required for Omni-Spatial-Forcing (i.e., OmniVGGT forward process).
    !!! Note: It is only used when training, but not used during inference !!!
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation.images.exterior_1_left",
                        "observation/exterior_image_2_left": "observation.images.exterior_2_left",
                        "observation/wrist_image_left": "observation.images.wrist_left",
                        "observation/joint_position": "observation.state.joint_position",
                        "observation/gripper_position": "observation.state.gripper_position",
                        "extrinsics/exterior_image_1_left": "camera_extrinsics.exterior_1_left",
                        "extrinsics/exterior_image_2_left": "camera_extrinsics.exterior_2_left",
                        "intrinsics/exterior_image_1_left": "camera_intrinsics.exterior_1_left",
                        "intrinsics/exterior_image_2_left": "camera_intrinsics.exterior_2_left",
                        "file_record_path": "file_record_path",
                        "actions": "action",
                        "language_instruction": "language_instruction",
                        "language_instruction_2": "language_instruction_2",
                        "language_instruction_3": "language_instruction_3",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.OmniDroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDroidDreamzeroDataConfig(DataConfigFactory):
    """
    Data config for Droid-Dreamzero dataset in LeRobot format.
    https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data
    """
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = False
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # If true, will convert actions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_actions: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False    

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation.images.exterior_image_1_left",
                        "observation/exterior_image_2_left": "observation.images.exterior_image_2_left",
                        "observation/wrist_image_left": "observation.images.wrist_image_left",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Action keys that will be used to read the action sequence from the dataset.
        action_sequence_keys: Sequence[str] = ("action",)    
            
        data_transforms = _transforms.Group(
            inputs=[droid_dreamzero_policy.DroidInputs(model_type=model_config.model_type, use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
            outputs=[droid_dreamzero_policy.DroidOutputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_actions:
            if self.use_eef_action:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
            else:
                delta_action_mask = _transforms.make_bool_mask(7, -1)
            so3_mask = rpy_mask if self.use_so3_rpy else None
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask, so3_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask, so3_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMolmoActDataConfig(DataConfigFactory):
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = True
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # If true, will convert actions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_actions: bool = True
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "primary": "observation.images.primary",
                            "secondary": "observation.images.secondary",
                            "wrist": "observation.images.wrist",
                        },
                        "state": "observation.state",
                        "actions": "action.abs_ee_action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action.abs_ee_action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")
        if not self.use_eef_action:
            raise ValueError("MolmoAct data only supports EEF actions")

        data_transforms = _transforms.Group(
            inputs=[
                molmoact_policy.MolmoActInputs(
                    model_type=model_config.model_type,
                    unified_gripper=self.unified_gripper,
                )
            ],
            outputs=[molmoact_policy.MolmoActOutputs(unified_gripper=self.unified_gripper)],
        )
        rpy_mask = _transforms.make_bool_mask(-3, 3, -1)
        if self.use_so3_rpy:
            data_transforms = data_transforms.push(
                inputs=[_transforms.EulerToRotvec(rpy_mask)],
                outputs=[_transforms.RotvecToEuler(rpy_mask)],
            )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
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
class LeRobotRobocoinDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_actions: bool = True
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = True
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_head_rgb",
                        "cam_left_wrist": "observation.images.cam_left_wrist_rgb",
                        "cam_right_wrist": "observation.images.cam_right_wrist_rgb",
                    },
                    "eef_sim_pose_state": "eef_sim_pose_state",
                    "gripper_open_scale_state": "gripper_open_scale_state",
                    "eef_sim_pose_action": "eef_sim_pose_action",
                    "gripper_open_scale_action": "gripper_open_scale_action",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("eef_sim_pose_action", "gripper_open_scale_action")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[robocoin_policy.RobocoinInputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
            outputs=[robocoin_policy.RobocoinOutputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
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
class LeRobotRobodojoSimDataConfig(DataConfigFactory):
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
            inputs=[robodojo_policy.RobodojoSimInputs()],
            outputs=[robodojo_policy.RobodojoSimOutputs()],
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
            inputs=[robodojo_policy.RobodojoRealInputs(arm_base_distance=self.arm_base_distance)],
            outputs=[robodojo_policy.RobodojoRealOutputs(arm_base_distance=self.arm_base_distance)],
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
class LeRobotAgibotBetaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_actions: bool = True
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = True
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.head",
                        "cam_left_wrist": "observation.images.hand_left",
                        "cam_right_wrist": "observation.images.hand_right",
                    },
                    "grippers": "observation.states.effector.position",
                    "state_eef_orientation": "observation.states.end.orientation",
                    "state_eef_position": "observation.states.end.position",
                    "state_joint": "observation.states.joint.position",
                    "actions_eef_orientation": "actions.end.orientation",
                    "actions_eef_position": "actions.end.position",
                    "actions_joint": "actions.joint.position",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = (
        # The original gripper action value is delayed, while gripper state value is accurate,
        # therefore we use the observation value as the action target.
        # Note that gripper value 35 means opening, and 120 means closed.
        "observation.states.effector.position",
        "actions.end.orientation",
        "actions.end.position",
        "actions.joint.position",
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[agibotbeta_policy.AgibotBetaInputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
            outputs=[agibotbeta_policy.AgibotBetaOutputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
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
class LeRobotPiperDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_actions: bool = True
    # use eef instead of joint as action and state representations.
    use_eef_action: bool = True
    # If true, represent EEF states and rotation actions as SO3 rotation vectors.
    use_so3_rpy: bool = False
    # Convert gripper values to a unified representation (0 = closed, 1 = open).
    unified_gripper: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.right_front",
                        "cam_left_wrist": "observation.images.left_wrist",
                        "cam_right_wrist": "observation.images.right_wrist",
                    },
                    "joint_state": "observation.state",
                    "eef_state": "observation.ee_pose",
                    "joint_actions": "action",
                    "eef_actions": "action.ee_pose",
                    "prompt": "prompt",
                })
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action", "action.ee_pose")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.use_so3_rpy and not self.use_eef_action:
            raise ValueError("SO(3) rotation vectors require EEF actions to be enabled")

        data_transforms = _transforms.Group(
            inputs=[piper_policy.PiperInputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
            outputs=[piper_policy.PiperOutputs(use_eef_action=self.use_eef_action, unified_gripper=self.unified_gripper)],
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
    project_name: str = "SF_openpi"
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

    # Spatial Forcing configs
    vggt_weight_path: str | None = None
    vggt_dim: int = 1024

    vla_layers_align: int | None = None  # total 18 for paligemma-2b
    vla_adapt_layers_align: bool | None = None
    vggt_layers_align: int | None = None  # total 24 for VGGT
    vggt_adapt_layers_align: bool | None = None

    pooling_func: str | None = None
    use_vggt_pe: bool | None = None
    use_vlm_norm: bool | None = None
    use_camera_params: bool | None = None
    ignore_img_padding_area: bool | None = True

    save_SF_feature: str | None = None
    load_SF_offline: str | None = None

    align_loss_coeff: float = 0.0
    iSF_conv: bool = False
    iSF_zscore: bool = False

    align_enabled: bool = False
    align_target_model: Literal["auto", "vggt", "omnivggt"] = "auto"
    # "shared" means vggt model use the last 'align_vggt_devices' GPUs, while PI model use all visible GPUs.
    # "reserved" means vggt model use the last 'align_vggt_devices' GPUs, while PI model use other GPUs.
    align_device_mode: Literal["shared", "reserved"] = "shared"
    align_vggt_devices: int = 1
    align_feature_batch_size: int | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

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
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
        save_interval=30_000,
    ),
]


def is_local_repo_pattern(repo_id: str) -> bool:
    return repo_id.startswith(("/", "./", "../", "~")) and glob.has_magic(repo_id)


def expand_local_repo_ids(repo_id: str) -> Sequence[str]:
    """Expand local patterns into sorted LeRobot directories; leave other IDs unchanged."""
    if not is_local_repo_pattern(repo_id):
        return (repo_id,)

    expanded_pattern = str(pathlib.Path(repo_id).expanduser())
    matches = sorted(pathlib.Path(match) for match in glob.glob(expanded_pattern))
    dataset_dirs = [path for path in matches if path.is_dir() and (path / "meta").is_dir()]  # LerobotDataset format
    if not dataset_dirs:
        raise ValueError(f"Pattern '{repo_id}' did not match any valid LeRobot dataset directories.")
    return tuple(str(path) for path in dataset_dirs)


def glob_mixed_data_source(source: MixedDataSourceConfig) -> Sequence[MixedDataSourceConfig]:
    """
    repo_id can be wildcard, like ./data/intern/data/*
    Expanded datasets inherit the parent source weight; the mixed sampler combines
    that weight with each dataset's sample count.
    """
    if not is_local_repo_pattern(source.data_factory.repo_id):
        return (source,)

    return tuple(
        dataclasses.replace(
            source,
            name=f"{source.name}:{pathlib.Path(repo_id).name}",
            data_factory=dataclasses.replace(source.data_factory, repo_id=repo_id),
        )
        for repo_id in expand_local_repo_ids(source.data_factory.repo_id)
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

EXTERNAL_CONFIG_DIR = pathlib.Path(__file__).parents[1] / "external_config"
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
