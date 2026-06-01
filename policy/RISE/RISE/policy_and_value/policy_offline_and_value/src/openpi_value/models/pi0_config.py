import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi_value.models import model as _model
import openpi_value.models.gemma as _gemma
from openpi_value.shared import array_typing as at
import openpi_value.shared.nnx_utils as nnx_utils
from typing import Union, List

if TYPE_CHECKING:
    from openpi_value.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi_value.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)



@dataclasses.dataclass(frozen=True)
class Pi0Config_Custom(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    
    # max_token_len: int = 48
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)  # * discrete_state_input is equal to self.pi05
            # * self.pi05 with discrete_state_input
            # * self.pi0, no discrete_state_input


    # * Custom
    with_value_head: bool = False
    loss_action_weight: float = 1.0
    loss_value_weight: float = 1.0
    loss_value_use_bce: bool = False  # * Use BCE loss for value head
    loss_value_td_weight: float = 1.0  # * Weight for TD loss for value head
    
    exist_negative_progress: bool = False  # * Whether to use suboptimal progress labels for failure data.
    norm_progress: bool = False  # * Whether to normalize progress to [0, 1] range.
    fix_value_from_prefix: bool = False  # * Whether to fix the value prediction from the prefix embeddings only.
    freeze_vlm_backbone: bool = False  # * Whether to freeze the visual-language model during training.

    # * TD learning
    value_TD_learning: bool = False
    value_TD_TAU: float = 0.005
    value_gamma: float = 0.99
    value_terminal_window: int = 10
    value_failure_reward: float = -1.0
    value_success_reward: float = 1.0  # * Added success reward for value function

    p_with_progress_loss: float = 1.0

    p_mask_ego_state: float = 0.0

    apply_shape_visual_aug: bool = False  # * Whether to apply visual augmentation during training.
    apply_blur_visual_aug: bool = False  # * Whether to apply official visual augmentation during training.
    p_mask_base: float = 0.0  # * Probability to mask the base camera image during training.
    
    state_noise_snr: float | None = None  # * SNR for adding noise to the state input during training. None means no noise.
    fixed_noise_and_time: bool = False
    
    advantage_bins: Union[int, str, List[float], List[int]] = 10

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi_value.models.pi0 import Pi0
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # TODO: add history for spec
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                
                frame_index=jax.ShapeDtypeStruct([batch_size], jnp.float32),  # * Add frame_index to observation spec
                frame_index_progress=jax.ShapeDtypeStruct([batch_size], jnp.float32),  # * Add frame_index_progress to observation spec
                
                episode_length=jax.ShapeDtypeStruct([batch_size], jnp.float32),  # * Add episode_length to observation spec
                is_failure_data=jax.ShapeDtypeStruct([batch_size], jnp.bool_),  # * Add is_failure_data to observation spec
                is_infer_data=jax.ShapeDtypeStruct([batch_size], jnp.bool_),  # * Add is_infer_data to observation spec
                
                inferred_action=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # * Add inferred_action to observation spec
                noise=jax.ShapeDtypeStruct([batch_size, 1, 50, self.action_dim], jnp.float32),  # * Add noise to observation spec
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params. (all loras are trainable)
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
