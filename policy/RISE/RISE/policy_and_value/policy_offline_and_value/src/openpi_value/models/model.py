import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

# from openpi_value.models_pytorch import pi0_pytorch
from openpi_value.shared import image_tools
import openpi_value.shared.array_typing as at
from typing import Callable, Optional, Union
# import openpi_value.policies.apolicy as aloha_policy
import warnings

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"



# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    # images: dict[str, at.Float[ArrayT, "*b h w c"]]
    images: dict[str, at.Float[ArrayT, "*b c h w"]]


    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # * string
    # prompt: str | None = None  # Optional, raw text prompt. Not used by the model.

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # * Custom
    episode_index: at.Int[ArrayT, "*b"] | None = None

    # frame_index: at.Int[ArrayT, "*b"] | None = None
    # * original frame_index
    frame_index: Union[at.Int[ArrayT, "*b"], at.Float[ArrayT, "*b"]]  | None = None
    
     
    # * modified frame_index for progress estimation
    frame_index_progress: Union[at.Int[ArrayT, "*b"], at.Float[ArrayT, "*b"]]  | None = None


    is_failure_data: at.Bool[ArrayT, "*b"] | None = None
    is_infer_data: at.Bool[ArrayT, "*b"] | None = None


    episode_length: Union[at.Int[ArrayT, "*b"], at.Float[ArrayT, "*b"]] | None = None

    action_advantage: Union[at.Int[ArrayT, "*b"], at.Float[ArrayT, "*b"]] | None = None

    action_advantage_original: Union[at.Int[ArrayT, "*b"], at.Float[ArrayT, "*b"]] | None = None

    image_original: dict[str, at.Float[ArrayT, "*b H W C"]] | None = None


    inferred_action: Union[at.Int[ArrayT, "*b d"], at.Float[ArrayT, "*b d"]] | None = None
    # noise: None | at.Float[ArrayT, "*b s"] = None
    
    
    noise: None | at.Float[ArrayT, "*b 1 50 d"] = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        
        # If images are uint8, convert them to [-1, 1] float32.

        if "image" in data:
            img_key = "image"
        else:
            raise ValueError("No image key found in observation.")

        for key in data[img_key]:
            if data[img_key][key].dtype == np.uint8:
                data[img_key][key] = data[img_key][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data[img_key][key], "dtype") and data[img_key][key].dtype == torch.uint8:
                data[img_key][key] = data[img_key][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        
        if data.get("image_original", None) is not None:
            for key in data["image_original"]:
                if data["image_original"][key].dtype == np.uint8:
                    data["image_original"][key] = data["image_original"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(data["image_original"][key], "dtype") and data["image_original"][key].dtype == torch.uint8:
                    data["image_original"][key] = data["image_original"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        # Handle noise reshaping: if noise is flattened [batch, 700], reshape to [batch, 1, 50, 14] 
        if data.get("noise", None) is not None:
            # Check if noise is flattened (2D with shape [batch, 700])
            if len(data["noise"].shape) == 2 and data["noise"].shape[1] == 700:
                # Reshape from [batch, 700] to [batch, 1, 50, 14]
                # 700 = 1 * 50 * 14
                data["noise"] = data["noise"].reshape(data["noise"].shape[0], 1, 50, 14)
            elif len(data["noise"].shape) == 2 and data["noise"].shape[1] == 50:  # * [batch, 50]
                # Reshape from [batch, 50] to [batch, 1, 50, 1] - might need action_dim
                # For now, assume action_dim=14 if shape[1] == 50*14
                data["noise"] = data["noise"].unsqueeze(1)

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),

            frame_index=data.get("frame_index"),
            frame_index_progress=data.get("frame_index_progress"),
            is_failure_data=data.get("is_failure_data"),
            is_infer_data=data.get("is_infer_data"),
            episode_length=data.get("episode_length"),

            action_advantage=data.get("action_advantage"),

            action_advantage_original=data.get("action_advantage_original", None),

            image_original=data.get("image_original", None),
            episode_index=data.get("episode_index", None),

            inferred_action=data.get("inferred_action", None),
            noise=data.get("noise", None),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result

    
    def drop_images(self, image_keys):
        for key in image_keys:
            self.images.pop(key, None)
            self.image_masks.pop(key, None)
        # return self

# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = [],   # ! Not used anymore
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    batch_shape = observation.state.shape[:-1]

    out_images = {}


    for key in observation.images:
        image = observation.images[key]
        
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])


    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        frame_index=observation.frame_index,
        frame_index_progress=observation.frame_index_progress,
        is_failure_data=observation.is_failure_data,
        is_infer_data=observation.is_infer_data,
        episode_length=observation.episode_length,
        action_advantage=observation.action_advantage,
        action_advantage_original=observation.action_advantage_original,
        image_original=observation.image_original,
        episode_index=observation.episode_index,
        inferred_action=observation.inferred_action,
        noise=observation.noise,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        
        def convert_keys_to_int(d):
            if isinstance(d, dict):
                return {int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return type(d)(convert_keys_to_int(x) for x in d)
            return d


        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)

        params = convert_keys_to_int(params)

        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        from openpi_value.models_pytorch import pi0_pytorch

        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)

        if str(weight_path).endswith(".pt") or str(weight_path).endswith(".pth"):
            ckpt = torch.load(weight_path, map_location="cpu") 
            model.load_state_dict(ckpt, strict=False)
        elif str(weight_path).endswith(".safetensors"):
            safetensors.torch.load_model(model, weight_path)
            
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
