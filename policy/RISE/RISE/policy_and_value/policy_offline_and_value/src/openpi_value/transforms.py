from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import List, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
import torch

# from openpi_value.shared.image_tools import batch_resize, resize_with_pad

from openpi_value.models import tokenizer as _tokenizer
from openpi_value.shared import array_typing as at
from openpi_value.shared import normalize as _normalize

import numpy as np
from PIL import Image
from typing import Union
import copy
# import torch
import json

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")



def _resize_pil(image: Image.Image, size: Union[int, tuple[int, int]], method: int) -> Image.Image:
    """Resizes a single image using PIL without padding.

    Args:
        image: The PIL Image to resize.
        size: The target size.
            - If int: Resizes the shortest edge to this size, maintaining aspect ratio.
            - If (height, width): Resizes directly to this size, potentially distorting.
        method: The interpolation method.

    Returns:
        The resized PIL Image.
    """
    if isinstance(size, int):
        # Case 1: Resize shortest edge to `size`
        cur_width, cur_height = image.size

        # Optimization: If shortest edge is already `size`, no resize needed.
        if min(cur_width, cur_height) == size:
            return image

        # Calculate new dimensions
        short_edge = min(cur_width, cur_height)
        ratio = size / short_edge
        new_width = int(round(cur_width * ratio))
        new_height = int(round(cur_height * ratio))

        # PIL resize expects (width, height)
        return image.resize((new_width, new_height), resample=method)

    elif isinstance(size, (tuple, list)) and len(size) == 2:
        # Case 2: Resize directly to (height, width)
        target_height, target_width = size
        cur_width, cur_height = image.size

        # Optimization: If size is already correct, no resize needed.
        if (cur_width, cur_height) == (target_width, target_height):
            return image

        # PIL resize expects (width, height)
        return image.resize((target_width, target_height), resample=method)

    else:
        raise ValueError(f"Unsupported size format: {size}. Must be int or (height, width).")


def resize_no_pad(images: np.ndarray, size: Union[int, tuple[int, int]], method=Image.BILINEAR) -> np.ndarray:
    """Resizes a batch of images without padding, following tf.image.resize logic.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        size: The target size.
            - If int: Resizes the shortest edge to this size, maintaining aspect ratio.
            - If (height, width): Resizes directly to this size, potentially distorting.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images.
        - If size=(h, w), shape is [..., h, w, channel].
        - If size=int, shape is [..., new_h, new_w, channel]. Note: This
          will raise an error if images in the batch have different aspect
          ratios, as they cannot be stacked into a single tensor.
    """
    original_shape = images.shape
    
    # --- Optimization: Check if resize is a no-op ---
    if isinstance(size, int):
        # Check shortest edge
        if min(original_shape[-3], original_shape[-2]) == size:
            return images
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        # Check exact (h, w)
        if original_shape[-3:-1] == size:
            return images
    # -------------------------------------------------

    # Reshape to a flat batch of images
    images = images.reshape(-1, *original_shape[-3:])

    # Handle empty batch case
    if images.shape[0] == 0:
        if isinstance(size, (tuple, list)):
            h, w = size
            return np.empty((*original_shape[:-3], h, w, original_shape[-1]), dtype=images.dtype)
        else:
            # Cannot determine output shape for 'int' size on empty batch,
            # so return the original empty batch.
            return images.reshape(original_shape)

    # Apply the helper function to each image in the batch
    resized_list = [_resize_pil(Image.fromarray(im), size, method=method) for im in images]

    # Stack them back into a single tensor
    # This will automatically convert PIL Images to np.ndarray
    resized = np.stack(resized_list)

    # Reshape back to original batch dimensions
    new_shape = resized.shape[-3:]  # (new_h, new_w, c)
    return resized.reshape(*original_shape[:-3], *new_shape)



@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)




        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:

        if self.prompt is not None:
            data["prompt"] = np.asarray(self.prompt) # ! use default prompt, if exists
        
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # if 'inferred_action' in data:

        #     # before_norm = data['inferred_action'].clone()

        #     data['inferred_action'] = data['inferred_action'].reshape(data['actions'].shape[0], -1)

        #     self.norm_stats['inferred_action'] = \
        #         self.norm_stats['actions']

        out = apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

        # if 'inferred_action' in data:
        #     out['inferred_action'] = out['inferred_action'].reshape(-1)
        #     self.norm_stats.pop('inferred_action')

        #     # after_norm = data['inferred_action']

        return out

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01 = pad_to_dim(stats.q01, x.shape[-1], axis=-1, value=0.0)[..., : x.shape[-1]]
        q99 = pad_to_dim(stats.q99, x.shape[-1], axis=-1, value=1.0)[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:

        data["image_original"] = {k: v.copy() for k, v in data["image"].items()}
        data["image_original"] = {k: resize_no_pad(v, size=self.height) for k, v in data["image_original"].items()}  # * resize without padding, resize the shorter side to height

        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    # advantage_bins: int = 10
    advantage_bins: Union[int, str, List[float], List[int]] = 10

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
        # if (prompt := data.get("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        
        action_advantage = data.get("action_advantage", None)
        
        action_advantage_original = copy.deepcopy(action_advantage)

        if action_advantage is not None:

            if len(action_advantage.shape) == 0:  # * True, get in

                if type(action_advantage) is torch.Tensor:
                    action_advantage = action_advantage.cpu().numpy()

                # * discretize advantages are in ints of [1, self.advantage_bins], 1, 2, 3, xxx, self.advantage_bins
                

                if isinstance(self.advantage_bins, list):
                    bins = np.array(self.advantage_bins)  # * manual advantage.
                elif isinstance(self.advantage_bins, int):
                    bins = np.linspace(-1, 1, self.advantage_bins + 1)[:-1]  # * add 0.1 offset to set 0 to low bins.
                else:
                    raise NotImplementedError(f"advantage_bins type {type(self.advantage_bins)} not supported.")
                
                action_advantage = np.digitize(action_advantage, bins=bins)  # * add 0.1 offset to set 0 to low bins.
                
                
        tokens, token_masks = self.tokenizer.tokenize(prompt, state, action_advantage=action_advantage)
        
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks,

                # * Custom
                "action_advantage": action_advantage,
                "action_advantage_original": action_advantage_original,
                }




@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:

        
        
        if self.tasks is not None:
            if "task_index" not in data:
                raise ValueError('Cannot extract prompt without "task_index"')
            
            task_index = int(data["task_index"])
            if (prompt := self.tasks.get(task_index)) is None:
                raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")
        else:
            assert "task" in data, 'Cannot extract prompt without "task"'
            prompt = data["task"]

            # * For gelexea dataset, only keep the part after '@'

        
        if '@' in prompt:
            prompt = prompt.split('@')[-1].strip()
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
