"""Mask-aware component conversion for action profiles."""

from collections.abc import Callable

import numpy as np
import torch

from morph.convs import geometry
from morph.types.fields import ActionField
from morph.types.fields import ActionResult
from morph.types.fields import Array


def apply(action: Array, mask: Array) -> Array:
    """Zero action values marked invalid by a boolean mask."""
    if isinstance(action, torch.Tensor):
        return torch.where(mask, action, torch.zeros_like(action))
    return np.where(mask, action, np.zeros_like(action))


def encode_field(
    field: ActionField,
    action: Array,
    mask: Array,
    *,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> ActionResult:
    """Encode one field according to its component mask semantics."""
    from morph.convs import encode_field as converter

    return _convert_field(
        field,
        action,
        mask,
        converter=converter,
        source_dim=field.input_dim,
        target_dim=field.model_dim,
        ref_state=ref_state,
        ref_frame=ref_frame,
    )


def decode_field(
    field: ActionField,
    action: Array,
    mask: Array,
    *,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> ActionResult:
    """Decode one field according to its component mask semantics."""
    from morph.convs import decode_field as converter

    return _convert_field(
        field,
        action,
        mask,
        converter=converter,
        source_dim=field.model_dim,
        target_dim=field.output_dim,
        ref_state=ref_state,
        ref_frame=ref_frame,
    )


def _convert_field(
    field: ActionField,
    action: Array,
    mask: Array,
    *,
    converter: Callable[..., Array],
    source_dim: int,
    target_dim: int,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> ActionResult:
    output_mask, component_valid = _map_mask(
        field,
        mask,
        source_dim=source_dim,
        target_dim=target_dim,
    )
    if component_valid is None:
        output = converter(
            field,
            action,
            ref_state=ref_state,
            ref_frame=ref_frame,
        )
    else:
        output = _convert_atomic(
            field,
            action,
            component_valid,
            converter=converter,
            output_dim=target_dim,
            ref_state=ref_state,
            ref_frame=ref_frame,
        )
    return ActionResult(action=output, mask=output_mask)


def _map_mask(
    field: ActionField,
    mask: Array,
    *,
    source_dim: int,
    target_dim: int,
) -> tuple[Array, Array | None]:
    if mask.ndim < 2 or mask.shape[-1] != source_dim:
        raise ValueError(
            f"mask for field {field.name!r} must have shape "
            f"batch_shape + (T, {source_dim}), got {mask.shape}"
        )
    if not field.mask_is_atomic:
        if source_dim != target_dim:
            raise ValueError(
                f"non-atomic field {field.name!r} cannot map mask dimension "
                f"{source_dim} to {target_dim}"
            )
        return mask, None

    component_valid = mask[..., 0]
    source_valid = _expand_component(component_valid, mask.shape)
    if not _all(mask == source_valid):
        raise ValueError(
            f"mask for atomic field {field.name!r} must be all false or all "
            "true within each action step"
        )
    output_shape = mask.shape[:-1] + (target_dim,)
    return _expand_component(component_valid, output_shape), component_valid


def _convert_atomic(
    field: ActionField,
    action: Array,
    component_valid: Array,
    *,
    converter: Callable[..., Array],
    output_dim: int,
    ref_state: Array | None,
    ref_frame: Array | None,
) -> Array:
    flat_action = action.reshape(-1, action.shape[-1])
    flat_valid = component_valid.reshape(-1)
    flat_output = geometry.zeros(
        action,
        flat_action.shape[:-1] + (output_dim,),
    )
    if _any(flat_valid):
        flat_output[flat_valid] = converter(
            field,
            flat_action[flat_valid],
            ref_state=_select_steps(
                ref_state,
                action,
                flat_valid,
                trailing_dims=1,
            ),
            ref_frame=_select_steps(
                ref_frame,
                action,
                flat_valid,
                trailing_dims=2,
            ),
        )
    return flat_output.reshape(action.shape[:-1] + (output_dim,))


def _select_steps(
    value: Array | None,
    action: Array,
    valid: Array,
    *,
    trailing_dims: int,
) -> Array | None:
    if value is None:
        return None
    while value.ndim < action.ndim - 1 + trailing_dims:
        axis = -trailing_dims - 1
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(axis)
        else:
            value = np.expand_dims(value, axis=axis)
    shape = action.shape[:-1] + value.shape[-trailing_dims:]
    if isinstance(value, torch.Tensor):
        value = value.expand(shape)
    else:
        value = np.broadcast_to(value, shape)
    return value.reshape((-1,) + value.shape[-trailing_dims:])[valid]


def _expand_component(value: Array, shape: tuple[int, ...]) -> Array:
    expanded = value[..., None]
    if isinstance(value, torch.Tensor):
        return expanded.expand(shape)
    return np.broadcast_to(expanded, shape)


def _all(value: Array) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.all().item())
    return bool(value.all())


def _any(value: Array) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return bool(value.any())
