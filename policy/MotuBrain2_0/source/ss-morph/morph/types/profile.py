"""Action profile declarations and backend-neutral layout orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import torch

from morph.types.fields import ActionField
from morph.types.fields import ActionResult
from morph.types.fields import Array
from morph.types.fields import Chassis
from morph.types.fields import EEF
from morph.types.fields import Gripper
from morph.types.fields import ModelRepr
from morph.types.fields import Qpos
from morph.types.fields import RotRepr
from morph.types.normalization import EmbodimentInput
from morph.types.normalization import NormalizationConfig

ActionInput: TypeAlias = Array | Mapping[str, Array]


@dataclass(frozen=True)
class ActionProfile:
    """Versioned mapping between semantic and model action layouts.

    One profile may normalize a mixed-embodiment batch when its normalization
    stats are keyed by embodiment. In that case actions have shape
    ``batch_shape + (T, D)`` and the embodiment labels have exactly
    ``batch_shape``. Reference states have ``batch_shape + (D,)`` and reference
    frames have ``batch_shape + (4, 4)``. An unbatched ``(T, D)`` action takes
    one string label, one state vector, and one camera transform.
    """

    name: str
    fields: tuple[ActionField, ...]
    normalization: NormalizationConfig | None = None
    use_ref_frame: bool = False
    version: str = "v1"
    _normalizer: object | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action profile name must not be empty")
        if not self.version:
            raise ValueError("action profile version must not be empty")
        if not self.fields:
            raise ValueError("action profile fields must not be empty")
        if not isinstance(self.fields, tuple):
            object.__setattr__(self, "fields", tuple(self.fields))
        names = [item.name for item in self.fields]
        if any(name is None for name in names):
            raise ValueError("action profile fields require explicit names")
        if len(names) != len(set(names)):
            raise ValueError(f"action profile field names must be unique: {names}")
        for item in self.fields:
            if isinstance(item, Qpos) and item.dim is None:
                raise ValueError(
                    f"action profile Qpos field {item.name!r} requires explicit dim"
                )
        for space in ("input", "model", "output"):
            if not any(getattr(item, f"{space}_used") for item in self.fields):
                raise ValueError(f"action profile must contain a used {space} field")
        if not isinstance(self.use_ref_frame, bool):
            raise TypeError("use_ref_frame must be bool")
        has_relative_eef = any(
            isinstance(item, EEF)
            and item.model_repr == "relative"
            and (item.input_used or item.output_used)
            for item in self.fields
        )
        if self.use_ref_frame and not has_relative_eef:
            raise ValueError(
                "use_ref_frame=True requires at least one used relative EEF field"
            )
        normalizer = None
        if self.normalization is not None:
            normalizer = self.normalization.build(
                fields=self.fields,
            )
        object.__setattr__(self, "_normalizer", normalizer)

    @property
    def input_dim(self) -> int:
        """Dimension of the full input layout."""
        return sum(item.input_dim for item in self.fields)

    @property
    def used_input_dim(self) -> int:
        """Dimension of the compact input layout."""
        return sum(item.input_dim for item in self.fields if item.input_used)

    @property
    def model_dim(self) -> int:
        """Dimension of the full model layout."""
        return sum(item.model_dim for item in self.fields)

    @property
    def used_model_dim(self) -> int:
        """Dimension of the compact model layout."""
        return sum(item.model_dim for item in self.fields if item.model_used)

    @property
    def output_dim(self) -> int:
        """Dimension of the full output layout."""
        return sum(item.output_dim for item in self.fields)

    @property
    def used_output_dim(self) -> int:
        """Dimension of the compact output layout."""
        return sum(item.output_dim for item in self.fields if item.output_used)

    def _mask_values(self, space: str) -> torch.Tensor:
        values = []
        for item in self.fields:
            values.extend(
                [getattr(item, f"{space}_used")] * getattr(item, f"{space}_dim")
            )
        return torch.tensor(values, dtype=torch.bool)

    @property
    def input_mask(self) -> torch.Tensor:
        """Static used mask over the full input layout."""
        return self._mask_values("input")

    @property
    def model_mask(self) -> torch.Tensor:
        """Static used mask over the full model layout."""
        return self._mask_values("model")

    @property
    def output_mask(self) -> torch.Tensor:
        """Static used mask over the full output layout."""
        return self._mask_values("output")

    def clamp_model_action(
        self,
        action: Array,
        *,
        action_dim_axis: int = -1,
    ) -> Array:
        """Zero dimensions disabled by the static model mask."""
        from morph.convs import geometry

        if action.ndim == 0:
            raise ValueError("action must have an action dimension")
        axis = (
            action_dim_axis if action_dim_axis >= 0 else action.ndim + action_dim_axis
        )
        if axis < 0 or axis >= action.ndim:
            raise ValueError(
                f"action_dim_axis {action_dim_axis} is out of range for "
                f"shape {action.shape}"
            )
        if action.shape[axis] != self.model_dim:
            raise ValueError(
                f"action dim on axis {action_dim_axis} is "
                f"{action.shape[axis]}, expected {self.model_dim}"
            )
        shape = [1] * action.ndim
        shape[axis] = self.model_dim
        mask = geometry.bool_array(self.model_mask.reshape(shape), action)
        if isinstance(action, torch.Tensor):
            return torch.where(mask, action, torch.zeros_like(action))
        return np.where(mask, action, np.zeros_like(action))

    def encode(
        self,
        action: ActionInput,
        *,
        embodiment: EmbodimentInput | None = None,
        action_mask: Array | None = None,
        ref_state: ActionInput | None = None,
        ref_frame: Array | None = None,
        ignore_unused: bool = False,
    ) -> ActionResult:
        """Convert semantic inputs to normalized model representation.

        Args:
            action: Semantic action with its field dimension last. For
                embodiment-keyed normalization, its shape must be
                ``batch_shape + (T, D)``.
            embodiment: Labels selecting normalization stats. It must be a
                string for an unbatched ``(T, D)`` action, or have shape
                ``batch_shape`` for a batched action. The time dimension is
                not repeated in this value.
            action_mask: Optional boolean validity with the same shape as a
                full-layout array ``action``. Field mask semantics determine
                which invalid values skip conversion.
            ref_state: Optional semantic reference state for relative fields.
                For an action shaped ``batch_shape + (T, D)``, this has shape
                ``batch_shape + (D,)``.
            ref_frame: Optional reference transforms for frame-relative fields.
                For an action shaped ``batch_shape + (T, D)``, this has shape
                ``batch_shape + (4, 4)``.
            ignore_unused: Whether to return only statically used dimensions.

        Returns:
            The normalized model action and either its dynamic converted mask
            or, when ``action_mask`` is omitted, its static model-layout mask.
        """
        from morph.convs import encode_field
        from morph.convs import geometry
        from morph.convs import mask as mask_conv

        parts = self._split(action, space="input", label="action")
        mask_parts = self._split_action_mask(
            action,
            action_mask,
            space="input",
        )
        refs = self._split_reference(ref_state)
        prototype = _prototype(parts, "action")
        self._validate_reference_shapes(refs, prototype)
        frame = self._validate_frame(ref_frame, prototype)
        outputs = []
        output_masks = []
        for item in self.fields:
            if not item.input_used or not item.model_used:
                output = geometry.zeros(
                    prototype,
                    prototype.shape[:-1] + (item.model_dim,),
                )
                if action_mask is not None:
                    output_masks.append(geometry.bool_array(output, prototype))
            elif action_mask is None:
                output = encode_field(
                    item,
                    parts[item.name],
                    ref_state=refs.get(item.name),
                    ref_frame=frame,
                )
            else:
                converted = mask_conv.encode_field(
                    item,
                    parts[item.name],
                    mask_parts[item.name],
                    ref_state=refs.get(item.name),
                    ref_frame=frame,
                )
                output = converted.action
                if converted.mask is None:
                    raise RuntimeError("masked field encode did not return a mask")
                output_masks.append(converted.mask)
            outputs.append(output)
        result = geometry.concatenate(outputs, axis=-1)
        if self._normalizer is not None:
            result = self._normalizer.normalize(result, embodiment)
        if action_mask is not None:
            mask = geometry.bool_array(
                geometry.concatenate(output_masks, axis=-1),
                result,
            )
            result = mask_conv.apply(result, mask)
        else:
            mask = geometry.bool_array(self.model_mask, result)
        if ignore_unused:
            static_mask = geometry.bool_array(self.model_mask, result)
            return ActionResult(action=result[..., static_mask], mask=None)
        return ActionResult(action=result, mask=mask)

    def decode(
        self,
        action: ActionInput,
        *,
        embodiment: EmbodimentInput | None = None,
        action_mask: Array | None = None,
        ref_state: ActionInput | None = None,
        ref_frame: Array | None = None,
        ignore_unused: bool = False,
    ) -> ActionResult:
        """Convert model actions to semantic output representation.

        Args:
            action: Model action with its field dimension last. For
                embodiment-keyed normalization, its shape must be
                ``batch_shape + (T, D)``.
            embodiment: Labels selecting normalization stats. It must be a
                string for an unbatched ``(T, D)`` action, or have shape
                ``batch_shape`` for a batched action. The time dimension is
                not repeated in this value.
            action_mask: Optional boolean validity with the same shape as a
                full-layout array ``action``. Field mask semantics determine
                which invalid values skip conversion.
            ref_state: Optional semantic reference state for relative fields.
                For an action shaped ``batch_shape + (T, D)``, this has shape
                ``batch_shape + (D,)``.
            ref_frame: Optional reference transforms for frame-relative fields.
                For an action shaped ``batch_shape + (T, D)``, this has shape
                ``batch_shape + (4, 4)``.
            ignore_unused: Whether to return only statically used dimensions.

        Returns:
            The semantic output action and either its dynamic converted mask
            or, when ``action_mask`` is omitted, its static output-layout mask.
        """
        from morph.convs import decode_field
        from morph.convs import geometry
        from morph.convs import mask as mask_conv

        parts = self._split(action, space="model", label="action")
        mask_parts = self._split_action_mask(
            action,
            action_mask,
            space="model",
        )
        refs = self._split_reference(ref_state)
        prototype = _prototype(parts, "action")
        self._validate_reference_shapes(refs, prototype)
        frame = self._validate_frame(ref_frame, prototype)
        full_parts = []
        for item in self.fields:
            value = parts.get(item.name)
            if value is None or not item.model_used:
                value = geometry.zeros(
                    prototype,
                    prototype.shape[:-1] + (item.model_dim,),
                )
            full_parts.append(value)
        full_action = geometry.concatenate(full_parts, axis=-1)
        if self._normalizer is not None:
            full_action = self._normalizer.denormalize(
                full_action,
                embodiment,
            )

        outputs = []
        output_masks = []
        offset = 0
        for item in self.fields:
            value = full_action[..., offset : offset + item.model_dim]
            offset += item.model_dim
            if not item.output_used:
                output = geometry.zeros(
                    prototype,
                    prototype.shape[:-1] + (item.output_dim,),
                )
                if action_mask is not None:
                    output_masks.append(geometry.bool_array(output, prototype))
            elif action_mask is None:
                output = decode_field(
                    item,
                    value,
                    ref_state=refs.get(item.name),
                    ref_frame=frame,
                )
            else:
                converted = mask_conv.decode_field(
                    item,
                    value,
                    mask_parts[item.name],
                    ref_state=refs.get(item.name),
                    ref_frame=frame,
                )
                output = converted.action
                if converted.mask is None:
                    raise RuntimeError("masked field decode did not return a mask")
                output_masks.append(converted.mask)
            outputs.append(output)
        result = geometry.concatenate(outputs, axis=-1)
        if action_mask is not None:
            mask = geometry.bool_array(
                geometry.concatenate(output_masks, axis=-1),
                result,
            )
            result = mask_conv.apply(result, mask)
        else:
            mask = geometry.bool_array(self.output_mask, result)
        if ignore_unused:
            static_mask = geometry.bool_array(self.output_mask, result)
            return ActionResult(action=result[..., static_mask], mask=None)
        return ActionResult(action=result, mask=mask)

    def _split_action_mask(
        self,
        action: ActionInput,
        action_mask: Array | None,
        *,
        space: str,
    ) -> dict[str, Array]:
        if action_mask is None:
            return {}
        if not isinstance(action, (np.ndarray, torch.Tensor)):
            raise TypeError("action_mask requires action to be one full-layout array")
        if action.shape != action_mask.shape:
            raise ValueError(
                f"action_mask shape {action_mask.shape} must match action "
                f"shape {action.shape}"
            )
        full_dim = getattr(self, f"{space}_dim")
        if action.shape[-1] != full_dim:
            raise ValueError(
                f"action_mask requires the full {space} layout dimension "
                f"{full_dim}, got {action.shape[-1]}"
            )
        parts = self._split_array(
            action_mask,
            space=space,
            label="action_mask",
        )
        return {
            name: value.astype(bool) if isinstance(value, np.ndarray) else value.bool()
            for name, value in parts.items()
        }

    def _validate_frame(
        self,
        ref_frame: Array | None,
        prototype: Array,
    ) -> Array | None:
        from morph.convs import geometry

        if not self.use_ref_frame:
            return None
        if ref_frame is None:
            raise ValueError(
                f"action profile {self.name!r} requires ref_frame because "
                "use_ref_frame=True"
            )
        geometry.validate_same_backend((prototype, ref_frame))
        expected_shape = prototype.shape[:-2] + (4, 4)
        if ref_frame.shape != expected_shape:
            raise ValueError(
                "ref_frame must align with the action batch dimensions: "
                f"expected {expected_shape} for action shape {prototype.shape}, "
                f"got {ref_frame.shape}"
            )
        return ref_frame

    @staticmethod
    def _validate_reference_shapes(
        refs: Mapping[str, Array],
        prototype: Array,
    ) -> None:
        batch_shape = prototype.shape[:-2]
        for name, reference in refs.items():
            expected_shape = batch_shape + (reference.shape[-1],)
            if reference.shape != expected_shape:
                raise ValueError(
                    f"ref_state field {name!r} must align with the action "
                    f"batch dimensions: expected {expected_shape} for action "
                    f"shape {prototype.shape}, got {reference.shape}"
                )

    def _split(
        self,
        value: ActionInput,
        *,
        space: str,
        label: str,
    ) -> dict[str, Array]:
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return self._split_array(value, space=space, label=label)
        if isinstance(value, Mapping):
            return self._split_mapping(value, space=space, label=label)
        raise TypeError(f"{label} must be a NumPy array, tensor, or mapping of arrays")

    def _split_array(
        self,
        value: Array,
        *,
        space: str,
        label: str,
    ) -> dict[str, Array]:
        if value.ndim == 0:
            raise ValueError(f"{label} must have an action dimension")
        full_dim = getattr(self, f"{space}_dim")
        used_dim = getattr(self, f"used_{space}_dim")
        if value.shape[-1] == full_dim:
            include_unused = True
        elif value.shape[-1] == used_dim:
            include_unused = False
        else:
            raise ValueError(
                f"{label} dim {value.shape[-1]} does not match full {space} "
                f"dim {full_dim} or used {space} dim {used_dim}"
            )
        parts = {}
        offset = 0
        for item in self.fields:
            dimension = getattr(item, f"{space}_dim")
            used = getattr(item, f"{space}_used")
            if include_unused or used:
                parts[item.name] = value[..., offset : offset + dimension]
                offset += dimension
        return parts

    def _split_mapping(
        self,
        value: Mapping[str, Array],
        *,
        space: str,
        label: str,
    ) -> dict[str, Array]:
        from morph.convs import geometry

        parts = {}
        leading_shape = None
        arrays = []
        for item in self.fields:
            if item.name not in value:
                if getattr(item, f"{space}_used"):
                    raise KeyError(f"{label} is missing used field {item.name!r}")
                continue
            part = value[item.name]
            if not isinstance(part, (np.ndarray, torch.Tensor)):
                raise TypeError(
                    f"{label} field {item.name!r} must be an array or tensor"
                )
            dimension = getattr(item, f"{space}_dim")
            if part.ndim == 0 or part.shape[-1] != dimension:
                raise ValueError(
                    f"{label} field {item.name!r} must end with dimension "
                    f"{dimension}, got {part.shape}"
                )
            if leading_shape is None:
                leading_shape = part.shape[:-1]
            elif part.shape[:-1] != leading_shape:
                raise ValueError(
                    f"{label} fields must share leading shape; got "
                    f"{leading_shape} and {part.shape[:-1]}"
                )
            arrays.append(part)
            parts[item.name] = part
        geometry.validate_same_backend(arrays)
        return parts

    def _split_reference(
        self,
        value: ActionInput | None,
    ) -> dict[str, Array]:
        from morph.convs import geometry

        if value is None:
            return {}
        if isinstance(value, Mapping):
            fields = {item.name: item for item in self.fields}
            parts = {}
            arrays = []
            for name, part in value.items():
                if name not in fields:
                    raise KeyError(f"ref_state contains unknown field {name!r}")
                if not isinstance(part, (np.ndarray, torch.Tensor)):
                    raise TypeError(
                        f"ref_state field {name!r} must be an array or tensor"
                    )
                expected = fields[name].input_dim
                if part.ndim == 0 or part.shape[-1] != expected:
                    raise ValueError(
                        f"ref_state field {name!r} must end with dimension "
                        f"{expected}, got {part.shape}"
                    )
                arrays.append(part)
                parts[name] = part
            geometry.validate_same_backend(arrays)
            return parts
        return self._split(value, space="input", label="ref_state")


def _prototype(parts: Mapping[str, Array], label: str) -> Array:
    try:
        return next(iter(parts.values()))
    except StopIteration as error:
        raise ValueError(f"{label} contains no array fields") from error


def canonical55_v1(
    *,
    qpos_model_repr: ModelRepr = "absolute",
    eef_model_repr: ModelRepr = "absolute",
    input_rot: RotRepr = "rot6d",
    output_rot: RotRepr = "rot6d",
    normalization: NormalizationConfig | None = None,
    use_ref_frame: bool = False,
) -> ActionProfile:
    """Build the shared training-time version-one Canonical55 profile."""
    return ActionProfile(
        name="canonical55",
        version="v1",
        use_ref_frame=use_ref_frame,
        normalization=normalization,
        fields=(
            Qpos(name="left_qpos", dim=7, model_repr=qpos_model_repr),
            EEF(
                name="left_eef",
                model_repr=eef_model_repr,
                input_rot=input_rot,
                output_rot=output_rot,
            ),
            Gripper(name="left_gripper"),
            Qpos(name="right_qpos", dim=7, model_repr=qpos_model_repr),
            EEF(
                name="right_eef",
                model_repr=eef_model_repr,
                input_rot=input_rot,
                output_rot=output_rot,
            ),
            Gripper(name="right_gripper"),
            Qpos(name="torso_qpos", dim=7, model_repr=qpos_model_repr),
            EEF(
                name="torso_eef",
                model_repr=eef_model_repr,
                input_rot=input_rot,
                output_rot=output_rot,
            ),
            Qpos(name="head_qpos", dim=2, model_repr=qpos_model_repr),
            Chassis(name="chassis"),
        ),
    )
