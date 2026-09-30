"""Dataset action metadata and shared prompt rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def _transform_name(transform: Any) -> str:
    target = _get(transform, "_target_")
    return str(target).rsplit(".", 1)[-1] if target else type(transform).__name__


def _transform_keys(transform: Any) -> set[str]:
    keys = _get(transform, "keys")
    if keys is None:
        return set()
    if isinstance(keys, (str, bytes)):
        return {str(keys)}
    return {str(key) for key in keys}


def _is_gripper(key: str) -> bool:
    lowered = key.lower()
    return "gripper" in lowered or "parallel_jaw" in lowered


def _is_joint(key: str) -> bool:
    lowered = key.lower()
    return (
        "joint" in lowered
        or lowered in {"left_arm", "right_arm", "arm"}
        or lowered.endswith("_arm")
    )


def _is_eef(key: str) -> bool:
    lowered = key.lower()
    return any(
        token in lowered
        for token in ("eef", "end_effector", "ee_pose", "flange")
    )


@dataclass(frozen=True)
class Metadata:
    """Language-visible semantics of one dataset's action target."""

    control_mode: str
    action_representation: str

    def __post_init__(self) -> None:
        for name in ("control_mode", "action_representation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @classmethod
    def from_config(
        cls,
        *,
        shape_meta: Any,
        action_state_transforms: Optional[Sequence[Any]] = None,
    ) -> "Metadata":
        """Infer semantics from action keys and configured transforms.

        Unknown and partially transformed layouts fail closed: emitting a
        misleading language condition is worse than omitting metadata.
        """

        action_meta = _get(shape_meta, "action")
        if not action_meta:
            raise ValueError("shape_meta.action is required to infer metadata")
        action_keys = tuple(str(_get(entry, "key")) for entry in action_meta)
        if any(key in {"", "None"} for key in action_keys):
            raise ValueError("every shape_meta.action entry must define a key")
        if len(set(action_keys)) != len(action_keys):
            raise ValueError(f"duplicate action keys are not allowed: {action_keys}")

        gripper_keys = {key for key in action_keys if _is_gripper(key)}
        control_keys = set(action_keys) - gripper_keys
        joint_keys = {key for key in control_keys if _is_joint(key)}
        eef_keys = {key for key in control_keys if _is_eef(key)}
        unknown = control_keys - joint_keys - eef_keys
        if not control_keys:
            raise ValueError("cannot infer a control mode from gripper-only actions")
        if unknown:
            raise ValueError(f"unknown action-key semantics: {sorted(unknown)}")
        if joint_keys and eef_keys:
            raise ValueError("mixed joint and end-effector control is not supported yet")

        relative_joint: set[str] = set()
        for transform in action_state_transforms or ():
            name, keys = _transform_name(transform), _transform_keys(transform)
            if name == "RelativeJointTransform":
                relative_joint.update(keys)

        transformed = relative_joint
        undeclared = transformed - set(action_keys)
        if undeclared:
            raise ValueError(
                "relative transform references undeclared action keys: "
                f"{sorted(undeclared)}"
            )
        if transformed & gripper_keys:
            raise ValueError("relative gripper targets are not supported yet")
        gripper_suffix = "; gripper targets absolute" if gripper_keys else ""

        native_delta = {
            str(_get(entry, "key")) for entry in action_meta
            if bool(_get(entry, "native_delta", False))
        }
        if native_delta:
            if native_delta != eef_keys or joint_keys or transformed:
                raise ValueError("native_delta requires all EEF controls and no relative transforms")
            return cls(
                "end effector",
                "per-step delta end-effector controller commands" + gripper_suffix,
            )

        if joint_keys:
            if relative_joint and relative_joint != joint_keys:
                raise ValueError("partially relative joint control is ambiguous")
            semantics = (
                "joint position targets relative to the current state"
                if relative_joint
                else "absolute joint position targets"
            )
            return cls("joint", semantics + gripper_suffix)

        if relative_joint:
            raise ValueError("RelativeJointTransform cannot target EEF actions")
        semantics = "absolute end-effector pose targets"
        return cls("end effector", semantics + gripper_suffix)


def metadata_from_dataset_config(dataset_config: Any) -> Optional[Metadata]:
    """Build metadata for one raw dataset config when explicitly enabled."""

    if not bool(_get(dataset_config, "metadata_enabled", False)):
        return None
    processor = _get(dataset_config, "processor")
    if processor is None:
        raise ValueError("metadata_enabled requires a processor config")
    shape_meta = _get(dataset_config, "shape_meta") or _get(processor, "shape_meta")
    return Metadata.from_config(
        shape_meta=shape_meta,
        action_state_transforms=_get(processor, "action_state_transforms"),
    )


def metadata_by_dataset_from_config(config: Any) -> dict[str, Metadata]:
    """Extract enabled per-dataset metadata from a full experiment config."""

    data = _get(config, "data", config)
    train = _get(data, "train")
    if train is None:
        return {}
    datasets = _get(train, "datasets")
    if datasets is None:
        sources = _get(train, "sources")
        datasets = list(sources.values()) if isinstance(sources, Mapping) else ()

    result: dict[str, Metadata] = {}
    for dataset_config in datasets or ():
        metadata = metadata_from_dataset_config(dataset_config)
        if metadata is None:
            continue
        name = _get(dataset_config, "dataset_name")
        if not name:
            raise ValueError("metadata-enabled dataset must define dataset_name")
        if str(name) in result:
            raise ValueError(f"duplicate metadata dataset_name: {name}")
        result[str(name)] = metadata
    return result


def render_instruction(instruction: str, metadata: Optional[Metadata] = None) -> str:
    """Render model text; passing no metadata preserves the old bytes."""

    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    if metadata is None:
        return instruction
    if not isinstance(metadata, Metadata):
        raise TypeError("metadata must be Metadata or None")
    return (
        f"Task: {instruction.strip()}\n"
        f"Control Mode: {metadata.control_mode}\n"
        f"Action Representation: {metadata.action_representation}"
    )


__all__ = [
    "Metadata",
    "metadata_by_dataset_from_config",
    "metadata_from_dataset_config",
    "render_instruction",
]
