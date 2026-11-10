# coding=utf-8
"""Configuration and lifecycle shared by the execution skills."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .state import json_value

SKILL_NAMES = ("pi05", "code-policy")
ARX_FIELDS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)


@dataclass(frozen=True)
class CodeSkillConfig:
    """One task controller inside code-policy, not a top-level policy."""

    description: str
    code_path: Path
    code_bundle: Path
    code_entrypoint: str = "create_policy"
    code_sha256: str | None = None
    code_parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Each code sub-skill requires a capability description")
        for name in ("code_path", "code_bundle"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"Sub-skill {name} must be a nonempty path")
            object.__setattr__(self, name, Path(value).expanduser().resolve())
        if not isinstance(self.code_parameters, dict):
            raise ValueError("Sub-skill code_parameters must be a JSON object")
        object.__setattr__(self, "code_parameters", json_value(self.code_parameters))

    def policy_config(self, parent):
        parameters = {
            **deepcopy(parent.code_parameters),
            **deepcopy(self.code_parameters),
        }
        return replace(
            parent,
            code_skills={},
            code_selection_prompt=None,
            code_path=self.code_path,
            code_bundle=self.code_bundle,
            code_entrypoint=self.code_entrypoint,
            code_sha256=self.code_sha256,
            code_parameters=parameters,
        )

    def public(self):
        result = asdict(self)
        for name in ("code_path", "code_bundle"):
            result[name] = str(result[name])
        return result


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "pi05"
    checkpoint: Path | None = None
    train_config_name: str | None = None
    normalization_id: str | None = None
    camera_map: dict[str, str] = field(
        default_factory=lambda: {
            "cam_high": "cam_high",
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
        }
    )
    state_fields: tuple[str, ...] = ARX_FIELDS
    state_dims: tuple[int, ...] = (6, 1, 6, 1)
    action_fields: tuple[str, ...] | None = None
    action_dims: tuple[int, ...] | None = None
    code_path: Path | None = None
    code_bundle: Path | None = None
    code_skills: dict[str, CodeSkillConfig] = field(default_factory=dict)
    code_selection_prompt: Path | None = None
    code_entrypoint: str = "create_policy"
    code_sha256: str | None = None
    code_parameters: dict = field(default_factory=dict)
    tokenizer_path: Path | None = None

    def __post_init__(self) -> None:
        if self.name not in SKILL_NAMES:
            raise ValueError(f"Unknown skill: {self.name}; choose one of {SKILL_NAMES}")
        for name in (
            "checkpoint",
            "code_path",
            "code_bundle",
            "code_selection_prompt",
            "tokenizer_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.code_bundle is not None and self.name != "code-policy":
            raise ValueError("code_bundle is only supported by code-policy")
        if not isinstance(self.code_skills, dict):
            raise ValueError("code_skills must be a named sub-skill mapping")
        children = {}
        for key, value in self.code_skills.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError("Code sub-skill IDs must be identifiers")
            if isinstance(value, dict):
                value = CodeSkillConfig(**value)
            if not isinstance(value, CodeSkillConfig):
                raise ValueError("Invalid code sub-skill configuration")
            children[key] = deepcopy(value)
        object.__setattr__(self, "code_skills", children)
        if children and (
            self.name != "code-policy"
            or self.code_path is not None
            or self.code_bundle is not None
            or self.code_selection_prompt is None
        ):
            raise ValueError(
                "Nested code_skills require code-policy, code_selection_prompt and no top-level code_path/code_bundle"
            )
        for name in ("train_config_name", "normalization_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or null")
        if self.normalization_id is not None:
            value = Path(self.normalization_id)
            if value.is_absolute() or ".." in value.parts or len(value.parts) != 1:
                raise ValueError(
                    "normalization_id must be a single asset directory name"
                )
        for prefix in ("state", "action"):
            fields = getattr(self, prefix + "_fields")
            dims = getattr(self, prefix + "_dims")
            if prefix == "action":
                fields = self.state_fields if fields is None else fields
                dims = self.state_dims if dims is None else dims
            if (
                not isinstance(fields, (list, tuple))
                or not fields
                or any(not isinstance(key, str) or not key for key in fields)
                or len(fields) != len(set(fields))
            ):
                raise ValueError(f"{prefix}_fields must be distinct nonempty names")
            if (
                not isinstance(dims, (list, tuple))
                or len(dims) != len(fields)
                or any(type(size) is not int or size < 1 for size in dims)
            ):
                raise ValueError(f"{prefix}_dims must match the named fields")
            object.__setattr__(self, prefix + "_fields", tuple(fields))
            object.__setattr__(self, prefix + "_dims", tuple(dims))
        if (
            not isinstance(self.camera_map, dict)
            or not self.camera_map
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in self.camera_map.items()
            )
        ):
            raise ValueError(
                "camera_map must map model camera names to observed camera names"
            )
        object.__setattr__(self, "camera_map", deepcopy(self.camera_map))
        if (
            not isinstance(self.code_entrypoint, str)
            or not self.code_entrypoint.isidentifier()
        ):
            raise ValueError("code_entrypoint must be a Python factory name")
        if self.code_sha256 is not None and (
            not isinstance(self.code_sha256, str)
            or len(self.code_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.code_sha256)
        ):
            raise ValueError("code_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.code_parameters, dict):
            raise ValueError("code_parameters must be a JSON object")
        object.__setattr__(self, "code_parameters", json_value(self.code_parameters))
        for child in children.values():
            child.policy_config(self)  # Validate the inherited action/code contract.


def create_policy(config: PolicyConfig):
    from .skills.code_policy import CodePolicy
    from .skills.pi05 import Pi05Policy

    classes = {
        "pi05": Pi05Policy,
        "code-policy": CodePolicy,
    }
    return classes[config.name](config)


# Preserve the original import path while keeping optional model imports lazy.
from .skills.pi05 import Pi05Policy  # noqa: E402,F401
