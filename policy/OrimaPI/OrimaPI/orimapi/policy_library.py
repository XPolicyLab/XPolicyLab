# coding=utf-8
"""Policy catalogue data. Runtime owns loading, episode binding, and cleanup."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from .policy import PolicyConfig, create_policy


@dataclass(frozen=True)
class PolicyCandidate:
    config: PolicyConfig
    description: str
    supports_subgoals: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, PolicyConfig):
            raise ValueError("A candidate requires a PolicyConfig")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Each policy candidate requires a capability description")
        supports = self.supports_subgoals
        if supports is None:
            supports = self.config.name != "code-policy"
        if type(supports) is not bool:
            raise ValueError("supports_subgoals must be a boolean")
        object.__setattr__(self, "supports_subgoals", supports)

    def capability(self) -> dict:
        """Selection metadata is available without importing a code program."""
        return {
            "skill": self.config.name,
            "description": self.description,
            "supports_subgoals": self.supports_subgoals,
            "action_fields": list(self.config.action_fields),
            "action_dims": list(self.config.action_dims),
        }

    def public(self) -> dict:
        settings = asdict(self.config)
        for name in (
            "checkpoint",
            "code_path",
            "code_bundle",
            "code_selection_prompt",
            "tokenizer_path",
        ):
            value = settings[name]
            settings[name] = str(value) if value is not None else None
        settings["code_skills"] = {
            key: value.public() for key, value in self.config.code_skills.items()
        }
        return {
            **settings,
            "description": self.description,
            "supports_subgoals": self.supports_subgoals,
        }


def policy_library(
    candidates: dict[str, PolicyCandidate], *, factory=create_policy
) -> dict:
    """Return a plain catalogue and its loader; this function owns no resources."""
    if (
        not isinstance(candidates, dict)
        or not candidates
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, PolicyCandidate)
            for key, value in candidates.items()
        )
    ):
        raise ValueError("policies must contain named policy candidates")
    return {"candidates": deepcopy(candidates), "factory": factory}


# Existing callers can keep constructing PolicyLibrary(...); it is now data.
PolicyLibrary = policy_library
