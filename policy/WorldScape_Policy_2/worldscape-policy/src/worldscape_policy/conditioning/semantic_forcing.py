"""Shared semantic-forcing target selection contracts.

Target selection is deliberately stateless: conditioner checkpoint ownership
and parameter names must not change when this policy is composed into Auto or
Interactive paths.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class SemanticTargetBuilder(Protocol):
    """Build the training-only teacher target used by semantic forcing."""

    def __call__(
        self,
        *,
        training: bool,
        explicit_target: Tensor | None,
        fallback_target: Tensor | None = None,
    ) -> Tensor | None: ...


class LegacySemanticTargetBuilder:
    """Preserve the legacy explicit-target-then-task-embedding fallback."""

    def __call__(
        self,
        *,
        training: bool,
        explicit_target: Tensor | None,
        fallback_target: Tensor | None = None,
    ) -> Tensor | None:
        if not training:
            return None
        return explicit_target if explicit_target is not None else fallback_target


LEGACY_SEMANTIC_TARGET_BUILDER: SemanticTargetBuilder = LegacySemanticTargetBuilder()


def build_semantic_target(
    *,
    training: bool,
    explicit_target: Tensor | None,
    fallback_target: Tensor | None = None,
) -> Tensor | None:
    """Use the legacy semantic-target policy without registering new state."""

    return LEGACY_SEMANTIC_TARGET_BUILDER(
        training=training,
        explicit_target=explicit_target,
        fallback_target=fallback_target,
    )


__all__ = [
    "LEGACY_SEMANTIC_TARGET_BUILDER",
    "LegacySemanticTargetBuilder",
    "SemanticTargetBuilder",
    "build_semantic_target",
]
