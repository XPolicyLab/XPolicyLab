from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from worldscape_policy.checkpoint.transforms import (
    TRANSFORM_BUNDLE_FILENAME,
    CheckpointTransformArtifact,
    NativeCheckpointTransform,
)
from worldscape_policy.embodiment import (
    bundle_key_aliases,
    canonical_embodiment,
    resolve_bundle_embodiment_key,
)


@dataclass(frozen=True)
class CheckpointTransformBundle:
    """Evaluation transforms and IDs recovered from a saved checkpoint."""

    transform: Any
    embodiment: str
    embodiment_id: int
    max_state_dim: int
    max_action_dim: int

    @property
    def embodiment_tag(self) -> str:
        """Deprecated alias for :attr:`embodiment`."""

        return self.embodiment


def load_checkpoint_transform_bundle(
    checkpoint_dir: str | Path,
    embodiment: str,
) -> CheckpointTransformBundle:
    """Load the required WSP-owned transform artifact."""

    directory = Path(checkpoint_dir)
    artifact_path = directory / TRANSFORM_BUNDLE_FILENAME
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint is missing required {artifact_path}; native checkpoints "
            "must include transform_bundle.json before evaluation"
        )
    artifact = CheckpointTransformArtifact.read(artifact_path)
    try:
        bundle_key = resolve_bundle_embodiment_key(artifact.embodiments, embodiment)
    except KeyError as error:
        raise KeyError(str(error)) from error
    transform_embodiment = artifact.embodiments[bundle_key]
    canonical = canonical_embodiment(embodiment)
    transform = NativeCheckpointTransform(
        image_input_range=artifact.image_input_range,
        embodiment=transform_embodiment,
    )
    return CheckpointTransformBundle(
        transform=transform,
        embodiment=canonical,
        embodiment_id=transform_embodiment.embodiment_id,
        max_state_dim=transform_embodiment.max_state_dim,
        max_action_dim=transform_embodiment.max_action_dim,
    )


def without_state_action_normalization(
    bundle: CheckpointTransformBundle,
) -> CheckpointTransformBundle:
    """Keep field layout/relative semantics while making numeric transforms identity."""

    embodiment = bundle.transform.embodiment

    def identity_field(field: Any) -> Any:
        return replace(
            field,
            normalization=None,
            statistics={},
            per_horizon_statistics=None,
        )

    identity_embodiment = replace(
        embodiment,
        state_fields=tuple(identity_field(field) for field in embodiment.state_fields),
        action_fields=tuple(identity_field(field) for field in embodiment.action_fields),
    )
    return replace(
        bundle,
        transform=NativeCheckpointTransform(
            image_input_range=bundle.transform.image_input_range,
            embodiment=identity_embodiment,
        ),
    )


def resolve_embodiment_id(source: Any, embodiment: str) -> int:
    """Resolve an embodiment ID from a transform or plain mapping."""

    mappings: list[Mapping[str, Any]] = []
    if isinstance(source, Mapping):
        mappings.append(source)
    direct = getattr(source, "embodiment_tag_mapping", None)
    if isinstance(direct, Mapping):
        mappings.append(direct)
    for transform in getattr(source, "transforms", ()):
        mapping = getattr(transform, "embodiment_tag_mapping", None)
        if isinstance(mapping, Mapping):
            mappings.append(mapping)

    canonical = canonical_embodiment(embodiment)
    lookup_keys = (embodiment, canonical, *bundle_key_aliases(canonical))
    for mapping in mappings:
        for key in lookup_keys:
            if key in mapping:
                value = mapping[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"Embodiment ID for {key!r} must be an integer"
                    )
                if value < 0:
                    raise ValueError("Embodiment ID must be non-negative")
                return value
    raise KeyError(
        f"Checkpoint transform does not map embodiment {embodiment!r}"
    )
