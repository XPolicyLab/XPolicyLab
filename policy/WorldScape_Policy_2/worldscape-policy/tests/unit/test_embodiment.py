import pytest

from evals.common.checkpoint_runtime import load_checkpoint_transform_bundle
from worldscape_policy.checkpoint.transforms import (
    CheckpointTransformArtifact,
    EmbodimentTransform,
    TransformField,
    TRANSFORM_BUNDLE_FILENAME,
)
from worldscape_policy.embodiment import (
    AGILEX,
    LIBERO,
    ROBOTWIN2,
    canonical_embodiment,
    canonical_embodiment_tag,
    coalesce_embodiment,
    expand_embodiment_ids,
    is_agilex_embodiment,
    resolve_bundle_embodiment_key,
)


def _field(key: str, size: int) -> TransformField:
    return TransformField(
        key=key,
        size=size,
        normalization=None,
        statistics={},
        per_horizon_statistics=None,
        absolute=True,
    )


def _agilex_embodiment() -> EmbodimentTransform:
    keys = (
        "state.left_pos",
        "state.left_rot6d",
        "state.left_gripper",
        "state.right_pos",
        "state.right_rot6d",
        "state.right_gripper",
    )
    sizes = (3, 6, 1, 3, 6, 1)
    state_fields = tuple(_field(key, size) for key, size in zip(keys, sizes, strict=True))
    action_fields = tuple(
        _field(key.replace("state.", "action."), size)
        for key, size in zip(keys, sizes, strict=True)
    )
    return EmbodimentTransform(
        embodiment_id=0,
        max_state_dim=64,
        max_action_dim=32,
        state_fields=state_fields,
        action_fields=action_fields,
    )


def test_canonical_embodiment_maps_legacy_agilex_names():
    assert canonical_embodiment("agilex") == AGILEX
    assert canonical_embodiment("lerobot_eef_lctx") == AGILEX
    assert canonical_embodiment("lerobot_eef") == AGILEX
    assert canonical_embodiment("eef") == AGILEX
    assert canonical_embodiment("libero") == LIBERO
    assert canonical_embodiment("robotwin") == ROBOTWIN2
    assert canonical_embodiment_tag("eef") == AGILEX


def test_coalesce_embodiment_prefers_canonical_key():
    assert coalesce_embodiment({"embodiment": "libero"}) == LIBERO
    assert coalesce_embodiment({"embodiment_tag": "robotwin"}) == ROBOTWIN2
    assert coalesce_embodiment({"robot_type": "agilex"}) == AGILEX


def test_expand_embodiment_ids_registers_legacy_aliases():
    expanded = expand_embodiment_ids({"agilex": 0})
    assert expanded["agilex"] == 0
    assert expanded["lerobot_eef_lctx"] == 0
    assert expanded["eef"] == 0


def test_resolve_bundle_embodiment_key_accepts_legacy_checkpoint_tags():
    embodiments = {"lerobot_eef_lctx": _agilex_embodiment()}
    assert resolve_bundle_embodiment_key(embodiments, "agilex") == "lerobot_eef_lctx"


def test_load_checkpoint_transform_bundle_returns_canonical_embodiment(tmp_path):
    artifact = CheckpointTransformArtifact(
        image_input_range="uint8",
        embodiments={"lerobot_eef_lctx": _agilex_embodiment()},
        provenance={"source_format": "unit-test"},
    )
    artifact.write(tmp_path / TRANSFORM_BUNDLE_FILENAME)
    bundle = load_checkpoint_transform_bundle(tmp_path, "agilex")
    assert bundle.embodiment == AGILEX
    assert bundle.embodiment_tag == AGILEX
    assert is_agilex_embodiment(bundle.embodiment)


def test_expand_embodiment_ids_rejects_conflicting_ids():
    with pytest.raises(ValueError, match="Conflicting"):
        expand_embodiment_ids({"agilex": 0, "lerobot_eef_lctx": 1})
