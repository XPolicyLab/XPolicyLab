"""Canonical layout, rotation maths and target construction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mmabc.canonical import EmbodimentSpec, TargetBuilder, load_layout
from mmabc.canonical import rotation as rot

REPO = Path(__file__).resolve().parents[1]
LAYOUT = str(REPO / "configs/canonical/canonical_80.yaml")
# The MM-ABC pretraining corpus, only for the cross-embodiment checks below.
DATA = Path(os.environ.get("MMABC_CORPUS_ROOT", "/nonexistent/mmabc_corpus"))


@pytest.fixture(scope="module")
def layout():
    return load_layout(LAYOUT)


def test_layout_covers_every_dim_exactly_once(layout):
    covered = np.zeros(layout.total_dim, dtype=int)
    for seg in layout.segments:
        covered[seg.slice] += 1
    assert (covered == 1).all()


def test_heads_partition_the_vector(layout):
    assert layout.head("manip").start == 0
    assert layout.head("manip").end == layout.head("aux").start == 58
    assert layout.head("aux").end == layout.total_dim


def test_rotation_segments_expose_three_target_dims(layout):
    for seg in layout.segments_of_kind("eef_rot"):
        assert seg.width == 6
        assert seg.target_width == 3


def test_reserved_never_supervised(layout):
    reserved = layout.segment("reserved")
    assert not layout.target_support()[reserved.slice].any()


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 0.3, 1.5, 3.0, np.pi - 1e-4])
def test_axis_angle_roundtrip(angle):
    axis = np.array([[0.3, -0.5, 0.81]])
    axis = axis / np.linalg.norm(axis)
    vec = axis * angle
    back = rot.matrix_to_axis_angle(rot.axis_angle_to_matrix(vec))
    assert np.abs(back - vec).max() < 1e-8


@pytest.mark.parametrize("angle", [np.pi - 1e-7, np.pi])
def test_axis_angle_roundtrip_at_half_turn(angle):
    """At exactly pi the axis sign is ambiguous, so compare rotations."""
    rng = np.random.default_rng(0)
    axis = rng.normal(size=(64, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    m = rot.axis_angle_to_matrix(axis * angle)
    back = rot.axis_angle_to_matrix(rot.matrix_to_axis_angle(m))
    assert np.abs(back - m).max() < 1e-6


def test_rot6d_roundtrip():
    rng = np.random.default_rng(0)
    m = rot.axis_angle_to_matrix(rng.normal(size=(32, 3)))
    back = rot.rot6d_to_matrix(rot.matrix_to_rot6d(m))
    assert np.abs(back - m).max() < 1e-10


def test_rot6d_gram_schmidt_orthonormalises():
    """A non-orthogonal 6D input must still produce a valid rotation."""
    raw = np.array([[1.0, 0.1, 0.0, 0.2, 1.0, 0.0]])
    m = rot.rot6d_to_matrix(raw)[0]
    assert np.abs(m @ m.T - np.eye(3)).max() < 1e-10
    assert abs(np.linalg.det(m) - 1.0) < 1e-10


def _builder(profile: str, layout, frame="base"):
    meta = json.loads((DATA / profile / "meta" / "embodiment.json").read_text())
    spec = EmbodimentSpec.from_meta(meta)
    return TargetBuilder(layout, spec, reference_frame=frame), spec


@pytest.mark.skipif(not DATA.exists(), reason="corpus not mounted")
@pytest.mark.parametrize(
    "profile,expect_aux",
    [
        ("mobile_platform", True),
        ("single_arm", False),
        ("dual_arm", False),
    ],
)
def test_aux_activity_matches_platform(profile, expect_aux, layout):
    builder, _ = _builder(profile, layout)
    at = builder.available_action_types[0]
    assert builder.aux_is_active(at) is expect_aux


@pytest.mark.skipif(not DATA.exists(), reason="corpus not mounted")
@pytest.mark.parametrize("frame", ["base", "eef_local"])
def test_build_integrate_roundtrip(frame, layout):
    """integrate() must invert build() on every supervised dimension."""
    builder, _ = _builder("dual_arm", layout, frame)
    rng = np.random.default_rng(1)
    state = rng.normal(size=80) * 0.3
    actions = state[None] + rng.normal(size=(32, 80)) * 0.05
    for seg in layout.segments_of_kind("eef_rot"):
        state[seg.slice] = rot.matrix_to_rot6d(rot.axis_angle_to_matrix(rng.normal(size=3) * 0.5))
        actions[:, seg.slice] = rot.matrix_to_rot6d(
            rot.axis_angle_to_matrix(rng.normal(size=(32, 3)) * 0.5)
        )
    at = builder.available_action_types[0]
    target, mask = builder.build(state, actions, action_type=at)
    back = builder.integrate(state, target, action_type=at)
    assert np.abs(np.where(mask, back - actions, 0.0)).max() < 1e-5


@pytest.mark.skipif(not DATA.exists(), reason="corpus not mounted")
def test_action_type_selection_is_exclusive(layout):
    """Choosing joint control must mask end-effector dims, and vice versa."""
    builder, _ = _builder("single_arm", layout)
    assert set(builder.available_action_types) == {"joint", "eef"}
    joint_mask = builder.static_mask("joint")
    eef_mask = builder.static_mask("eef")
    arm = layout.segment("right_arm_joints")
    pos = layout.segment("right_eef_position")
    assert joint_mask[arm.slice].any() and not joint_mask[pos.slice].any()
    assert eef_mask[pos.slice].any() and not eef_mask[arm.slice].any()


@pytest.mark.skipif(not DATA.exists(), reason="corpus not mounted")
def test_padding_past_episode_end_is_masked(layout):
    builder, _ = _builder("single_arm", layout)
    at = builder.available_action_types[0]
    state = np.zeros(80)
    actions = np.zeros((32, 80))
    _, mask = builder.build(state, actions, action_type=at, valid_steps=10)
    assert mask[:10].any()
    assert not mask[10:].any()


@pytest.mark.skipif(not DATA.exists(), reason="corpus not mounted")
def test_delta_requires_both_endpoints(layout):
    """A dim with an action label but no state label cannot form an increment."""
    meta = json.loads(
        (DATA / "single_arm" / "meta" / "embodiment.json").read_text()
    )
    spec = EmbodimentSpec.from_meta(meta)
    crippled = EmbodimentSpec(
        embodiment_tag=spec.embodiment_tag,
        fps=spec.fps,
        state_valid=np.zeros_like(spec.state_valid),
        action_valid=spec.action_valid,
    )
    builder = TargetBuilder(layout, crippled)
    arm = layout.segment("right_arm_joints")
    assert not builder.static_mask("joint")[arm.slice].any()
