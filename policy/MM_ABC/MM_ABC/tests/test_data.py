"""Mobile data path: key layout, rotations, converted profiles, batching."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mmabc.canonical import rotation as rot  # noqa: E402
from mmabc.canonical.layout import load_layout  # noqa: E402
from mmabc.embodiments import mobile  # noqa: E402

MIXTURE = REPO / "configs/mixture/mobile_all.yaml"
LAYOUT = REPO / "configs/canonical/canonical_mobile.yaml"


def _random_raw(rng, lead=(5,)):
    raw = {}
    for key, dim, kind in mobile.KEYS:
        value = rng.normal(0, 0.5, lead + (dim,))
        if kind == "pose":
            q = rng.normal(size=lead + (4,))
            value[..., 3:] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        raw[key] = value
    return raw


def test_quaternion_matrix_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1000, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    back = rot.matrix_to_quat_wxyz(rot.quat_wxyz_to_matrix(q))
    same = np.minimum(np.abs(back - q).max(-1), np.abs(back + q).max(-1))
    assert same.max() < 1e-9
    assert (back[:, 0] >= 0).all()


def test_pack_unpack_round_trip_both_key_styles():
    rng = np.random.default_rng(1)
    raw = _random_raw(rng)
    vec = mobile.pack(raw)
    assert vec.shape == (5, mobile.MODEL_DIM) == (5, 75)
    singular = {mobile.singular(k): v for k, v in raw.items()}
    np.testing.assert_allclose(mobile.pack(singular), vec)
    back = mobile.unpack(vec, key_style="plural")
    for key, dim, kind in mobile.KEYS:
        ref, got = raw[key], back[key].astype(np.float64)
        if kind == "pose":
            sign = np.sign((ref[..., 3:] * got[..., 3:]).sum(-1, keepdims=True))
            got = np.concatenate([got[..., :3], got[..., 3:] * sign], -1)
        np.testing.assert_allclose(got, ref, atol=1e-5)


def test_pack_rejects_missing_and_wrong_dims_unless_allowed():
    rng = np.random.default_rng(2)
    raw = _random_raw(rng, lead=())
    del raw["waist_joint_states"]
    with pytest.raises(KeyError):
        mobile.pack(raw)
    filled = mobile.pack(raw, allow_missing=True)
    assert filled.shape == (75,)
    raw["waist_joint_states"] = np.zeros(3)
    with pytest.raises(ValueError):
        mobile.pack(raw)


def test_legacy_aliases_are_accepted():
    rng = np.random.default_rng(3)
    raw = {mobile.singular(k): v for k, v in _random_raw(rng, lead=()).items()}
    raw["base_head_pose"] = raw.pop("head_base_pose")
    assert mobile.pack(raw).shape == (75,)


def test_canonical_yaml_matches_key_layout():
    layout = load_layout(str(LAYOUT))
    assert layout.total_dim == mobile.MODEL_DIM
    starts = {s.start for s in layout.segments}
    for slot in mobile.SLOTS:
        assert slot.start in starts, slot
    heads = {h.name: (h.start, h.end) for h in layout.heads}
    assert heads == {"manip": (0, 56), "aux": (56, 75)}


@pytest.mark.skipif(not MIXTURE.exists(), reason="run scripts/convert_mobile.py first")
def test_mixture_samples_have_fixed_shapes():
    from mmabc.data import MixtureDataset, collate

    ds = MixtureDataset(MIXTURE, chunk_size=32, image_size=224, future_offsets=(32, 32),
                        norm_stats_dir=REPO / "configs/norm_stats", seed=0)
    if not all((Path(profile.path) / "meta/info.json").exists() for profile in ds.profile_configs):
        pytest.skip("converted training data is not available")
    it = iter(ds)
    samples = [next(it) for _ in range(4)]
    batch = collate(samples)
    assert batch["target"].shape == (4, 32, 75)
    assert batch["state"].shape == (4, 75)
    assert batch["images"].shape == (4, 3, 3, 224, 224, 3)
    assert float(batch["view_mask"].min()) == 1.0
    # min-max normalised absolute commands stay inside [-1, 1] up to float error.
    t = batch["target"][batch["target_mask"] > 0]
    assert float(t.abs().max()) <= 1.0 + 1e-4
    assert all(p.startswith("task: ") for p in batch["prompt"])
    assert ds.skipped == 0


def test_adapter_round_trip_matches_pack():
    from mmabc.eval.adapters.mobile import MobileAdapter

    rng = np.random.default_rng(4)
    raw = {mobile.singular(k): v for k, v in _random_raw(rng, lead=()).items()}
    obs = {
        "state": raw,
        "vision": {cam: {"color": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)}
                   for cam in mobile.CAMERAS.values()},
    }
    adapter = MobileAdapter()
    state = adapter.to_canonical_state(obs)
    np.testing.assert_allclose(state, mobile.pack(raw))
    images = adapter.to_canonical_images(obs)
    assert set(images) == set(mobile.CAMERAS)
    actions = adapter.from_canonical_action(np.repeat(state[None], 3, 0))
    assert len(actions) == 3 and set(actions[0]) == set(raw)
    random.seed(0)


def test_c80_round_trip_and_increment_targets_invert():
    import json

    from mmabc.canonical.transforms import EmbodimentSpec, TargetBuilder

    rng = np.random.default_rng(5)
    raw = _random_raw(rng, lead=(33,))
    raw["root_linear_velocity"][..., 2] = 0.0   # vz, wx, wy are not carried in c80
    raw["root_angular_velocity"][..., :2] = 0.0
    m75 = mobile.pack(raw)
    c80 = mobile.to_c80(m75)
    np.testing.assert_allclose(mobile.from_c80(c80), m75, atol=1e-9)

    layout = load_layout(str(REPO / "configs/canonical/canonical_80_mobile.yaml"))
    emb = {"canonical": {"embodiment_tag": mobile.C80_TAG, "total_dim": 80,
                         "action_valid_dims": mobile.C80_VALID, "state_valid_dims": mobile.C80_VALID}}
    builder = TargetBuilder(layout, EmbodimentSpec.from_meta(json.loads(json.dumps(emb)), fps=30.0))
    state, actions = c80[0], c80[1:]
    target, mask = builder.build(state, actions, action_type="joint")
    assert mask[:, mobile.C80_VALID].any(0).sum() == 64  # rot6d trailing dims are not targets
    back = builder.integrate(state, target, action_type="joint")
    np.testing.assert_allclose(mobile.from_c80(back), m75[1:], atol=1e-5)
