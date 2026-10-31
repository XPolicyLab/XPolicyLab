"""RoboDojo LeRobot v3 reader: X5 kinematics against simulator records, and parity with the HDF5 reader."""

from __future__ import annotations

import io
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset
from openwam.dataloader.robodojo_lerobot import (
    LEROBOT_CAMERAS,
    STATE_NAMES,
    MultiTaskRoboDojoLeRobotDataset,
    load_lerobot_episodes,
    recorded_ee_poses,
)

# Round-1 RSI episode stack_bowls-L0/episode_000001.hdf5, frames 172 and 345: state/<arm>_arm_joint_states and
# state/<arm>_ee_poses as Isaac recorded them.
SIM_RECORDS = [
    (
        "left",
        [
            -0.9745814800262451,
            1.6760358810424805,
            1.2881598472595215,
            -0.8106520771980286,
            -0.42376601696014404,
            0.6831588745117188,
        ],
        [
            -0.03343534469604492,
            -0.2552107572555542,
            0.9995235204696655,
            0.6952815651893616,
            0.19553989171981812,
            0.6645329594612122,
            0.19168609380722046,
        ],
    ),
    (
        "right",
        [
            0.3337162435054779,
            1.5431194305419922,
            0.8692522644996643,
            -0.14738580584526062,
            -0.019967198371887207,
            1.5804208517074585,
        ],
        [
            0.19994860887527466,
            -0.16191577911376953,
            0.9494948983192444,
            0.5982161164283752,
            0.1450892835855484,
            0.699665904045105,
            0.36270377039909363,
        ],
    ),
    (
        "right",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.3004709482192993,
            -0.35229986906051636,
            0.921500027179718,
            0.7069999575614929,
            -2.8373033700290762e-08,
            -4.5104908252824316e-08,
            0.7072135210037231,
        ],
    ),
]
FPS = 25
HEIGHT, WIDTH = 48, 64
UNIFY = ["0-9", "34-43"]


@pytest.mark.parametrize("arm, joints, recorded", SIM_RECORDS)
def test_rebuilt_ee_pose_matches_the_simulator(arm, joints, recorded):
    pose = recorded_ee_poses(np.asarray([joints]), arm)[0]
    np.testing.assert_allclose(pose[:3], recorded[:3], atol=1e-6)
    quaternion = pose[3:] if np.dot(pose[3:], recorded[3:]) > 0 else -pose[3:]
    np.testing.assert_allclose(quaternion, recorded[3:], atol=2e-6)


def _episode(seed: int, length: int):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)[:, None]
    arms = {
        arm: (rng.uniform(-0.6, 0.6, 6) * t + rng.uniform(-0.3, 0.3, 6)).astype(np.float32) for arm in ("left", "right")
    }
    grippers = {arm: np.clip(rng.uniform(0, 1, (length, 1)), 0, 1).astype(np.float32) for arm in ("left", "right")}
    shades = (np.linspace(30, 220, length)[:, None] + 7 * np.arange(len(LEROBOT_CAMERAS))[None, :] + seed * 11) % 256
    return arms, grippers, shades.astype(np.uint8)


def _frame(shade: int) -> np.ndarray:
    """Red and blue equal, so the legacy channel order the HDF5 decoder assumes for unmarked JPEGs changes nothing."""
    image = np.full((HEIGHT, WIDTH, 3), shade, dtype=np.uint8)
    image[..., 1] = 255 - shade
    return image


def _jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _write_hdf5(path: Path, arms, grippers, shades, instruction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for arm in ("left", "right"):
            f[f"state/{arm}_arm_joint_states"] = arms[arm]
            f[f"state/{arm}_ee_poses"] = recorded_ee_poses(arms[arm], arm).astype(np.float32)
            f[f"state/{arm}_ee_joint_states"] = grippers[arm]
        for c, camera in enumerate(LEROBOT_CAMERAS):
            f[f"vision/{camera}/colors"] = np.asarray([_jpeg(_frame(s)) for s in shades[:, c]])
        f["instruction"] = instruction


def _write_lerobot(root: Path, episodes, drop_last_frame_of: int | None = None) -> None:
    state = np.concatenate(
        [np.concatenate([a["left"], g["left"], a["right"], g["right"]], axis=1) for a, g, *_ in episodes]
    )
    lengths = [len(e[0]["left"]) for e in episodes]
    starts = np.cumsum([0, *lengths])[:-1]
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [list(STATE_NAMES)]},
        **{
            key: {"dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channels"]}
            for key in LEROBOT_CAMERAS.values()
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": FPS, "features": features}))
    rows = {
        "index": np.arange(len(state)),
        "episode_index": np.repeat(np.arange(len(episodes)), lengths),
        "frame_index": np.concatenate([np.arange(n) for n in lengths]),
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 14),
    }
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    meta = {
        "episode_index": np.arange(len(episodes)),
        "tasks": [[e[3]] for e in episodes],
        "length": lengths,
        "dataset_from_index": starts,
        "dataset_to_index": starts + lengths,
        "data/chunk_index": [0] * len(episodes),
        "data/file_index": [0] * len(episodes),
    }
    for c, key in enumerate(LEROBOT_CAMERAS.values()):
        video = root / "videos" / key / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        container = av.open(str(video), "w")
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"qscale": "1"}
        spans = []
        for e, (_, _, shades, *_) in enumerate(episodes):
            kept = shades[:-1] if e == drop_last_frame_of else shades
            spans.append(len(kept))
            for shade in kept[:, c]:
                for packet in stream.encode(av.VideoFrame.from_ndarray(_frame(int(shade)), format="rgb24")):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        first = np.cumsum([0, *spans])[:-1]
        meta[f"videos/{key}/chunk_index"] = [0] * len(episodes)
        meta[f"videos/{key}/file_index"] = [0] * len(episodes)
        meta[f"videos/{key}/from_timestamp"] = first / FPS
        meta[f"videos/{key}/to_timestamp"] = (first + spans) / FPS
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    sidecar = [{"episode_index": i, "task": e[4]} for i, e in enumerate(episodes)]
    (root / "rsi_episodes.json").write_text(json.dumps(sidecar))


@pytest.fixture()
def release(tmp_path):
    specs = [
        ("stack_bowls", 9, "Stack the bowls."),
        ("hang_mugs", 7, "Hang the mug."),
        ("stack_bowls", 12, "Stack the bowls."),
    ]
    episodes = []
    for i, (task, length, instruction) in enumerate(specs):
        arms, grippers, shades = _episode(i, length)
        _write_hdf5(
            tmp_path / "hdf5" / task / "arx_x5" / "data" / f"episode_{i:06d}.hdf5", arms, grippers, shades, instruction
        )
        episodes.append((arms, grippers, shades, instruction, task))
    _write_lerobot(tmp_path / "lerobot", episodes)
    return tmp_path, episodes


def _readers(root: Path, **kwargs):
    common = dict(normalize_mode=None, unify_action_map=UNIFY, split="val", num_frames=5, video_stride=2, **kwargs)
    return MultiTaskRoboDojoDataset(root / "hdf5", **common), MultiTaskRoboDojoLeRobotDataset(
        [root / "lerobot"], **common
    )


def test_samples_match_the_hdf5_reader(release):
    root, _ = release
    h5, lr = _readers(root)
    assert lr.tasks == h5.tasks == ["hang_mugs", "stack_bowls"] and len(lr) == len(h5)
    assert [d._window_index for d in lr._sub_datasets] == [d._window_index for d in h5._sub_datasets]
    for i in range(len(h5)):
        x, y = h5[i], lr[i]
        for key in ("prompt", "start_frame", "end_frame", "episode_length", "task_name"):
            assert x[key] == y[key], key
        for key in ("action_mask", "video_mask", "proprio_mask"):
            assert bool((x[key] == y[key]).all()), key
        for key in ("action", "proprio"):
            np.testing.assert_allclose(y[key].numpy(), x[key].numpy(), atol=1e-6)
        for a, b in zip(x["video"], y["video"]):
            assert np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean() < 3.0


def test_normalization_needs_explicit_stats(release):
    root, _ = release
    with pytest.raises(ValueError, match="normalization_stats_path"):
        MultiTaskRoboDojoLeRobotDataset(root / "lerobot", unify_action_map=UNIFY)


def test_dropped_video_frames_are_rejected(tmp_path, release):
    _, episodes = release
    _write_lerobot(tmp_path / "dropped", episodes, drop_last_frame_of=1)
    with pytest.raises(ValueError, match="dropped"):
        load_lerobot_episodes(tmp_path / "dropped")
