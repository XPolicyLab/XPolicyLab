import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from PIL import Image

    from eventvla.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
except Exception as exc:  # pragma: no cover - only used when optional deps are absent.
    Image = None
    LeRobotMixtureDataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeDataset:
    dataset_name = "fake_robotwin"
    tag = "robotwin_mem"
    modality_keys = {
        "video": [
            "video.cam_left_wrist",
            "video.cam_high",
            "video.cam_right_wrist",
        ],
    }
    absolute_indices = {
        "video.cam_high": np.array([0], dtype=int),
    }
    delta_indices = {
        "video.cam_high": np.array([-30, -15, 0], dtype=int),
    }

    def __init__(self):
        self.frame_requests = []

    def get_video_frame(self, trajectory_id, video_key, frame_index):
        self.frame_requests.append((int(trajectory_id), video_key, int(frame_index)))
        return np.full((8, 6, 3), int(frame_index) % 255, dtype=np.uint8)

    def get_step_data(self, *args, **kwargs):
        raise AssertionError("keyframe memory should not build a full step sample")


def _make_mixture():
    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.data_cfg = {
        "keyframe_image_memory": {
            "enabled": True,
            "max_keyframes": 2,
            "include_current_keyframe": True,
            "order": "chronological",
            "selection": "latest",
            "view_mode": "include_names",
            "include_names": ["cam_high", "head", "main"],
            "exclude_name_patterns": ["wrist"],
            "strict_single_view": True,
        }
    }
    return mixture


@unittest.skipIf(IMPORT_ERROR is not None, f"missing dataset test dependencies: {IMPORT_ERROR}")
class KeyframeMemoryLightweightFetchTest(unittest.TestCase):
    def test_visible_memory_fetch_decodes_only_selected_video_key(self):
        mixture = _make_mixture()
        dataset = _FakeDataset()

        images, metas, steps = mixture._build_visible_keyframe_image_memory(
            dataset=dataset,
            trajectory_id=7,
            step=10,
            keyframe_steps=[1, 5, 10],
        )

        self.assertEqual(dataset.frame_requests, [(7, "video.cam_high", 5), (7, "video.cam_high", 10)])
        self.assertEqual(steps, [5, 10])
        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(image, Image.Image) for image in images))
        self.assertTrue(all(image.size == (224, 224) for image in images))
        self.assertEqual([meta["video_key"] for meta in metas], ["video.cam_high", "video.cam_high"])
        self.assertEqual([meta["source_timestep"] for meta in metas], [5, 10])

    def test_runtime_exact_fetch_uses_same_lightweight_reader(self):
        mixture = _make_mixture()
        dataset = _FakeDataset()
        mixture.datasets = [dataset]

        fetched = mixture.get_memory_image_at_step(
            {
                "dataset_index": 0,
                "trajectory_id": 7,
                "target_step": 42,
                "slot_idx": 1,
                "episode_id": "fake_robotwin::7",
            }
        )

        self.assertIsNotNone(fetched)
        self.assertEqual(dataset.frame_requests, [(7, "video.cam_high", 42)])
        self.assertEqual(fetched["target_step"], 42)
        self.assertEqual(fetched["image_metas"][0]["video_key"], "video.cam_high")
        self.assertEqual(fetched["images"][0].size, (224, 224))


if __name__ == "__main__":
    unittest.main()
