"""CPU interface, protocol and released-checkpoint regression checks."""

import json
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .. import model
from ..runtime.loader import activate_source, SOURCE


class FakeSession:
    def __init__(self):
        self.pending_model_actions = deque()
        self.acks = 0

    def get_action(self, request):
        if self.pending_model_actions:
            raise RuntimeError("pending actions")
        self.pending_model_actions.extend(range(3))
        return np.tile(np.arange(14, dtype=np.float32), (3, 1))

    def update_obs(self, obs):
        self.pending_model_actions.popleft()
        self.acks += 1

    def reset_model(self):
        self.pending_model_actions.clear()
        self.acks = 0

    def close(self):
        self.reset_model()


def observation(i=0):
    return dict(
        env_idx=i,
        instruction="put the block in the box",
        vision={
            k: {"color": np.zeros((48, 64, 3), np.uint8)}
            for k in ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        },
        state={
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.array([0.2]),
            "right_arm_joint_state": np.arange(6, dtype=np.float32) + 10,
            "right_ee_joint_state": np.array([0.8]),
        },
    )


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with (
            patch.object(
                model,
                "get_robot_action_dim_info",
                return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]},
            ),
            patch.object(model, "get_batch_size", return_value=2),
            patch.object(model, "build_runtime", return_value=(object(), FakeSession)),
        ):
            self.m = model.Model(
                dict(
                    action_type="joint", env_cfg_type="arx_x5", ckpt_name=self.tmp.name
                )
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_action_mapping_no_extra_clipping(self):
        self.m.update_obs(observation())
        a = self.m.get_action()
        self.assertEqual(len(a), 3)
        np.testing.assert_array_equal(a[0]["left_arm_joint_state"], np.arange(6))
        np.testing.assert_array_equal(a[0]["right_arm_joint_state"], np.arange(7, 13))
        np.testing.assert_array_equal(a[0]["right_ee_joint_state"], [13])

    def test_acknowledgment_cadence(self):
        self.m.update_obs(observation())
        self.m.get_action()
        with self.assertRaises(RuntimeError):
            self.m.get_action()
        for _ in range(3):
            self.m.update_obs(observation())
        self.assertEqual(self.m.single.acks, 3)
        self.m.get_action()

    def test_batch_shrink_and_reset(self):
        self.m.update_obs_batch([observation(4), observation(9)])
        self.m.get_action_batch([9, 4])
        self.m.update_obs_batch([observation(9)])
        self.assertEqual(self.m.sessions[9].acks, 1)
        self.assertEqual(self.m.sessions[4].acks, 0)
        self.m.reset()
        self.assertFalse(self.m.sessions)
        self.assertIsNone(self.m.last)

    def test_invalid_observations(self):
        o = observation()
        o["vision"]["cam_head"]["color"] = np.zeros((4, 4, 3), np.float32)
        with self.assertRaises(ValueError):
            self.m.update_obs(o)
        o = observation()
        o["instruction"] = ""
        with self.assertRaises(ValueError):
            self.m.update_obs(o)
        with self.assertRaises(ValueError):
            self.m.update_obs_batch([observation(2), observation(2)])

    def test_rgb_view_order(self):
        o = observation()
        for i, k in enumerate(o["vision"]):
            o["vision"][k]["color"][:] = i + 1
        a, _ = self.m._adapt(o)
        self.assertEqual(
            [
                int(a["observation"][k]["rgb"][0, 0, 0])
                for k in ["head_camera", "left_camera", "right_camera"]
            ],
            [1, 2, 3],
        )


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate_source()
        torch.set_num_threads(1)

    def test_repeat9_full_encoder(self):
        from eval.robotwin.current_visual_adapter import (
            install_current_repeat9,
            restore_current_encoder,
        )

        class Encoder:
            def encode_raw_video_latents(self, rgb, num_views=None):
                self.last = rgb.clone()
                return (
                    torch.arange(3.0)
                    .view(1, 1, 3, 1, 1)
                    .expand(rgb.shape[0], 1, 3, 2, 2)
                )

        e = Encoder()
        install_current_repeat9(e)
        result = e.encode_raw_video_latents(torch.ones(1, 3, 1, 8, 8), num_views=3)
        self.assertEqual(e.last.shape[2], 9)
        self.assertEqual(result.shape[2], 1)
        self.assertEqual(result.max(), 0)
        restore_current_encoder(e)

    def test_normalization_and_real_contract(self):
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from eval.robotwin.joint96_robot37_policy import validate_contract, current_rgb
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        cfg = OmegaConf.load(SOURCE / "configs/inference/phoenix_robotwin.yaml")
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(SOURCE / "assets/stats/robotwin.json"))
        )
        self.assertEqual(cfg.data.train.processor.norm_default_mode, "z-score")
        obs = {
            "observation": {
                k: {"rgb": np.full((480, 640, 3), 128, np.uint8)}
                for k in ["head_camera", "left_camera", "right_camera"]
            }
        }
        rgb = current_rgb(processor, obs)
        self.assertEqual(tuple(rgb.shape), (1, 3, 3, 256, 256))
        self.assertTrue(torch.isfinite(rgb).all())
        path = os.environ.get("PHOENIX_ROBOTWIN_TEST_WEIGHT")
        if not path:
            self.skipTest(
                "Set PHOENIX_ROBOTWIN_TEST_WEIGHT for real checkpoint contract"
            )
        payload = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
        self.assertEqual(payload["step"], 35000)
        validate_contract(cfg, payload, SOURCE / "assets/stats/robotwin.json")
        payload["joint96_metadata"]["soft_prompt"]["training_contract"][
            "embodiment_id"
        ] = 4
        with self.assertRaises(ValueError):
            validate_contract(cfg, payload, SOURCE / "assets/stats/robotwin.json")

    def test_indexed_cache_roundtrip(self):
        from ..training.data import write_shard
        from wam.datasets.text_embedding_cache import TaskIndexShardedTextEmbeddingCache

        prompts = ["a", "b"]
        contexts = [torch.randn(128, 4096, dtype=torch.bfloat16) for _ in prompts]
        masks = [torch.arange(128) < n for n in (4, 7)]
        with tempfile.TemporaryDirectory() as d:
            write_shard(d, 0, prompts, contexts, masks)
            Path(d, "manifest.json").write_text(
                json.dumps(
                    dict(
                        format="task_index_sharded_mmap_v1",
                        total_tasks=2,
                        context_len=128,
                        embedding_dim=4096,
                        shard_size=64,
                    )
                )
            )
            bank = TaskIndexShardedTextEmbeddingCache(d)
            for i in range(2):
                c, m = bank.get(i, prompts[i])
                torch.testing.assert_close(c[m], contexts[i][m], rtol=0, atol=0)
                self.assertFalse(c[~m].any())
            with self.assertRaises(ValueError):
                bank.get(0, "wrong prompt")

    def test_source_snapshot_hashes(self):
        from ..runtime.checkpoints import sha256

        manifest = json.loads((SOURCE / "source_manifest.json").read_text())
        for name, digest in manifest["sha256"].items():
            self.assertEqual(sha256(SOURCE / name), digest, name)


if __name__ == "__main__":
    unittest.main()
