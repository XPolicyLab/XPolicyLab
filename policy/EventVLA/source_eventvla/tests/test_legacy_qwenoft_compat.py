import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eventvla.model.legacy_compat import normalize_legacy_checkpoint_config
from eventvla.model.memory_ablation import get_memory_ablation_profile


def _legacy_pure_image_config():
    return {
        "framework": {
            "name": "QwenOFT",
            "memory_ablation_mode": "pure_image_keyframe_memory",
            "qwenvl": {
                "base_vlm": "/path/to/Qwen3-VL-4B-Instruct",
            },
            "action_model": {
                "action_dim": 14,
                "future_action_window_size": 49,
                "past_action_window_size": 0,
            },
            "memory_buffer": {
                "enable": False,
                "qwen_memory_injection": {
                    "enabled": True,
                    "mode": "pure_image_keyframe_memory",
                    "max_keyframe_images": 4,
                },
            },
        },
        "datasets": {
            "vla_data": {
                "provide_teacher_commit_images": False,
                "keyframe_image_memory": {
                    "enabled": True,
                    "max_keyframes": 4,
                },
                "temporal": {
                    "image": {
                        "absolute_indices": [0],
                        "delta_indices": [-30, -15, 0],
                    },
                },
            },
        },
        "trainer": {
            "pretrained_checkpoint": None,
        },
    }


def _legacy_raw_anchors_config(include_memory_ablation_mode=False):
    cfg = _legacy_pure_image_config()
    if include_memory_ablation_mode:
        cfg["framework"]["memory_ablation_mode"] = "raw_anchors_only"
    else:
        cfg["framework"].pop("memory_ablation_mode", None)
    cfg["framework"]["memory_buffer"]["qwen_memory_injection"]["mode"] = "raw_anchors_only"
    cfg["datasets"]["vla_data"]["keyframe_image_memory"]["enabled"] = False
    return cfg


class LegacyQwenOFTCompatTest(unittest.TestCase):
    def test_maps_legacy_pure_image_qwenoft_to_eventvla(self):
        cfg = _legacy_pure_image_config()

        normalized, info = normalize_legacy_checkpoint_config(cfg)

        self.assertIs(normalized, cfg)
        self.assertTrue(info["enabled"])
        self.assertEqual(cfg["framework"]["name"], "EventVLA")
        self.assertEqual(cfg["framework"]["legacy_source_name"], "QwenOFT")
        self.assertEqual(cfg["framework"]["compat_loaded_as"], "EventVLA")
        self.assertFalse(cfg["framework"]["memory_buffer"]["enable"])
        self.assertEqual(cfg["framework"]["memory_buffer"]["memory_write_policy"], "disabled")
        self.assertFalse(cfg["framework"]["memory_buffer"]["force_memory_write_current_for_event_commit"])
        self.assertTrue(cfg["framework"]["memory_buffer"]["disable_current_frame_keyframe_write_in_eval"])
        self.assertFalse(cfg["framework"]["memory_buffer"]["use_teacher_future_frame_write_in_train"])
        self.assertFalse(cfg["datasets"]["vla_data"]["provide_teacher_commit_images"])
        self.assertTrue(cfg["datasets"]["vla_data"]["keyframe_image_memory"]["enabled"])
        self.assertEqual(
            cfg["framework"]["memory_buffer"]["qwen_memory_injection"]["mode"],
            "pure_image_keyframe_memory",
        )
        self.assertEqual(cfg["framework"]["memory_buffer"]["qwen_memory_injection"]["max_keyframe_images"], 4)

    def test_maps_legacy_raw_anchors_only_without_memory_ablation_mode(self):
        cfg = _legacy_raw_anchors_config(include_memory_ablation_mode=False)

        normalized, info = normalize_legacy_checkpoint_config(cfg)

        self.assertIs(normalized, cfg)
        self.assertTrue(info["enabled"])
        self.assertEqual(cfg["framework"]["name"], "EventVLA")
        self.assertEqual(cfg["framework"]["legacy_source_name"], "QwenOFT")
        self.assertEqual(cfg["framework"]["compat_loaded_as"], "EventVLA")
        self.assertEqual(cfg["framework"]["memory_ablation_mode"], "raw_anchors_only")
        self.assertFalse(cfg["framework"]["memory_buffer"]["enable"])
        self.assertEqual(cfg["framework"]["memory_buffer"]["memory_write_policy"], "disabled")
        self.assertEqual(
            cfg["framework"]["memory_buffer"]["qwen_memory_injection"]["mode"],
            "raw_anchors_only",
        )
        self.assertFalse(cfg["datasets"]["vla_data"]["provide_teacher_commit_images"])
        self.assertFalse(cfg["datasets"]["vla_data"]["keyframe_image_memory"]["enabled"])

    def test_maps_legacy_raw_anchors_only_with_explicit_memory_ablation_mode(self):
        cfg = _legacy_raw_anchors_config(include_memory_ablation_mode=True)

        normalized, info = normalize_legacy_checkpoint_config(cfg)

        self.assertIs(normalized, cfg)
        self.assertTrue(info["enabled"])
        self.assertEqual(cfg["framework"]["name"], "EventVLA")
        self.assertEqual(cfg["framework"]["memory_ablation_mode"], "raw_anchors_only")
        self.assertFalse(cfg["datasets"]["vla_data"]["keyframe_image_memory"]["enabled"])

    def test_eventvla_config_is_noop(self):
        cfg = _legacy_pure_image_config()
        cfg["framework"]["name"] = "EventVLA"

        normalized, info = normalize_legacy_checkpoint_config(cfg)

        self.assertIs(normalized, cfg)
        self.assertFalse(info["enabled"])
        self.assertEqual(cfg["framework"]["name"], "EventVLA")
        self.assertNotIn("legacy_source_name", cfg["framework"])

    def test_rejects_legacy_token_memory_mode(self):
        cfg = _legacy_pure_image_config()
        cfg["framework"]["memory_ablation_mode"] = "memory_tokens_only"
        cfg["framework"]["memory_buffer"]["enable"] = True
        cfg["framework"]["memory_buffer"]["qwen_memory_injection"]["mode"] = "memory_tokens_only"

        with self.assertRaisesRegex(ValueError, "Token-memory|token-memory"):
            normalize_legacy_checkpoint_config(cfg)

    def test_rejects_legacy_bad_temporal_profile(self):
        cfg = deepcopy(_legacy_pure_image_config())
        cfg["datasets"]["vla_data"]["temporal"]["image"]["delta_indices"] = [0]

        with self.assertRaisesRegex(ValueError, "temporal image config mismatch"):
            normalize_legacy_checkpoint_config(cfg)

    def test_rejects_legacy_raw_anchors_only_bad_temporal_profile(self):
        cfg = _legacy_raw_anchors_config(include_memory_ablation_mode=False)
        cfg["datasets"]["vla_data"]["temporal"]["image"]["delta_indices"] = [0]

        with self.assertRaisesRegex(ValueError, "temporal image config mismatch"):
            normalize_legacy_checkpoint_config(cfg)

    def test_raw_anchors_only_memory_ablation_profile(self):
        profile = get_memory_ablation_profile("raw_anchors_only")

        self.assertEqual(profile.name, "raw_anchors_only")
        self.assertFalse(profile.enable_memory_buffer)
        self.assertFalse(profile.enable_keyframe_image_memory)
        self.assertEqual(profile.temporal_absolute_indices, (0,))
        self.assertEqual(profile.temporal_anchor_deltas, (-30, -15, 0))


if __name__ == "__main__":
    unittest.main()
