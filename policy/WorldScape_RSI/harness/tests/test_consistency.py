"""PR/runtime regression checks; actual adapter control flow, no model weights."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_harness import ROOT, backends, framework, hold, release
from r2si_under_test import wam


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        if not framework:
            self.skipTest("Install the pinned XPolicyLab adapter for integration checks")
        backends.register_vendor()
        self.skills = json.loads((ROOT / "configs/wam_skills.json").read_text())

    def test_tools_and_wait_are_identical_across_tasks(self):
        from XPolicyLab.policy.RoboRSI_RoboProbe import tools
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        root = ROOT / "vendor/policy/RoboRSI_RoboProbe/skills/tasks"
        tasks = sorted({p.name for p in root.iterdir() if p.is_dir()} |
                       {"arrange_largest_number", "imitate_sorting_sequence", "unseen_task"})
        reference = None
        for task in tasks:
            # Hold the allowed text catalog constant to isolate behavioral routing.
            library = types.SimpleNamespace(task_id=task, catalog=lambda kind: [], initial=lambda: "")
            actor = object.__new__(SkillEefPolicy)
            actor.library = library
            actor._held_action_data = lambda obs: hold()
            actor.action_spec = types.SimpleNamespace(control_hz=25)
            current = (tools.specs(library), actor._system_message())
            if reference is None:
                reference = current
            with self.subTest(task=task):
                self.assertEqual(current, reference)
                default = tools.primitive(actor, {"op": "wait"}, None)
                explicit = tools.primitive(actor, {"op": "wait", "steps": 5}, None)
                self.assertEqual(len(default.chunk.actions), 1)
                self.assertEqual(len(explicit.chunk.actions), 5)
                self.assertEqual(default.tool_result, explicit.tool_result)

    def test_both_wam_configs_are_accepted_by_pinned_adapter(self):
        from XPolicyLab.policy.WorldScape_Policy_2.xpolicylab_adapter.runtime_config import validate_config
        policy_dir = Path(framework) / "policy/WorldScape_Policy_2"
        for name, spec in self.skills.items():
            with self.subTest(skill=name):
                wam._verify_files(policy_dir, spec["source_files"], "Pinned WSP adapter")
                cfg = validate_config(spec["config"])
                self.assertNotIn("task_name", cfg)
                self.assertFalse(cfg["short_window_for_non_memory_tasks"])

    def test_release_rejects_memory_semantic_drift(self):
        original = json.loads((ROOT / "release.json").read_text())
        for skill, key, value in [
            ("worldscape-wam-nomemory", "use_observation_history", True),
            ("worldscape-wam-nomemory", "use_causal_memory", True),
            ("worldscape-wam-nomemory", "reset_independent_blocks", False),
            ("worldscape-wam-nomemory", "wam_history_mode", "rolling"),
            ("worldscape-wam-memory", "use_causal_memory", False),
            ("worldscape-wam-memory", "short_window_for_non_memory_tasks", True),
        ]:
            with self.subTest(skill=skill, key=key):
                changed = copy.deepcopy(original)
                changed["wam_skills"][skill]["config"][key] = value
                with self.assertRaises(ValueError):
                    release.validate(changed)

    def test_nomemory_adapter_samples_latest_frame_and_resets_each_prediction(self):
        import torch
        from XPolicyLab.policy.WorldScape_Policy_2.xpolicylab_adapter import robodojo_model as native
        cfg = self.skills["worldscape-wam-nomemory"]["config"]
        model = object.__new__(native.RoboDojoModel)
        for key in ("use_observation_history", "use_causal_memory", "reset_independent_blocks",
                    "wam_history_mode", "short_window_for_non_memory_tasks", "observation_history_segments"):
            setattr(model, key, cfg[key])
        model.device = torch.device("cpu")
        model.seed = 7
        model.action_steps = 1
        model.mode = "fixture"
        model.input_color_order = "rgb"
        model._resize_image = lambda image: image
        model.bundle = object()
        model.robot_action_dim_info = {}
        resets, received_frames, inputs = [], [], []

        @dataclass
        class Output:
            action: object
            next_memory: object
            next_visual_memory: object

            def require_action(self):
                return self.action

        def sample(**kwargs):
            inputs.append(kwargs)
            return Output(torch.zeros((1, 1, 1)), object(), object())

        model.policy = types.SimpleNamespace(reset_episode=lambda: resets.append(1), sample=sample)
        model.reset()

        def frame(value):
            return {"vision": {name: {"color": np.full((4, 4, 3), value, dtype=np.uint8)}
                               for name in native.CAMERA_ORDER}}

        def adapt(env):
            cameras = model._prepare_observation_cameras(env)
            received_frames.append(cameras)
            return types.SimpleNamespace(observation=object(), prompts=object(), raw_state={})

        model._adapt_observation = adapt
        with patch.object(native, "_stack_observation_batches", side_effect=lambda x: x), \
             patch.object(native, "_stack_prompt_batches", side_effect=lambda x: x), \
             patch.object(native, "_action_fields", return_value={}), \
             patch.object(native, "_fields_to_robodojo_actions", return_value=[hold()]), \
             patch.object(native, "match_memory_task", side_effect=AssertionError("Task routing invoked")):
            model.update_obs(frame(10))
            model.update_obs(frame(20))
            for value in (30, 40):
                model.get_action()
                model.commit_action({"executed_steps": {0: 1}, "finished_env_idx_list": []})
                self.assertIsNone(model._event_memory)
                self.assertIsNone(model._visual_memory)
                self.assertEqual(model._memory_env_idx_list, [])
                self.assertEqual(model._rolling_history_by_env, {})
                model.update_obs(frame(value))
        self.assertEqual(len(resets), 3)  # initial reset, then one per prediction
        self.assertEqual(model._prediction_index, 2)
        for cameras, value in zip(received_frames, (20, 30)):
            for camera in cameras:
                self.assertEqual(camera.shape[0], 1)
                self.assertTrue(np.all(camera == value))
        self.assertTrue(all(x["event_memory"] is None and x["visual_memory"] is None for x in inputs))


if __name__ == "__main__":
    unittest.main()
