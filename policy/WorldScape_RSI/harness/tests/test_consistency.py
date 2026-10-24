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
        from XPolicyLab.policy.RoboRSI_RoboProbe.library import SkillLibrary
        tasks = ["press_by_number", "imitate_sorting_sequence", "unseen_task", "../outside"]
        reference = None
        for task in tasks:
            # Use the actual shared library, not a stub hiding task-conditioned text.
            library = SkillLibrary(ROOT / "vendor/policy/RoboRSI_RoboProbe/skills")
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
        from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.runtime_config import validate_config
        policy_dir = Path(framework) / "policy/WorldScape_RSI"
        for name, spec in self.skills.items():
            with self.subTest(skill=name):
                wam._verify_files(policy_dir, spec["source_files"], "Pinned WSP adapter")
                cfg = validate_config(spec["config"])
                self.assertNotIn("task_name", cfg)
                self.assertEqual(cfg["wam_history_mode"], spec["config"]["wam_history_mode"])

    def test_release_rejects_memory_semantic_drift(self):
        original = json.loads((ROOT / "release.json").read_text())
        for skill, key, value in [
            ("worldscape-wam-nomemory", "use_observation_history", False),
            ("worldscape-wam-nomemory", "use_causal_memory", False),
            ("worldscape-wam-nomemory", "reset_independent_blocks", True),
            ("worldscape-wam-nomemory", "wam_history_mode", "periodic"),
            ("worldscape-wam-memory", "use_causal_memory", False),
            ("worldscape-wam-memory", "short_window_for_non_memory_tasks", True),
        ]:
            with self.subTest(skill=skill, key=key):
                changed = copy.deepcopy(original)
                changed["wam_skills"][skill]["config"][key] = value
                with self.assertRaises(ValueError):
                    release.validate(changed)

    def test_flash_rolling_keeps_latest_frame_and_continuous_rng_with_execution_ack(self):
        import torch
        from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter import robodojo_model as native
        cfg = self.skills["worldscape-wam-nomemory"]["config"]
        model = object.__new__(native.RoboDojoModel)
        for key in ("use_observation_history", "use_causal_memory", "reset_independent_blocks",
                    "wam_history_mode", "observation_history_segments"):
            setattr(model, key, cfg[key])
        model.device = torch.device("cpu")
        model.seed = 7
        model.action_steps = cfg["replan_steps"]
        model.wam_max_chunk_size = 1  # Published Flash checkpoint geometry.
        model.mode = "fixture"
        model.input_color_order = "rgb"
        model._resize_image = lambda image: image
        model.bundle = object()
        model.robot_action_dim_info = {}
        model.short_window_for_non_memory_tasks = False
        model.gripper_discretization = types.SimpleNamespace(enabled=False)
        resets, received_frames, inputs, draws = [], [], [], []

        @dataclass
        class Memory:
            step: int

        @dataclass
        class Output:
            action: object
            next_memory: object
            next_visual_memory: object

            def require_action(self):
                return self.action

        def sample(**kwargs):
            inputs.append(kwargs)
            draws.append(torch.rand(1, generator=kwargs["generator"]).item())
            return Output(torch.zeros((1, model.action_steps, 1)),
                          Memory(len(inputs)), Memory(len(inputs)))

        model.policy = types.SimpleNamespace(reset_episode=lambda: resets.append(1), sample=sample)
        model.reset()

        def frame(value):
            return {"vision": {name: {"color": np.full((4, 4, 3), value, dtype=np.uint8)}
                               for name in native.CAMERA_ORDER}}

        def adapt(env, *, cameras=None):
            if cameras is None:
                cameras = model._prepare_observation_cameras(env)
            received_frames.append(cameras)
            return types.SimpleNamespace(observation=object(), prompts=object(), raw_state={})

        model._adapt_observation = adapt
        with patch.object(native, "_stack_observation_batches", side_effect=lambda x: x), \
             patch.object(native, "_stack_prompt_batches", side_effect=lambda x: x), \
             patch.object(native, "_action_fields", return_value={}), \
             patch.object(native, "_fields_to_robodojo_actions",
                          return_value=[hold() for _ in range(model.action_steps)]):
            model.update_obs(frame(0))
            for start in (0, 24):
                model.get_action()
                # Intermediate observations do not commit proposed model state.
                committed_memory = model._event_memory
                for step in range(1, model.action_steps):
                    model.update_obs(frame(start + step))
                self.assertIs(model._event_memory, committed_memory)
                # The official lifecycle commits before forwarding the final frame.
                model.commit_action({"executed_steps": {0: model.action_steps},
                                     "finished_env_idx_list": []})
                model.update_obs(frame(start + model.action_steps))
                self.assertEqual(model._event_memory.step, start // 24 + 1)
                self.assertEqual(model._visual_memory.step, start // 24 + 1)
                self.assertEqual(model._memory_env_idx_list, [0])
            self.assertEqual(len(resets), 1)  # Only the initial reset.
            model.get_action()
            model.discard_action({"execution_started": True})
            self.assertIsNone(model._event_memory)
            self.assertIsNone(model._visual_memory)
            self.assertEqual(model._rolling_history_by_env, {})
            model.update_obs(frame(49))
            model.get_action()
        self.assertEqual(len(resets), 2)  # Partial execution rebases model context.
        self.assertEqual(model._prediction_index, 2)
        self.assertEqual(len(received_frames), 4)
        for cameras, value in zip(received_frames, (0, 24, 48, 49)):
            for camera in cameras:
                self.assertEqual(camera.shape[0], 1)
                self.assertTrue(np.all(camera == value))
        self.assertIsNone(inputs[0]["event_memory"])
        self.assertEqual(inputs[1]["event_memory"].step, 1)
        self.assertEqual(inputs[2]["event_memory"].step, 2)
        self.assertIsNone(inputs[3]["event_memory"])
        self.assertIsNone(inputs[3]["visual_memory"])
        expected_rng = torch.Generator().manual_seed(model.seed)
        self.assertEqual(draws, [torch.rand(1, generator=expected_rng).item() for _ in draws])


if __name__ == "__main__":
    unittest.main()
