"""WAM Skill lifecycle fixtures; no weights or GPU inference required."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import os

from test_harness import ROOT, DIMS, obs, hold, backends, model, release, framework
from r2si_under_test import wam

MEM = "worldscape-wam-memory"
NOMEM = "worldscape-wam-nomemory"


class FakeWsp:
    def __init__(self):
        self.events = []
        self.closed = False
    def reset(self): self.events.append(("reset",))
    def update_obs(self, observation): self.events.append(("obs",))
    def get_action(self): self.events.append(("predict",)); return [hold(.1), hold(.2)]
    def commit_action(self, receipt): self.events.append(("commit", receipt))
    def discard_action(self, receipt): self.events.append(("discard", receipt))
    def close(self): self.closed = True


class WamSkillTests(unittest.TestCase):
    def setUp(self):
        self.loaded = {}
        self.catalog = {MEM: {"profile": "memory"}, NOMEM: {"profile": "nomemory"}}
        def factory(spec, task, deployment):
            m = FakeWsp(); self.loaded[spec["profile"]] = m; return m
        self.skills = wam.WamSkills(self.catalog, "any_task", {}, factory=factory, dims=DIMS)
        self.skills.observe(obs(), None)

    def execute(self, skill=MEM, interrupted=False):
        actions = self.skills.plan(skill)
        for action in actions:
            self.skills.before(action)
            self.skills.observe(obs(), action, interrupted)
            if interrupted: break

    def test_both_models_are_skills_and_load_only_when_called(self):
        self.assertEqual(self.loaded, {})
        self.execute(MEM); self.execute(NOMEM)
        self.assertEqual(set(self.loaded), {"memory", "nomemory"})

    def test_full_commit_precedes_final_observation(self):
        self.execute()
        events = self.loaded["memory"].events
        self.assertEqual(events[-2], ("commit", {"executed_steps": {0: 2}, "finished_env_idx_list": []}))
        self.assertEqual(events[-1], ("obs",))

    def test_consecutive_same_model_keeps_memory_without_duplicate_start_frame(self):
        self.execute(); events = self.loaded["memory"].events
        resets = events.count(("reset",)); observations = events.count(("obs",))
        self.execute()
        self.assertEqual(events.count(("reset",)), resets)
        self.assertEqual(events.count(("obs",)), observations + 2)

    def test_interruption_discards_and_rebases_from_live_frame(self):
        self.execute(interrupted=True)
        events = self.loaded["memory"].events
        self.assertEqual(events[-2:], [("discard", {"execution_started": True}), ("obs",)])
        self.assertFalse(any(e[0] == "commit" for e in events))
        self.execute()
        self.assertTrue(any(e[0] == "commit" for e in events))

    def test_native_motion_breaks_continuity_and_stale_history(self):
        self.execute(); self.skills.native_selected()
        self.skills.observe(obs(), hold())
        self.execute()
        self.assertEqual(self.loaded["memory"].events.count(("reset",)), 3)

    def test_variant_switch_resets_the_old_context(self):
        self.execute(MEM); old = self.loaded["memory"]
        self.execute(NOMEM)
        self.assertEqual(old.events[-1], ("reset",))
        self.execute(MEM)
        self.assertEqual(old.events.count(("reset",)), 3)

    def test_final_unobserved_action_discards_instead_of_committing(self):
        a = self.skills.plan(MEM); self.skills.before(a[0]); self.skills.close()
        m = self.loaded["memory"]
        self.assertTrue(m.closed)
        self.assertIn(("discard", {"execution_started": True}), m.events)
        self.assertFalse(any(e[0] == "commit" for e in m.events))
        self.assertIsNone(self.skills.latest)

    def test_invalid_chunk_is_rejected_before_execution(self):
        self.execute(); self.loaded["memory"].get_action = lambda: [hold(), {"bad": [1]}]
        with self.assertRaises(ValueError): self.skills.plan(MEM)
        self.skills.close()
        self.assertIn(("discard", {"execution_started": False}), self.loaded["memory"].events)

    def test_no_switch_or_prediction_replay_while_pending(self):
        self.skills.plan(MEM)
        with self.assertRaises(RuntimeError): self.skills.plan(NOMEM)
        with self.assertRaises(RuntimeError): self.skills.native_selected()

    def test_inference_failure_has_cleanup_ownership_and_is_not_replayed(self):
        self.execute(); m = self.loaded["memory"]
        calls = []
        def failed(): calls.append(1); raise RuntimeError("ambiguous response")
        m.get_action = failed
        with self.assertRaises(RuntimeError): self.skills.plan(MEM)
        with self.assertRaises(RuntimeError): self.skills.plan(MEM)
        self.skills.close()
        self.assertEqual(len(calls), 1)
        self.assertTrue(m.closed)

    def test_missing_execution_acknowledgement_rejected(self):
        a = self.skills.plan(MEM)
        with self.assertRaises(RuntimeError): self.skills.observe(obs(), a[0])

    def test_wam_tool_dispatch_and_forbidden_overrides(self):
        if not framework: self.skipTest("Install XPolicyLab for retained tool dispatch")
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe import tools
        library = types.SimpleNamespace(entries={MEM: {"stages": ["predict"]}})
        calls = []
        actor = types.SimpleNamespace(library=library, wam_call=lambda key, ob: calls.append(key) or "chunk")
        self.assertEqual(tools.execute(actor, "call_tool", {"id": MEM, "arguments": {}}, None), "chunk")
        with self.assertRaises(ValueError):
            tools.execute(actor, "call_tool", {"id": MEM, "arguments": {"checkpoint": "/injected"}}, None)
        self.assertEqual(calls, [MEM])

    def test_real_agent_tool_loop_returns_to_agent_between_wam_chunks(self):
        if not framework: self.skipTest("Install XPolicyLab for retained Agent")
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, Observation
        from test_harness import FakeRelease
        b = object.__new__(backends.AgentBackend)
        b.action_type, b.observation_type = Action, Observation
        b.task = "unseen_task"; b.steps = 0; b.current = None; b.reasons = []
        b.spec = types.SimpleNamespace(control_hz=25)
        b.planner = types.SimpleNamespace(set_observation=lambda obs: None)
        b.monitor = types.SimpleNamespace(begin=lambda ob: None, before=lambda *a: None,
                                         after=lambda *a: {"reasons": []})
        b.wam = self.skills
        actor = object.__new__(SkillEefPolicy)
        actor.action_spec = b.spec
        actor.library = types.SimpleNamespace(entries={MEM: {"stages": ["predict"]},
                                                       NOMEM: {"stages": ["predict"]},
                                                       "primitive-control": {"stages": ["control"]}})
        actor.wam_call = b._wam_call
        actor._model_refusal = None; actor._keep_all_images = True; actor._calls = 0
        actor._messages = [{"role": "system", "content": "fixture"}]
        actor._tools = [{"function": {"name": "call_tool"}}]
        actor.extra_images = []; actor._transcript = []
        actor.prepare = lambda ob: None
        actor._observation_message = lambda ob: {"role": "user", "content": "RGB fixture"}
        actor._normalize_tool_calls = lambda calls: calls
        actor._record_turn = lambda *a, **kw: {}
        actor._held_action_data = lambda ob: hold()
        actor.confirm_executed = lambda played: None
        calls = []
        sequence = iter([(MEM, {}), ("primitive-control", {"op": "wait", "steps": 1}), (NOMEM, {})])
        def complete(messages, tools):
            skill, arguments = next(sequence); calls.append(skill)
            return {"choices": [{"message": {"content": None, "tool_calls": [{"id": "c" + str(len(calls)),
                "function": {"name": "call_tool", "arguments": json.dumps({"id": skill, "arguments": arguments})}}]}}]}
        b.client = actor._client = types.SimpleNamespace(complete=complete, close=lambda: None, reset=lambda: None)
        actor.reset = lambda: None
        b.policy = actor
        cfg = {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint",
               "policy_name": "WorldScape_RSI", "task_name": "unseen_task", "eval_batch": False}
        with patch.object(model, "FrozenRelease", return_value=FakeRelease()), \
                patch.object(model.contracts, "action_dimensions", return_value=DIMS), \
                patch.object(backends, "build_backend", return_value=b):
            adapter = model.Model(cfg); adapter.reset(); adapter.update_obs(obs())
            for _ in range(5): adapter.get_action(); adapter.update_obs(obs())
            adapter.close()
        self.assertEqual(calls, [MEM, "primitive-control", NOMEM])
        self.assertTrue(any(e[0] == "commit" for e in self.loaded["memory"].events))
        self.assertTrue(any(e[0] == "commit" for e in self.loaded["nomemory"].events))
        self.assertTrue(all(m.closed for m in self.loaded.values()))
        self.assertEqual(actor._gripper_goals, {"left": 0., "right": 0.})

    def test_adjacent_checkpoint_loading_is_pinned_and_explicit(self):
        if not framework: self.skipTest("Install XPolicyLab for checkpoint resolver")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); own = root / "WorldScape_RSI/harness"; own.mkdir(parents=True)
            policy = root / "WorldScape_Policy_2"; policy.mkdir()
            source = policy / "model.py"; source.write_text("# external WSP fixture")
            weights = root / "weights"; weights.mkdir(); weight = weights / "model.bin"; weight.write_bytes(b"fixture")
            spec = {"policy_name": "WorldScape_Policy_2", "config": {}, "checkpoint_env": "FIXTURE_WAM_CKPT",
                    "worldscape_root_env": "FIXTURE_WAM_ROOT", "device_env": "FIXTURE_WAM_DEVICE",
                    "checkpoint_files": {"model.bin": release.digest(weight)}, "source_files": {"model.py": release.digest(source)}}
            class External(FakeWsp):
                def __init__(self, cfg): super().__init__(); self.config = cfg
                def get_runtime_info(self): return {"requires_explicit_commit": True}
            with patch.dict(os.environ, {"FIXTURE_WAM_CKPT": str(weights)}), patch.object(wam, "ROOT", own), \
                    patch.object(wam.inspect, "getfile", return_value=str(source)), \
                    patch.object(wam.importlib, "import_module", return_value=types.SimpleNamespace(Model=External)) as load:
                m = wam.create_model(spec, "task_a", {"seed": 7})
                self.assertEqual(m.config["checkpoint_path"], str(weights))
                self.assertEqual(m.config["seed"], 7)
                self.assertNotIn("task_name", m.config)
                weight.write_bytes(b"tampered"); load.reset_mock()
                with self.assertRaises(RuntimeError): wam.create_model(spec, "task_a", {})
                load.assert_not_called()


if __name__ == "__main__": unittest.main()
