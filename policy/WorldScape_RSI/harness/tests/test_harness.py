"""CPU-only contract tests. Fixtures do not establish robot task success."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
framework = os.environ.get("R2SI_TEST_XPL_ROOT")
if framework:
    sys.path[:0] = [str(Path(framework).parent), framework]
try:
    import XPolicyLab.model_template
    if not framework:
        framework = str(Path(XPolicyLab.model_template.__file__).parent)
        sys.path.insert(0, framework)
except ModuleNotFoundError:
    # Only framework installation and robot-dimension lookup are fixture dependencies.
    xpl = types.ModuleType("XPolicyLab")
    xpl.__path__ = []
    template = types.ModuleType("XPolicyLab.model_template")
    template.ModelTemplate = type("ModelTemplate", (), {})
    sys.modules.update({"XPolicyLab": xpl, "XPolicyLab.model_template": template})

spec = importlib.util.spec_from_file_location("r2si_under_test", ROOT / "__init__.py",
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from r2si_under_test import backends, contracts, development, model, provider, release

DIMS = {side + suffix: size for side in ("left", "right")
        for suffix, size in (("_arm_joint_state", 6), ("_ee_joint_state", 1))}


def obs():
    state = {key: np.zeros(n) for key, n in DIMS.items()}
    state.update({side + "_ee_pose": np.array([0., 0., 1., 1., 0., 0., 0.])
                  for side in ("left", "right")})
    return {"instruction": "Follow the current visible goal", "state": state,
            "vision": {"head_camera": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}}}


def hold(value=0.):
    return {key: np.full(n, value) for key, n in DIMS.items()}


class FakeBackend:
    def __init__(self):
        self.closed = False
        self.plans = 0
        self.confirmed = []
        self.interrupt = False
        self.invalid = False

    def reset(self):
        pass

    def observe(self, observation, previous):
        return self.interrupt

    def plan(self):
        self.plans += 1
        return [hold(.1), {"bad": np.zeros(1)}] if self.invalid else [hold(.1), hold(.2)]

    def before(self, action):
        pass

    def completed(self, played, planned, interrupted):
        self.confirmed.append((played, planned, interrupted))

    def close(self):
        self.closed = True


class FakeRelease:
    root = ROOT
    sha256 = "fixture-only"

    def verify(self):
        pass

    def agent_spec(self):
        return {"kind": "fixture-only"}


class CoreTests(unittest.TestCase):
    def test_large_hold_is_constant_space_and_stops_at_trial_end(self):
        from r2si_under_test.plans import RepeatedAction
        self.backend.plan = lambda: RepeatedAction(hold(), 10_000_000)
        self.model.update_obs(obs()); self.model.get_action()
        self.assertEqual(len(self.model._queue), 0)
        self.assertEqual(self.model._repeat_remaining, 9_999_999)
        self.model.update_obs(obs()); self.model.get_action()
        self.model.on_trial_end({})
        self.assertEqual(self.model._repeat_remaining, 0)

    def setUp(self):
        self.backend = FakeBackend()
        self.patches = [patch.object(model, "FrozenRelease", return_value=FakeRelease()),
                        patch.object(contracts, "action_dimensions", return_value=DIMS),
                        patch.object(backends, "build_backend", side_effect=lambda *a: self.backend)]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.cfg = {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint",
                    "policy_name": "WorldScape_RSI", "task_name": "task_a", "eval_batch": False}
        self.model = model.Model(self.cfg)
        self.model.reset()

    def test_native_action_acknowledgement_and_chunk_reuse(self):
        self.model.update_obs(obs())
        self.model.get_action()
        self.model.update_obs(obs())
        self.model.get_action()
        self.assertEqual(self.backend.plans, 1)
        self.model.update_obs(obs())
        self.assertEqual(self.backend.confirmed, [(2, 2, False)])
        self.model.get_action()
        self.assertEqual(self.backend.plans, 2)

    def test_no_action_without_observation(self):
        with self.assertRaises(RuntimeError):
            self.model.get_action()

    def test_no_duplicate_get_action(self):
        self.model.update_obs(obs()); self.model.get_action()
        with self.assertRaises(RuntimeError):
            self.model.get_action()

    def test_no_duplicate_observation_ack(self):
        self.model.update_obs(obs())
        with self.assertRaises(RuntimeError):
            self.model.update_obs(obs())

    def test_interrupt_discards_suffix_and_confirms_only_observed_action(self):
        self.model.update_obs(obs()); self.model.get_action()
        self.backend.interrupt = True
        self.model.update_obs(obs())
        self.assertEqual(self.backend.confirmed, [(1, 2, True)])
        self.backend.interrupt = False
        self.model.get_action()
        self.assertEqual(self.backend.plans, 2)

    def test_validate_whole_chunk_before_first_action(self):
        self.backend.invalid = True
        self.model.update_obs(obs())
        with self.assertRaises(ValueError):
            self.model.get_action()
        self.assertIsNone(self.model._pending)

    def test_observation_and_action_are_copied(self):
        raw = obs(); self.model.update_obs(raw)
        raw["state"]["left_arm_joint_state"][:] = 99
        self.assertTrue(np.all(self.model._observation["state"]["left_arm_joint_state"] == 0))
        result = self.model.get_action()
        result[0]["left_arm_joint_state"][:] = 99
        self.assertTrue(np.all(self.model._pending["left_arm_joint_state"] == .1))

    def test_failure_requires_reset_and_never_selects_fallback(self):
        class BackendFailure(BaseException):
            pass
        self.backend.plan = lambda: (_ for _ in ()).throw(BackendFailure())
        self.model.update_obs(obs())
        with self.assertRaises(BackendFailure):
            self.model.get_action()
        with self.assertRaises(RuntimeError):
            self.model.update_obs(obs())
        self.assertEqual(self.model.release_metadata()["executor"], "inner_agent")

    def test_no_task_switch_within_active_trial(self):
        with self.assertRaises(RuntimeError):
            self.model.prepare_case({"task_name": "task_b"})
        self.model.close(); self.model.prepare_case({"task_name": "task_b"})
        self.model.reset()
        self.assertEqual(self.model.release_metadata()["task"], "task_b")

    def test_conflicting_case_metadata_rejected(self):
        with self.assertRaises(ValueError):
            self.model.prepare_case({"task_name": "task_a", "action_case_id": "task_b_case"})

    def test_no_cross_episode_queue_or_observation(self):
        self.model.update_obs(obs()); self.model.get_action()
        self.model.on_trial_end({})
        self.assertFalse(self.model._queue)
        self.assertIsNone(self.model._pending)
        self.assertTrue(self.backend.closed)

    def test_new_public_task_uses_the_same_inner_agent(self):
        self.model.close()
        self.model.prepare_case({"task_name": "new_public_task"})
        self.model.reset()
        self.assertEqual(self.model.release_metadata()["executor"], "inner_agent")
        self.assertEqual(self.model.release_metadata()["selection_scope"], "agent_selected_skill")

    def test_batch_and_model_override_rejected(self):
        for extra in ({"eval_batch": True}, {"checkpoint_path": "/changed"}):
            with self.assertRaises(ValueError):
                model.Model({**self.cfg, **extra})
        with self.assertRaises(NotImplementedError):
            self.model.get_action_batch([0, 1])


class ContractTests(unittest.TestCase):
    def test_private_fields_never_reach_executor(self):
        raw = obs()
        raw.update(score=1, layout_id=2, seed=3, task_id="injected", depth=np.ones(3))
        raw["state"]["object_pose"] = np.ones(7)
        raw["vision"]["head_camera"]["depth"] = np.ones((8, 8))
        filtered = contracts.observation(raw, DIMS)
        self.assertEqual(set(filtered), {"instruction", "state", "vision", "data_format_version"})
        self.assertNotIn("object_pose", filtered["state"])
        self.assertEqual(set(filtered["vision"]["head_camera"]), {"color"})

    def test_reject_encoded_colors_and_invalid_actions(self):
        raw = obs(); raw["vision"]["head_camera"]["color"] = b"encoded JPEG"
        with self.assertRaises(ValueError):
            contracts.observation(raw, DIMS)
        for item in (hold(float("nan")), hold(2), {"unexpected": [1]}):
            with self.assertRaises(ValueError):
                contracts.action(item, DIMS)


class ReleaseTests(unittest.TestCase):
    def test_freeze_detects_skill_tampering_and_manifest_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "configs").mkdir()
            for name in ("inner_agent.json", "wam_skills.json"):
                (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
            asset = root / "SKILL.md"; asset.write_text("initial strategy")
            frozen = release.FrozenRelease(release.freeze(root))
            asset.write_text("changed strategy")
            with self.assertRaises(RuntimeError): frozen.verify()
            asset.write_text("initial strategy")
            frozen.verify()
            frozen.path.write_text("{}")
            with self.assertRaises(RuntimeError): frozen.verify()

    def test_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "target").write_text("x")
            (root / "link").symlink_to(root / "target")
            for name in ("../escape", "link"):
                with self.assertRaises(ValueError): release.contained(root, name)

    def test_task_routing_and_extra_models_rejected(self):
        data = {"schema": release.SCHEMA, "mode": "inner_agent_wam_skills",
                "official_single_entry_eligible": False,
                "supported": {"bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint"},
                "inner_agent": json.loads((ROOT / "configs/inner_agent.json").read_text()),
                "wam_skills": json.loads((ROOT / "configs/wam_skills.json").read_text())}
        release.validate(data)
        routed = copy.deepcopy(data); routed["routes"] = {"a": "worldscape-wam-memory"}
        with self.assertRaises(ValueError): release.validate(routed)
        extra = copy.deepcopy(data); extra["wam_skills"]["unsupported_model"] = {}
        with self.assertRaises(ValueError): release.validate(extra)
        missing = copy.deepcopy(data); missing["wam_skills"].pop("worldscape-wam-nomemory")
        with self.assertRaises(ValueError): release.validate(missing)

    def test_decisions_without_matching_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.json"
            path.write_text(json.dumps({"schema": "r2si.asset-revisions.v2", "author_type": "outer_agent",
                "agent_run_id": "fixture", "updates": [{"target":
                "vendor/policy/RoboRSI_RoboProbe/skills/generic/primitive-control/SKILL.md",
                "source": "missing", "reason": "fixture", "evidence": []}]}))
            with self.assertRaises(ValueError): development.import_decisions(path, Path(tmp) / "candidate")


class ProviderTests(unittest.TestCase):
    def test_legacy_external_provider_profile_preserves_frozen_model(self):
        spec = json.loads((ROOT / "configs/inner_agent.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "secrets.json"
            path.write_text(json.dumps({"vlm": {"astra": {"model": spec["model"],
                "base_url": "https://fixture.invalid/v1", "api_key": "fixture"}}}))
            with patch.dict(os.environ, {"DEXBOTIC_SECRETS": str(path), "VLM_AGENT_PROFILE": "astra",
                                         "R2SI_API_BASE": "", "R2SI_API_KEY": ""}):
                client = provider.ResponsesClient(spec)
                self.assertEqual(client.config["model"], spec["model"]); client.close()
                path.write_text(json.dumps({"vlm": {"astra": {"model": "different-model",
                    "base_url": "https://fixture.invalid/v1", "api_key": "fixture"}}}))
                with self.assertRaises(ValueError): provider.ResponsesClient(spec)

    def test_sticky_session_and_context_integrity(self):
        env = {"R2SI_API_BASE": "https://fixture.invalid/v1", "R2SI_API_KEY": "fixture-only"}
        spec = json.loads((ROOT / "configs/inner_agent.json").read_text())
        with patch.dict(os.environ, env): client = provider.ResponsesClient(spec)
        calls = []
        raw = {"status": "completed", "output": [{"type": "function_call", "call_id": "call_1",
                                                     "name": "read_skill", "arguments": "{}"}]}
        def request(req, timeout):
            calls.append(req)
            return io.StringIO(json.dumps(raw))
        client.opener.open = request
        tools = [{"type": "function", "function": {"name": "read_skill", "description": "read",
                  "parameters": {"type": "object", "properties": {}}}}]
        messages = [{"role": "system", "content": "fixture"}, {"role": "user", "content": "observation"}]
        client.complete(messages, tools)
        session = client.session_id
        messages.append({"role": "user", "content": "next observation"})
        client.complete(messages, tools)
        self.assertEqual(client.session_id, session)
        with self.assertRaises(RuntimeError): client.complete(messages[:1], tools)
        self.assertEqual(len(calls), 2)
        client.reset(); self.assertNotEqual(client.session_id, session)
        client.close()
        with self.assertRaises(RuntimeError): client.complete(messages, tools)


class ActualAgentRepairTests(unittest.TestCase):
    @unittest.skipUnless(framework, "Install XPolicyLab to exercise the retained Agent")
    def test_repeated_format_repairs_preserve_api_history(self):
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import MotionOutcome
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action, ActionChunk
        actor = object.__new__(SkillEefPolicy)
        actor._model_refusal = None; actor._keep_all_images = True; actor._calls = 0
        actor._messages = [{"role": "system", "content": "fixture"}]
        actor._tools = [{"type": "function", "function": {"name": "call_tool", "description": "fixture",
                         "parameters": {"type": "object", "properties": {}}}}]
        actor.extra_images = []
        actor.prepare = lambda observation: None
        actor._observation_message = lambda observation: {"role": "user", "content": "RGB fixture"}
        actor._normalize_tool_calls = lambda calls: calls
        actor._record_turn = lambda *a, **kw: {}
        actor._handle_motion = lambda *a: MotionOutcome(
            chunk=ActionChunk(actions=[Action(data=hold(), meta={})], control_hz=25), tool_result="fixture plan")
        spec = json.loads((ROOT / "configs/inner_agent.json").read_text())
        with patch.dict(os.environ, {"R2SI_API_BASE": "https://fixture.invalid", "R2SI_API_KEY": "fixture"}):
            actor._client = provider.ResponsesClient(spec)
        counter = []
        def response(request, timeout):
            counter.append(request)
            items = [] if len(counter) < 3 else [{"type": "function_call", "call_id": "call_3",
                                                "name": "call_tool", "arguments": "{}"}]
            return io.StringIO(json.dumps({"status": "completed", "output": items}))
        actor._client.opener.open = response
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Observation
        raw = obs()
        current = Observation(images={k:v["color"] for k,v in raw["vision"].items()},
                              state=raw["state"], instruction=raw["instruction"], step=0,
                              remaining_steps=None, extra={"task": "task_a"})
        chunk = actor.act(current)
        self.assertEqual(len(counter), 3)
        self.assertEqual(len(chunk.actions), 1)
        self.assertEqual(sum("Response format invalid" in str(m.get("content")) for m in actor._messages), 2)


if __name__ == "__main__":
    unittest.main()
