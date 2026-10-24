"""Skills: manifests, resolution against services, kernel namespaces, and prompts that
describe exactly the API a kernel exposes."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.runtime import skills as registry  # noqa: E402
from harness_policy.runtime.prompts import assemble_prompt, documentation_gaps, select  # noqa: E402
from harness_policy.runtime.skills import (  # noqa: E402
    available_services, exported_names, load_skills, resolve, skills_for_layers)

ALL_SERVICES = {"segmenter", "pointer", "vlm", "verifier", "policy", "world_model"}
# Every skill of the full configuration (every layer on; optional skills only by name).
EVERYTHING = skills_for_layers({"learned": True, "reactive": True, "handlers": True})
PACKS = [ROOT / "prompts" / "robodojo"]
PACK_SKILLS = {"robodojo": list(EVERYTHING)}
PROFILE = {"control_hz": 25.0, "groups": [{"name": "left_arm", "joints": [0, 1, 2, 3, 4, 5], "gripper": 6},
                                         {"name": "right_arm", "joints": [7, 8, 9, 10, 11, 12], "gripper": 13}],
           "arm_bases": {"left_arm": [-0.3, -0.45, 0.765, 1, 0, 0, 0], "right_arm": [0.3, -0.45, 0.765, 1, 0, 0, 0]},
           "planner_limits": {"joint_velocity": 3.14, "joint_acceleration": 10.0}}


class FakeRobot:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, name, **arguments):
        if name == "profile":
            return dict(PROFILE)
        if name == "state":
            return {"step": 0, "step_limit": 100, "ended": False, "joint_position": [0.0] * 14,
                    "left_ee_pose": [0, 0, 1, 1, 0, 0, 0], "right_ee_pose": [0, 0, 1, 1, 0, 0, 0]}
        raise AssertionError(name)

    def close(self):
        pass


def session_names(options):
    from harness_policy.runtime import episode as episode_module, session as session_module

    with mock.patch.object(episode_module, "RemoteEnvironment", FakeRobot), \
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        session = session_module.create_session({**options, "endpoint": "http://127.0.0.1:1", "episode_id": "e"})
    return sorted(session.namespace)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RegistryTests(unittest.TestCase):
    def test_manifests_load_and_names_are_unique(self):
        skills = load_skills()
        self.assertIn("core", skills)
        self.assertTrue(all(skill.mixin.startswith(f"skills.{name}:") for name, skill in skills.items()))
        names = exported_names()
        self.assertEqual(sum(len(owners) for owners in names.values()),
                         sum(len(skill.exports) for skill in skills.values()))
        for name, owners in names.items():  # shared only by skills that never run together
            for i, a in enumerate(owners):
                for b in owners[i + 1:]:
                    self.assertTrue(b in skills[a].conflicts or a in skills[b].conflicts, name)

    def test_optional_and_conflicting_skills(self):
        # The tools by object id are the default; the manipulation skill's pick and place by
        # name are an ablation that replaces them (never both); scene's tools come with either.
        self.assertTrue({"eef", "scene"} <= set(EVERYTHING))
        self.assertNotIn("manipulation", EVERYTHING)
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            resolve([*EVERYTHING, "manipulation"], ALL_SERVICES)
        names = session_names({"skills": resolve(list(EVERYTHING), ALL_SERVICES).enabled})
        self.assertTrue({"locate", "pick", "release", "transfer", "push", "watch_scene", "clear_view", "press"}
                        <= set(names))
        self.assertNotIn("place", names)
        analytic = resolve([n for n in EVERYTHING if n != "eef"] + ["manipulation"], ALL_SERVICES).enabled
        names = session_names({"skills": analytic})
        self.assertTrue({"pick", "place", "push", "watch_scene", "clear_view", "press"} <= set(names))
        self.assertNotIn("locate", names)
        self.assertEqual(documentation_gaps(PACKS[0], analytic, exported_names()), {"missing": [], "stray": []})

    def test_layers_choose_skills(self):
        self.assertNotIn("vla", skills_for_layers({}))
        self.assertIn("vla", skills_for_layers({"learned": True}))
        self.assertNotIn("reactive", skills_for_layers({"reactive": False}))
        with self.assertRaises(ValueError):
            skills_for_layers({"memory": True})

    def test_missing_services_drop_skills_and_their_dependents(self):
        everything = list(EVERYTHING)
        resolution = resolve(everything, ALL_SERVICES - {"segmenter"})
        self.assertNotIn("segmentation", resolution.enabled)
        self.assertNotIn("eef", resolution.enabled)
        self.assertIn("pointing", resolution.enabled)
        self.assertEqual(resolution.dropped["segmentation"], "no segmenter service")
        resolution = resolve(everything, {"pointer"})
        self.assertIn("pointing", resolution.enabled)
        self.assertNotIn("vlm", resolution.enabled)
        self.assertNotIn("vla", resolution.enabled)
        self.assertTrue({"core", "motion", "kinematics", "geometry"} <= set(resolution.enabled))

    def test_strict_resolution_refuses_a_smaller_system(self):
        with self.assertRaisesRegex(ValueError, "wam"):
            resolve(list(EVERYTHING), ALL_SERVICES - {"world_model"}, strict=True)
        resolve(list(EVERYTHING), ALL_SERVICES, strict=True)

    def test_unknown_skills_and_bad_manifests(self):
        with self.assertRaisesRegex(ValueError, "Unknown skills"):
            resolve(["core", "teleport"], ALL_SERVICES)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "x").mkdir()
            (Path(root) / "x" / "skill.toml").write_text('name = "y"\nmixin = "m:C"\nexports = []\n')
            with self.assertRaisesRegex(ValueError, "directory name"):
                load_skills(Path(root))

    def test_services_from_kernel_options(self):
        self.assertEqual(available_services({"vlm": {"model": "m"}, "policies": {"g05": {}}, "world_models": {}}),
                         {"vlm", "policy"})

    def test_turn_reports_reach_the_drain(self):
        """Skills that define turn_report(session) add to the model's next turn after every
        program; a report that fails becomes a note, and an ended episode gets none."""
        from harness_policy.runtime import episode as episode_module, session as session_module

        self.assertIn("eef", [r.__module__.rsplit(".", 1)[-1] for r in registry.reporters(list(EVERYTHING))])
        with mock.patch.object(episode_module, "RemoteEnvironment", FakeRobot), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            session = session_module.create_session({"skills": ["core"], "endpoint": "http://127.0.0.1:1",
                                                     "episode_id": "e"})
        self.assertNotIn("report", session.drain())
        session._reporters = [lambda s: {"text": "Scene this turn: o1", "images": {"cam_head": "data:x"}},
                              lambda s: None, lambda s: 1 / 0]
        report = session.drain()["report"]
        self.assertTrue(report["text"].startswith("Scene this turn: o1"))
        self.assertIn("ZeroDivisionError", report["text"])
        self.assertEqual(report["images"], {"cam_head": "data:x"})
        session.state = {**session.state, "ended": True}
        self.assertNotIn("report", session.drain())

    def test_namespace_is_the_enabled_skills_exports(self):
        everything = list(EVERYTHING)
        names = session_names({"skills": everything})
        exported = {name for skill in load_skills().values() if skill.name in everything for name in skill.exports}
        self.assertEqual(set(names), exported | {"__name__"})
        some = resolve(everything, {"vlm"}).enabled
        names = session_names({"skills": some})
        self.assertNotIn("segment_text", names)
        self.assertIn("query_vlm", names)


class PromptMarkerTests(unittest.TestCase):
    def test_markers_select_lines(self):
        text = "a\n<!-- if: x -->\nb\n<!-- else -->\nc\n<!-- end -->\nd"
        self.assertEqual(select(text, {"x"}), "a\nb\nd")
        self.assertEqual(select(text, set()), "a\nc\nd")
        nested = "<!-- if: x -->\n<!-- if: y -->\nxy\n<!-- end -->\nx\n<!-- end -->"
        self.assertEqual(select(nested, {"x"}), "x")
        for broken in ("<!-- if: x -->\nb", "<!-- end -->", "<!-- else -->"):
            with self.assertRaises(ValueError):
                select(broken, set())

    def test_every_pack_documents_exactly_the_enabled_api(self):
        exported = exported_names()
        for pack in PACKS:
            everything = PACK_SKILLS[pack.name]
            self.assertEqual(documentation_gaps(pack, everything, exported), {"missing": [], "stray": []}, pack.name)
            for name, skill in load_skills().items():
                if skill.required or name not in everything:
                    continue
                enabled = resolve([n for n in everything if n != name], ALL_SERVICES).enabled
                with self.subTest(pack=pack.name, without=name):
                    self.assertEqual(documentation_gaps(pack, enabled, exported), {"missing": [], "stray": []})

    def test_markers_never_reach_the_model(self):
        for pack in PACKS:
            for enabled in (list(EVERYTHING), resolve(list(EVERYTHING), {"vlm"}).enabled):
                self.assertNotIn("<!--", assemble_prompt(pack, enabled))


if __name__ == "__main__":
    unittest.main()
