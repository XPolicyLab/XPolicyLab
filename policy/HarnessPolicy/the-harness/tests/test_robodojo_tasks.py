import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benches.robodojo.tasks import (
    DEFAULT_TASK,
    layout_count,
    layout_dir,
    task_catalog,
    task_settings,
    validate_layout,
    validate_layout_pack_seed,
    validate_robot_config,
    validate_task,
)

CHECKOUT = ROOT / "third_party" / "RoboDojo"
INDEX = {
    "common": {"scene_config": "default", "robot_config": "dual_x5", "data_source": "datagen"},
    "tasks": {
        "match_and_pick_from_conveyor": {"data_source": "teleop", "scene_config": "conveyor"},
        "make_kong": {"robot_config": "dual_x5_and_franka_competition"},
        "odd_robot": {"robot_config": "single_franka"},
    },
}


class RoboDojoTaskCatalogTests(unittest.TestCase):
    def test_catalog_lists_task_configs_and_skips_shared_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "task" / "RoboDojo" / "config"
            config.mkdir(parents=True)
            for name in ("b_random.yml", "_task.yml", "a.yml", "notes.txt"):
                (config / name).write_text("")
            self.assertEqual(task_catalog(directory), ["a", "b_random"])

    def test_catalog_requires_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "task config directory"):
                task_catalog(directory)

    def test_validate_task_rejects_unknown_names_with_a_hint(self):
        catalog = ["stack_blocks", "stack_bowls"]
        self.assertEqual(validate_task("stack_bowls", catalog), "stack_bowls")
        with self.assertRaisesRegex(ValueError, "Did you mean stack_bowls"):
            validate_task("stack_bowl", catalog)
        for bad in ("_task", "", None):
            with self.assertRaises(ValueError):
                validate_task(bad, catalog)

    def test_settings_follow_task_overrides_then_common(self):
        self.assertEqual(task_settings("stack_bowls", INDEX)["scene_config"], "default")
        self.assertEqual(task_settings("stack_bowls", INDEX)["robot_config"], "dual_x5")
        conveyor = task_settings("match_and_pick_from_conveyor", INDEX)
        self.assertEqual((conveyor["scene_config"], conveyor["data_source"]), ("conveyor", "teleop"))
        self.assertEqual(task_settings("make_kong", INDEX)["robot_config"], "dual_x5_and_franka_competition")
        self.assertEqual(task_settings("anything", {})["robot_config"], "dual_x5")
        with self.assertRaisesRegex(ValueError, "robot config"):
            task_settings("odd_robot", INDEX)

    def test_robot_and_layout_pack_validation(self):
        validate_robot_config("dual_x5")
        validate_robot_config("dual_x5_and_franka_competition")
        with self.assertRaises(ValueError):
            validate_robot_config("franka")
        self.assertEqual(validate_layout_pack_seed(2), 2)
        for bad in (3, -1, True, "0"):
            with self.assertRaises(ValueError):
                validate_layout_pack_seed(bad)

    def test_layout_count_matches_seed_manager_pattern(self):
        with tempfile.TemporaryDirectory() as directory:
            layouts = layout_dir(directory, "arx_x5", 0)
            layouts.mkdir(parents=True)
            for name in ("stack_bowls_0.json", "stack_bowls_1.json", "stack_bowls_10.json",
                         "stack_bowls_random_0.json", "stack_bowls_x.json"):
                (layouts / name).write_text("{}")
            self.assertEqual(layout_count(layouts, "stack_bowls"), 3)
            self.assertEqual(layout_count(layouts, "stack_bowls_random"), 1)
            self.assertEqual(layout_count(layouts, "push_T"), 0)
        with self.assertRaisesRegex(RuntimeError, "layout directory"):
            layout_count(Path(directory) / "missing", "stack_bowls")

    def test_validate_layout_bounds_the_seed(self):
        self.assertEqual(validate_layout(0, 3), 0)
        self.assertEqual(validate_layout(2, 3), 2)
        for seed in (3, -1, True, 1.0):
            with self.assertRaises(ValueError):
                validate_layout(seed, 3)
        with self.assertRaisesRegex(ValueError, "no evaluation layouts"):
            validate_layout(0, 0)

    @unittest.skipUnless((CHECKOUT / "task" / "RoboDojo" / "config").is_dir(), "RoboDojo checkout absent")
    def test_local_checkout_has_every_task_with_layouts(self):
        catalog = task_catalog(CHECKOUT)
        self.assertIn(DEFAULT_TASK, catalog)
        self.assertEqual(len(catalog), 54)
        layouts = layout_dir(CHECKOUT, "arx_x5", 0)
        if not layouts.is_dir():
            self.skipTest("RoboDojo assets absent")
        missing = [task for task in catalog if layout_count(layouts, task) < 1]
        self.assertEqual(missing, [])


class RoboDojoGeneralImportTests(unittest.TestCase):
    def test_general_import_does_not_launch_isaac_and_checks_options(self):
        forbidden = {"isaacsim", "isaaclab", "torch", "omni", "carb", "curobo", "omegaconf"}
        before = set(sys.modules)
        import benches.robodojo.general as general

        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".", 1)[0] in forbidden for name in imported))
        with self.assertRaisesRegex(RuntimeError, "HARNESS_ROBODOJO_RUNTIME"):
            general.create_environment(task="push_T")
        previous = os.environ.get("HARNESS_ROBODOJO_RUNTIME")
        os.environ["HARNESS_ROBODOJO_RUNTIME"] = "1"
        try:
            with self.assertRaisesRegex(ValueError, "Unknown RoboDojo options"):
                general.create_environment(tasks="push_T")
            with self.assertRaisesRegex(RuntimeError, "RoboDojo checkout not found"):
                general.create_environment(checkout=str(ROOT / "third_party" / "missing-robodojo"))
            with tempfile.TemporaryDirectory() as directory:
                fake = Path(directory)
                (fake / "src" / "eval_client").mkdir(parents=True)
                (fake / "src" / "eval_client" / "eval_env.py").write_text("")
                config = fake / "task" / "RoboDojo" / "config"
                config.mkdir(parents=True)
                (config / "stack_bowls.yml").write_text("")
                with self.assertRaisesRegex(ValueError, "Unknown RoboDojo task"):
                    general.create_environment(checkout=directory, task="stack_bowl")
                with self.assertRaisesRegex(ValueError, "layout_pack_seed"):
                    general.create_environment(checkout=directory, layout_pack_seed=7)
                layouts = layout_dir(fake, "arx_x5", 0)
                layouts.mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, "No arx_x5/0 evaluation layouts"):
                    general.create_environment(checkout=directory)
        finally:
            if previous is None:
                os.environ.pop("HARNESS_ROBODOJO_RUNTIME", None)
            else:
                os.environ["HARNESS_ROBODOJO_RUNTIME"] = previous
        imported = set(sys.modules) - before
        self.assertFalse(any(name.split(".", 1)[0] in forbidden for name in imported))


def _load_parity():
    spec = importlib.util.spec_from_file_location("robodojo_parity", ROOT / "scripts" / "robodojo_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(directory, *, joint=0.0, extra_state=None, reward=0.0, environment="robodojo-isaac51"):
    directory.mkdir(parents=True)

    def observation(step, offset):
        state = {"task": "stack_bowls", "seed": 0, "robot_config": "dual_x5",
                 "joint_position": [joint + offset] * 14, "physics_device": "cuda:0"}
        state.update(extra_state or {})
        return {"episode_id": "e", "step": step, "instruction": "Stack the three bowls together.",
                "state": state, "images": {}}

    events = [
        {"seq": 0, "event": "reset", "observation": observation(0, 0.0),
         "tools": [{"name": "joint_position"}], "environment": environment},
        {"seq": 1, "event": "decision", "step": 0,
         "action": {"name": "joint_position", "arguments": {"position": [joint] * 14}}},
        {"seq": 2, "event": "transition", "step": 0, "transition": {
            "observation": observation(1, 0.05), "reward": reward, "success": False,
            "terminated": False, "truncated": False, "info": {"executed_position": [joint] * 14},
            "videos": {}}},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (directory / "summary.json").write_text(json.dumps(
        {"task": "stack_bowls", "seed": 0, "status": "budget_exhausted", "success": False, "steps": 1}))


class RoboDojoParityTests(unittest.TestCase):
    def test_matching_runs_ignore_general_worker_extras(self):
        parity = _load_parity()
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a", Path(directory) / "b"
            _write_run(left)
            _write_run(right, joint=1e-6, extra_state={"scene_config": "default", "layout_count": 55},
                       environment="robodojo-isaac51-general")
            result = parity.compare(left, right, 1e-4)
        self.assertTrue(result["match"], result["differences"])
        self.assertAlmostEqual(result["max_abs_diff"]["reset.state.joint_position"], 1e-6)

    def test_divergent_state_and_reward_are_reported(self):
        parity = _load_parity()
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a", Path(directory) / "b"
            _write_run(left)
            _write_run(right, joint=0.01, reward=0.15, extra_state={"robot_config": "other"})
            result = parity.compare(left, right, 1e-4)
        self.assertFalse(result["match"])
        where = {item["where"] for item in result["differences"]}
        self.assertIn("reset.state.joint_position", where)
        self.assertIn("reset.state.robot_config", where)
        self.assertIn("transition[0].reward", where)
        self.assertIn("decision[0].position", where)


if __name__ == "__main__":
    unittest.main()
