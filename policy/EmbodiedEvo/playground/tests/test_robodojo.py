"""Semantic checks for the RoboDojo adapter without importing Isaac Sim."""

import copy
import json
import multiprocessing
import os
import select
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from environments import Environment
from orchestrator.config import config_dict, config_from_dict, load_task
from orchestrator.files import test_seeds as sample_test_seeds
from orchestrator.session import Session, serve
from orchestrator.session_runner import create_workspace, prepare_inputs
from simulators.primitives import rgb_images
from simulators.robodojo import RoboDojoTask
from simulators.robodojo_api import RoboDojoLowLevelApi
from simulators.robodojo_layouts import available_scenes, decode_scene


class ArticulationResetTests(unittest.TestCase):
    def test_initial_reset_is_untouched_and_later_resets_restore_original_joints(self):
        class Object:
            def __init__(self, position):
                self.position = np.array([position])
                self.initial_joint_positions = np.array([0.0002])

            def initialize(self):
                self.initial_joint_positions = self.position.copy()

            def apply_saved_pose(self):
                self.position = self.initial_joint_positions.copy()

        objects = [Object(0.005), Object(0.007)]
        native = SimpleNamespace(scene_manager=SimpleNamespace(
            _articulation_objects=[dict(enumerate(objects))]))
        repair = SimpleNamespace(native=native, _restoring_scene=False)
        with RoboDojoTask._preserve_reset_articulation_defaults(repair):
            for obj in objects:
                obj.initialize()
                obj.apply_saved_pose()
        np.testing.assert_allclose([obj.position[0] for obj in objects], [0.005, 0.007])
        repair._restoring_scene = True
        for _ in range(2):
            for obj in objects:
                obj.position[:] = -0.005
            with RoboDojoTask._preserve_reset_articulation_defaults(repair):
                for obj in objects:
                    obj.initialize()
                    obj.apply_saved_pose()
            np.testing.assert_allclose([obj.position[0] for obj in objects], [0.005, 0.007])
            self.assertTrue(all("initialize" not in obj.__dict__ for obj in objects))


class Tensor(np.ndarray):
    """Small tensor stand-in; semantic tests do not depend on Torch or Isaac."""

    def clone(self):
        return self.copy()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class Articulation:
    def __init__(self):
        positions = np.array([[0.0] * 6 + [0.044, 0.044]]).view(Tensor)
        velocities = np.zeros_like(positions)
        self.data = SimpleNamespace(
            default_joint_pos=positions,
            default_joint_vel=velocities,
            joint_pos=positions.copy(),
            joint_vel=velocities.copy(),
        )
        self.position_target = positions.copy()
        self.velocity_target = velocities.copy()
        self.history = 0

    def reset(self):
        self.history = 0

    def write_joint_state_to_sim(self, position, velocity):
        self.data.joint_pos[:] = position
        self.data.joint_vel[:] = velocity


class RobotManager:
    def __init__(self):
        self.robot_list = [SimpleNamespace(robot_type="arm") for _ in range(2)]
        self.robot_key = [Articulation(), Articulation()]

    def set_robot_init_pose(self):
        # Deliberately match native reset: targets alone do not reset physics.
        for arm in self.robot_key:
            arm.position_target = arm.data.default_joint_pos.clone()
            arm.velocity_target = arm.data.default_joint_vel.clone()


class Native:
    def __init__(self):
        self.robot_manager = RobotManager()
        self.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(
            object_records_by_type={"Articulation": SimpleNamespace(layout_records_by_env=[[]])},
            _generate_object_paths=lambda *args, **kwargs: ("new/path", "new_name"),
        ), _articulation_objects=[{}], _rigid_and_dynamic_objects=[{}],
            delete_scene_object=lambda obj: None)
        self.step_lim = 3
        self.episode_nums = 1
        self.take_action_cnt = [0]
        self.success = [True]  # Upstream viability sentinel, NOT completion.
        self.end_flag = [False]
        self.unstable_envs = set()
        self.obs_manager = SimpleNamespace(instruction=["native instruction"])
        self.reward_registrations = 0
        self.obs = {
            "instruction": "native instruction",
            "vision": {"cam_head": {"color": np.zeros((2, 2, 3), dtype=np.uint8)}},
            "state": {
                "left_arm_joint_state": [0.0] * 6,
                "right_arm_joint_state": [0.0] * 6,
                "left_ee_joint_state": [1.0],
                "right_ee_joint_state": [1.0],
                "left_ee_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "right_ee_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            },
        }

    def reset(self, **kwargs):
        self.robot_manager.set_robot_init_pose()
        self.step_lim = 3
        self.take_action_cnt = [0]
        self.success, self.end_flag = [True], [False]
        self.obs_manager.instruction[0] = "regenerated instruction"
        self.obs["instruction"] = "regenerated instruction"

    def run_reward(self):
        self.reward_registrations += 1

    def get_obs(self):
        obs = copy.deepcopy(self.obs)
        for side, arm in zip(("left", "right"), self.robot_manager.robot_key, strict=True):
            obs["state"][side + "_arm_joint_state"] = arm.data.joint_pos[0, :6].copy()
        return obs

    def validate_action_dict(self, action):
        pass

    def get_action_type(self, action):
        return "ee" if any(k.endswith("ee_pose") for k in action) else "joint"

    def take_action(self, action):
        if self.take_action_cnt[0] >= self.step_lim or self.end_flag[0]:
            return
        self.take_action_cnt[0] += 1
        if self.take_action_cnt[0] >= self.step_lim:
            self.success, self.end_flag = [False], [True]

    def close(self):
        pass


def task(*, step_limit=3):
    env = RoboDojoTask.__new__(RoboDojoTask)
    Environment.__init__(env)
    env.native = Native()
    env.scene_seed = env.ordinal = 0
    env.native_step_limit = env.max_steps = env.native.step_lim = step_limit
    env._failure = None
    env._physx_monitor = None
    env.initialize(RoboDojoLowLevelApi(env))
    env.execution_phase = "playground"
    env.reset(seed=0)
    return env


def joint_action(env):
    return {k: v for k, v in env.public_observation()["state"].items() if k.endswith("joint_state")}


def serve_budget_fixture(cfg, run, address, phase, limit, ready_fd):
    with patch("simulators.robodojo.robodojo_cwd", nullcontext):
        env = task(step_limit=limit)
        env.close = lambda: None
        with (
            patch("simulators.backend.build", return_value=env),
            patch("orchestrator.session.write_videos", return_value={}),
        ):
            serve(cfg, run, address, phase=phase, seed=0, budget=100,
                  seconds=None, execution_seconds=10, ready_fd=ready_fd)


class RoboDojoTests(unittest.TestCase):
    def test_retired_child_colliders_disable_without_removing_rigid_registration(self):
        env = task()
        env._restoring_scene = True
        env.disable_retired_collisions = True
        manager = env.native.scene_manager
        events = []

        def object_tree(name):
            shape = SimpleNamespace(enabled=True, rigid_registered=True)
            shape.HasAPI = lambda schema: True
            shape.CreateCollisionEnabledAttr = lambda: SimpleNamespace(Set=lambda value: setattr(shape, "enabled", value))
            prim = SimpleNamespace(IsValid=lambda: True, shapes=[shape])
            return SimpleNamespace(prim_path=name, shape=shape,
                stage=SimpleNamespace(GetPrimAtPath=lambda path: prim),
                relocate_offscreen=lambda: events.append(("relocate", name)))

        rigid, geometry, unrelated = [object_tree(name) for name in ("rigid", "geometry", "unrelated")]
        manager._rigid_and_dynamic_objects = [{"old": rigid}]
        manager._geometry_objects = [{"old": geometry}]
        delete = manager.delete_scene_object = lambda obj: events.append(("delete", obj.prim_path))
        pxr = SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda prim: prim.shapes),
                             UsdPhysics=SimpleNamespace(CollisionAPI=lambda shape: shape))
        with patch.dict(sys.modules, {"pxr": pxr}):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                with env._retire_reset_rigid_bodies():
                    for obj in (rigid, geometry, unrelated):
                        manager.delete_scene_object(obj)
                    raise RuntimeError("reset failed")
        self.assertIs(manager.delete_scene_object, delete)
        self.assertEqual([obj.shape.enabled for obj in (rigid, geometry, unrelated)], [False, False, True])
        self.assertTrue(all(obj.shape.rigid_registered for obj in (rigid, geometry, unrelated)))
        self.assertEqual(events, [("relocate", "rigid"), ("delete", "rigid"),
                                  ("delete", "geometry"), ("delete", "unrelated")])

    def test_fixed_rigid_reset_keeps_lookup_and_physics_identity(self):
        env = task()
        env._restoring_scene = True
        env.reuse_reset_rigid_bodies = True
        manager = env.native.scene_manager
        layout = manager.layout_manager
        layout.object_records_by_type["Rigid"] = SimpleNamespace(layout_records_by_env=[[
            {"prim_path": "/coin_0_1", "inst_name": "coin_0_1"},
            {"prim_path": "/coin_0_2", "inst_name": "coin_0_2"},
        ]])
        events = []
        manager.clear_scene_objects = lambda env_id, exclude_types=None: events.append(("clear", exclude_types))
        manager.spawn_scene_objects = lambda env_id, exclude_types=None: events.append(("spawn", exclude_types))
        manager.relocate_stale_objects = lambda env_id, obj_types=None: events.append(("relocate", obj_types))
        generate, clear = layout._generate_object_paths, manager.clear_scene_objects
        with self.assertRaisesRegex(RuntimeError, "native reset failed"):
            with env._preserve_reset_articulation_names(), env._retire_reset_rigid_bodies():
                for name in ("coin_0_1", "coin_0_2"):
                    self.assertEqual(layout._generate_object_paths(0, "coin", 0, type="rigid"),
                                     ("/" + name, name))
                manager.relocate_stale_objects(0, ["articulation"])
                manager.clear_scene_objects(0, ["articulation"])
                manager.spawn_scene_objects(0, ["articulation"])
                manager.relocate_stale_objects(0)
                raise RuntimeError("native reset failed")
        self.assertEqual(events, [("relocate", ["articulation", "rigid"]),
                                  ("clear", ["articulation", "rigid"]),
                                  ("spawn", ["articulation", "rigid"]), ("relocate", None)])
        self.assertIs(layout._generate_object_paths, generate)
        self.assertIs(manager.clear_scene_objects, clear)
        env._restoring_scene = False
        with env._preserve_reset_articulation_names(), env._retire_reset_rigid_bodies():
            self.assertEqual(layout._generate_object_paths(0, "coin", 0, type="rigid"),
                             ("new/path", "new_name"))
            manager.clear_scene_objects(0)
        self.assertEqual(events[-1], ("clear", None))

    def setUp(self):
        recipe_switch = patch.dict(os.environ, {"EMBODIEDEVO_USE_RECIPE": "0"})
        recipe_switch.start()
        self.addCleanup(recipe_switch.stop)
        self.cwd = patch("simulators.robodojo.robodojo_cwd", nullcontext)
        self.cwd.start()
        self.addCleanup(self.cwd.stop)

    def test_constructor_defaults_to_native_limit(self):
        task_limits = {"stack_bowls": 800, "general_pickup": 200, "fasten_screws": 1900}

        def create(env):
            env.native = Native()
            env.native.step_lim = task_limits[env.task_name]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("simulators.robodojo.ROOT", Path(directory)),
            patch("simulators.robodojo.decode_scene", return_value=(0, 0, Path("fixture"))),
            patch.object(RoboDojoTask, "_create_native", create),
        ):
            for task_name, limit in task_limits.items():
                with self.subTest(task=task_name):
                    env = RoboDojoTask(task_name, seed=0)
                    try:
                        self.assertTrue(env.official_action_limit)
                        self.assertEqual(env.native.step_lim, limit)
                        self.assertEqual(env.max_steps, limit)
                        env.reset(seed=0)
                        self.assertEqual(env.native.step_lim, limit)
                    finally:
                        env.close()

    def test_success_on_last_allowed_action_is_not_truncated(self):
        env = task()
        action = joint_action(env)
        for _ in range(2):
            env.control_step(action)
        take_action = env.native.take_action

        def succeed(action):
            take_action(action)
            env.native.success[0] = True

        env.native.take_action = succeed
        self.assertEqual(env.control_step(action)[1:4], (1.0, True, False))
        self.assertEqual(env.control_steps, 3)

    def test_native_failure_before_limit_is_termination(self):
        env = task()

        def fail(action):
            env.native.take_action_cnt[0] += 1
            env.native.success, env.native.end_flag = [False], [True]

        env.native.take_action = fail
        self.assertEqual(env.control_step(joint_action(env))[1:4], (0.0, True, False))
        self.assertEqual(env.control_steps, 1)

    def test_partial_score_survives_native_cutoff(self):
        env = task()
        env.native.get_score = lambda: None
        env.native.reward_manager = SimpleNamespace(get_score=lambda: [40.0])
        env._scoring_source_hashes = {}
        for _ in range(5):
            env.control_step(joint_action(env))
        score = env.private_scoring_snapshot()
        self.assertTrue(env.truncated)
        self.assertEqual(score["score"], 40)
        self.assertTrue(score["budget_matches_official"])
        self.assertEqual(score["native_action_limit"], 3)
        self.assertEqual(score["native_steps"], 3)

    def test_reset_relocates_discarded_rigid_bodies_before_native_hide(self):
        env = task()
        manager = env.native.scene_manager
        events = []
        rigid = SimpleNamespace(relocate_offscreen=lambda: events.append("relocate"))
        other = SimpleNamespace()
        manager._rigid_and_dynamic_objects = [{"old": rigid}]
        manager.delete_scene_object = lambda obj: events.append(
            "hide_rigid" if obj is rigid else "delete_other"
        )
        original = manager.delete_scene_object
        env._restoring_scene = True
        with env._retire_reset_rigid_bodies():
            manager.delete_scene_object(rigid)
            manager.delete_scene_object(other)
        self.assertEqual(events, ["relocate", "hide_rigid", "delete_other"])
        self.assertIs(manager.delete_scene_object, original)
        events.clear()
        env._restoring_scene = False
        with env._retire_reset_rigid_bodies():
            manager.delete_scene_object(rigid)
        self.assertEqual(events, ["hide_rigid"])

    def test_native_viability_is_not_reported_as_success(self):
        env = task()
        obs, reward, terminated, truncated, info = env.control_step(joint_action(env))
        self.assertTrue(env.native.success[0])
        self.assertEqual(
            (reward, terminated, truncated, info["success"]), (0.0, False, False, False)
        )
        obs["state"]["left_arm_joint_state"][0] = 12
        self.assertEqual(env.public_observation()["state"]["left_arm_joint_state"][0], 0)

    def test_success_requires_official_end_flag(self):
        env = task()
        env.native.take_action = lambda action: env.native.end_flag.__setitem__(0, True)
        result = env.control_step(joint_action(env))
        self.assertEqual(result[1:4], (1.0, True, False))
        before = env.control_steps
        env.control_step(joint_action(env))
        self.assertEqual(env.control_steps, before)

    def test_reset_renews_native_budget_preserving_telemetry_and_instruction(self):
        env = task()
        instruction = env.task_prompt()
        env.control_step(joint_action(env))
        env.step("remember = 7")
        env.reset_from_program()
        self.assertEqual(env.control_steps, 1)
        self.assertEqual(env.native.take_action_cnt, [0])
        self.assertEqual(env.native.step_lim, 3)
        self.assertEqual(env.max_steps - env.control_steps, 3)
        self.assertEqual(env.task_prompt(), instruction)
        self.assertEqual(env.public_observation()["instruction"], instruction)
        self.assertTrue(env.step("assert remember == 7")[-1]["ok"])
        for _ in range(10):
            env.control_step(joint_action(env))
        self.assertEqual(env.control_steps, 4)
        self.assertEqual(env.native.take_action_cnt, [3])
        self.assertTrue(env.truncated)
        self.assertFalse(env.success)
        env.reset_from_program()
        self.assertEqual(env.max_steps - env.control_steps, 3)
        env.control_step(joint_action(env))
        self.assertEqual(env.control_steps, 5)
        self.assertEqual(env.native.take_action_cnt, [1])
        self.assertEqual(env.native.step_lim, 3)
        self.assertFalse(env.truncated)

    def test_reset_restores_displaced_joints_velocities_grippers_and_targets(self):
        env = task()
        manager = env.native.robot_manager
        original_reset = manager.set_robot_init_pose
        for arm in manager.robot_key:
            arm.data.joint_pos[:] = [0, 0, 0.793, -2.124, 0.860, 0.410, 0, 0]
            arm.data.joint_vel[:] = 2.0
            arm.position_target[:] = 1.0
            arm.velocity_target[:] = 3.0
            arm.history = 9
        obs, _ = env.reset_from_program()
        for side, arm in zip(("left", "right"), manager.robot_key, strict=True):
            np.testing.assert_array_equal(arm.data.joint_pos, arm.data.default_joint_pos)
            np.testing.assert_array_equal(arm.data.joint_vel, arm.data.default_joint_vel)
            np.testing.assert_array_equal(arm.position_target, arm.data.default_joint_pos)
            np.testing.assert_array_equal(arm.velocity_target, arm.data.default_joint_vel)
            np.testing.assert_array_equal(obs["state"][side + "_arm_joint_state"], [0] * 6)
            self.assertEqual(arm.history, 0)
        self.assertEqual(manager.set_robot_init_pose, original_reset)

    def test_reset_restores_state_before_native_reload_can_step_physics(self):
        env = task()
        native_reset = env.native.reset
        arm = env.native.robot_manager.robot_key[1]
        arm.data.joint_pos[:] = 5518.0
        arm.data.joint_vel[:] = 1378.0
        arm.position_target[:] = 100.0

        def reload(**kwargs):
            # Native scene reload takes a physics step before robot.reset().
            np.testing.assert_array_equal(arm.data.joint_pos, arm.data.default_joint_pos)
            np.testing.assert_array_equal(arm.data.joint_vel, arm.data.default_joint_vel)
            np.testing.assert_array_equal(arm.position_target, arm.data.default_joint_pos)
            native_reset(**kwargs)

        env.native.reset = reload
        env.reset_from_program()

    def test_first_reset_does_not_access_uninitialized_robot_views(self):
        env = task()
        env._restoring_scene = False
        with patch.object(env, "_check_reset_robot_state") as check:
            with env._restore_robot_state_on_reset():
                check.assert_not_called()
                env.native.robot_manager.set_robot_init_pose()
                check.assert_called_once()

    def test_reset_keeps_retained_articulation_lookup_identity(self):
        env = task()
        layout = env.native.scene_manager.layout_manager
        records = layout.object_records_by_type["Articulation"].layout_records_by_env
        records[0] = [
            {"inst_name": "spring_button_3_17", "prim_path": "/button17"},
            {"inst_name": "spring_button_3_18", "prim_path": "/button18"},
        ]
        original = layout._generate_object_paths
        native_reset = env.native.reset

        def reload(**kwargs):
            for item in records[0]:
                path, name = layout._generate_object_paths(
                    0, "spring_button", 3, type="articulation"
                )
                self.assertEqual((path, name), (item["prim_path"], item["inst_name"]))
            self.assertEqual(layout._generate_object_paths(0, "cube", 3, type="rigid"),
                             ("new/path", "new_name"))
            native_reset(**kwargs)

        env.native.reset = reload
        for _ in range(2):
            env.reset_from_program()
            self.assertEqual(layout._generate_object_paths, original)

    def test_failed_reset_is_fatal_even_if_agent_catches_it(self):
        for field in ("joint_pos", "joint_vel"):
            with self.subTest(field=field):
                env = task()
                manager = env.native.robot_manager
                original_reset = manager.set_robot_init_pose
                arm = manager.robot_key[0]
                getattr(arm.data, field)[:] = 1.0
                with (
                    patch.object(arm, "write_joint_state_to_sim"),
                    self.assertRaisesRegex(RuntimeError, "simulator failed"),
                ):
                    env.step("try:\n    reset()\nexcept Exception:\n    pass")
                self.assertTrue(env.truncated)
                self.assertIn(field, str(env._failure))
                self.assertEqual(manager.set_robot_init_pose, original_reset)

    def test_native_settling_can_move_restored_joints(self):
        env = task()
        native_reset = env.native.reset
        arm = env.native.robot_manager.robot_key[1]

        def settle(**kwargs):
            native_reset(**kwargs)
            arm.data.joint_pos[0, 6:] -= 0.0078
            arm.data.joint_vel[0, 6:] = 0.02

        env.native.reset = settle
        env.reset_from_program()
        self.assertIsNone(env._failure)
        # The physical outcome is retained, not teleported back after settling.
        np.testing.assert_allclose(arm.data.joint_pos[0, 6:], [0.0362, 0.0362])

    def test_bad_state_write_is_rejected_before_settling_can_hide_it(self):
        env = task()
        arm = env.native.robot_manager.robot_key[0]
        arm.data.joint_pos[0, 0] = 0.005
        native_reset = env.native.reset
        settled = []

        def settle(**kwargs):
            native_reset(**kwargs)
            settled.append(True)
            arm.data.joint_pos[:] = arm.data.default_joint_pos

        env.native.reset = settle
        with patch.object(arm, "write_joint_state_to_sim"), self.assertRaisesRegex(
            RuntimeError, "failed to restore robot 0 joint_pos"
        ):
            env.reset_from_program()
        self.assertEqual(settled, [])

    def test_nonfinite_state_after_settling_is_rejected(self):
        for field in ("joint_pos", "joint_vel"):
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    env = task()
                    native_reset = env.native.reset

                    def settle(
                        *, env=env, native_reset=native_reset, field=field, value=value, **kwargs
                    ):
                        native_reset(**kwargs)
                        getattr(env.native.robot_manager.robot_key[0].data, field)[0, 0] = value

                    env.native.reset = settle
                    with self.assertRaisesRegex(RuntimeError, "non-finite robot 0 " + field):
                        env.reset_from_program()

    def test_both_phases_stop_at_each_native_task_limit(self):
        for phase in ("playground", "test"):
            with self.subTest(phase=phase):
                for limit in (3, 200, 800, 1900):
                    env = task(step_limit=limit)
                    env.execution_phase = phase
                    action = joint_action(env)
                    for _ in range(limit - 1):
                        env.control_step(action)
                    self.assertFalse(env.done)
                    result = env.control_step(action)
                    self.assertEqual(result[1:4], (0.0, False, True))
                    self.assertTrue(env.native.end_flag[0])
                    with patch.object(env.native, "take_action") as native_action:
                        for _ in range(3):
                            env.control_step(action)
                        native_action.assert_not_called()
                    self.assertEqual(env.control_steps, limit)
                    self.assertEqual(env.native.take_action_cnt, [limit])
                    self.assertFalse(env.success)

    def test_native_budget_is_shared_across_100_submissions(self):
        cfg = load_task("RoboDojo")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_inputs(cfg, root)
            for phase in ("playground", "test"):
                with self.subTest(phase=phase):
                    run = root / phase
                    create_workspace(cfg, run, root / "inputs", phase=phase)
                    session = Session(
                        task(), run, phase=phase, seed=0, budget=100,
                        seconds=None, execution_seconds=10,
                    )
                    code = (
                        "state = get_observation()['state']\n"
                        "action = {k: v for k, v in state.items() "
                        "if k.endswith('joint_state')}\n"
                        "for i in range(801):\n    step(action)"
                    )

                    def submit(source, session=session):
                        return session.handle({
                            "op": "exec",
                            "files": [{"path": "submission/solution.py", "code": source}],
                        })

                    with patch("orchestrator.session.write_videos", return_value={}):
                        result = submit(code)
                        self.assertTrue(result["ok"], result)
                        self.assertEqual(result["native_steps_used"], 3)
                        self.assertEqual(result["native_steps_remaining"], 0)
                        self.assertTrue(result["truncated"])
                        self.assertEqual(result["executions_remaining"], 99)
                        for _ in range(99):
                            result = submit("step(action)")
                            self.assertTrue(result["ok"], result)
                        with self.assertRaisesRegex(ValueError, "budget exhausted"):
                            submit("step(action)")
                    status = session.handle({"op": "status"})
                    self.assertEqual(status["native_steps_used"], 3)
                    self.assertEqual(status["native_steps_remaining"], 0)
                    self.assertEqual(status["executions_remaining"], 0)
                    self.assertTrue(status["truncated"])
                    json.dumps(status, allow_nan=False)
                    self.assertTrue(session.handle({"op": "finish"})["closed"])

    def test_playground_reset_renews_actions_but_never_submission_or_time_budget(self):
        cfg = load_task("RoboDojo")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            prepare_inputs(cfg, run)
            create_workspace(cfg, run, run / "inputs", phase="playground")
            session = Session(
                task(), run, phase="playground", seed=0, budget=4,
                seconds=600, execution_seconds=10,
            )
            started = session.started
            code = (
                "s = get_observation()['state']\n"
                "a = {k: v for k, v in s.items() if k.endswith('joint_state')}\n"
                "for i in range(5):\n    step(a)"
            )
            request = {"op": "exec", "files": [{"path": "x.py", "code": code}]}
            with patch("orchestrator.session.write_videos", return_value={}):
                for attempt in (1, 2):
                    response = session.handle(request)
                    self.assertTrue(response["truncated"])
                    self.assertEqual(response["native_steps_used"], attempt * 3)
                    self.assertEqual(response["native_steps_remaining"], 0)
                    response = session.handle({"op": "reset"})
                    self.assertFalse(response["truncated"])
                    self.assertEqual(response["native_steps_remaining"], 3)
                    self.assertEqual(response["executions_used"], attempt * 2)
                    self.assertEqual(response["executions_remaining"], 4 - attempt * 2)
                    self.assertEqual(session.started, started)
                with self.assertRaisesRegex(ValueError, "budget exhausted"):
                    session.handle({"op": "reset"})
                with self.assertRaisesRegex(ValueError, "budget exhausted"):
                    session.handle(request)
            self.assertEqual(session.env.control_steps, 6)

    def test_phase_manuals_describe_action_and_execution_budgets(self):
        cfg = load_task("RoboDojo")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_inputs(cfg, root)
            for phase in ("playground", "test"):
                workspace = create_workspace(cfg, root / phase, root / "inputs", phase=phase)
                instruction = (workspace / "instruction.md").read_text()
                self.assertIn("official task action budget", instruction)
                self.assertIn("execution slot", instruction)
                self.assertNotIn("no step-count limit", instruction)
                if phase == "playground":
                    self.assertIn("reset()", instruction)
                    self.assertIn("replenishes only the native action allowance", instruction)
                    self.assertIn("total execution budget is never reset", instruction)
                else:
                    self.assertIn("Reset is forbidden", instruction)

    def test_runtime_instruction_has_actual_task_limits_and_feedback_tracks_remaining(self):
        for task_path, env_task, phase, limit, budget in (
            ("RoboDojo", "stack_bowls", "playground", 800, 100),
            ("RoboDojo", "general_pickup", "test", 200, 100),
            ("tasks_baseline/RoboDojo", "fasten_screws", "test", 1900, 17),
        ):
            with self.subTest(task=task_path, phase=phase), tempfile.TemporaryDirectory() as tmp:
                cfg = load_task(task_path, env_task=env_task)
                root = Path(tmp)
                prepare_inputs(cfg, root)
                source = root / "inputs" / (
                    "instruction.md" if cfg.baseline else f"{phase}.md"
                )
                original = source.read_bytes()
                run = root / "episode"
                workspace = create_workspace(cfg, run, root / "inputs", phase=phase)
                session = Session(
                    task(step_limit=limit), run, phase=phase, seed=0, budget=budget,
                    seconds=None, execution_seconds=10,
                )
                instruction = (workspace / "instruction.md").read_text()
                self.assertNotIn("{{", instruction)
                self.assertIn(f"This task permits {limit} native action steps", instruction)
                self.assertIn(f"| {budget} | {budget} |", instruction)
                self.assertIn(f"| {limit} | {limit} |", instruction)
                self.assertIn("executions_remaining", instruction)
                self.assertIn("native_steps_remaining", instruction)
                self.assertIn("native_action_limit", instruction)
                self.assertIn("execution_budget", instruction)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(session.status()["native_steps_remaining"], limit)
                self.assertEqual(session.status()["executions_remaining"], budget)
                self.assertEqual(session.status()["execution_budget"], budget)
                self.assertEqual(session.status()["native_action_limit"], limit)
                with patch("orchestrator.session.write_videos", return_value={}):
                    response = session.handle({"op": "exec", "files": [{
                        "path": "x.py", "code": (
                            "s = get_observation()['state']\n"
                            "a = {k: v for k, v in s.items() if k.endswith('joint_state')}\n"
                            "for i in range(5):\n    step(a)"
                        ),
                    }]})
                    self.assertEqual(response["executions_remaining"], budget - 1)
                    self.assertEqual(response["native_steps_remaining"], limit - 5)
                    self.assertEqual(response["native_action_limit"], limit)
                    self.assertEqual(response["execution_budget"], budget)
                    if phase == "playground":
                        response = session.handle({"op": "reset"})
                        self.assertEqual(response["executions_remaining"], budget - 2)
                        self.assertEqual(response["native_steps_remaining"], limit)
                        self.assertEqual(response["native_action_limit"], limit)
                        self.assertEqual(response["execution_budget"], budget)
                # Startup instructions stay immutable; current counts come from feedback.
                self.assertEqual((workspace / "instruction.md").read_text(), instruction)
                metadata = json.loads((run / "instruction-budget.json").read_text())
                self.assertEqual(metadata["native_action_limit"], limit)
                self.assertEqual(metadata["execution_budget"], budget)

    def test_instruction_rejects_missing_native_limit_before_agent_start(self):
        from orchestrator.session import bind_instruction_budgets

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            path = workspace / "instruction.md"
            text = "Actions: {{NATIVE_ACTION_LIMIT}}; executions: {{EXECUTION_BUDGET}}"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "finite native action limit"):
                bind_instruction_budgets(workspace, SimpleNamespace(max_steps=None), 100)
            self.assertEqual(path.read_text(), text)

    def test_workspace_status_cli_returns_live_task_specific_budgets(self):
        for phase, limit in (("playground", 200), ("test", 800)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = load_task("RoboDojo")
                prepare_inputs(cfg, root)
                run = root / "episode"
                workspace = create_workspace(cfg, run, root / "inputs", phase=phase)
                address = root / "session.sock"
                read_fd, write_fd = os.pipe()
                process = multiprocessing.get_context("fork").Process(
                    target=serve_budget_fixture,
                    args=(cfg, run, address, phase, limit, write_fd),
                )
                process.start()
                os.close(write_fd)

                def command(*args, ok=True, workspace=workspace, address=address):
                    result = subprocess.run(
                        [sys.executable, str(workspace / "scripts/env.py"), *args],
                        cwd=workspace, env={**os.environ, "EMBODIEDEVO_SOCKET": str(address)},
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0 if ok else 1, result.stderr)
                    return json.loads(result.stdout)

                try:
                    self.assertTrue(select.select([read_fd], [], [], 10)[0])
                    self.assertIn(b'"ready"', os.read(read_fd, 1000))
                    initial = command("status")
                    self.assertEqual(initial["native_action_limit"], limit)
                    self.assertEqual(initial["native_steps_remaining"], limit)
                    self.assertEqual(initial["execution_budget"], 100)
                    self.assertEqual(initial["executions_remaining"], 100)
                    (workspace / "submission/solution.py").write_text(
                        "s = get_observation()['state']\n"
                        "a = {k: v for k, v in s.items() if k.endswith('joint_state')}\n"
                        "for i in range(5):\n    step(a)\n"
                    )
                    feedback = command("exec")
                    current = command("status")
                    for data in (feedback, current):
                        self.assertEqual(data["native_action_limit"], limit)
                        self.assertEqual(data["native_steps_remaining"], limit - 5)
                        self.assertEqual(data["executions_remaining"], 99)
                    if phase == "playground":
                        command("reset")
                        current = command("status")
                        self.assertEqual(current["native_action_limit"], limit)
                        self.assertEqual(current["native_steps_remaining"], limit)
                        self.assertEqual(current["executions_remaining"], 98)
                    else:
                        self.assertIn("forbidden in Test", command("reset", ok=False)["error"])
                        current = command("status")
                        self.assertEqual(current["native_steps_remaining"], limit - 5)
                        self.assertEqual(current["executions_remaining"], 99)
                    command("finish")
                    process.join(5)
                    self.assertEqual(process.exitcode, 0)
                finally:
                    os.close(read_fd)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_reset_is_forbidden_in_test_even_through_alias(self):
        env = task()
        env.execution_phase = "test"
        registrations = env.native.reward_registrations
        result = env.step("r = reset\nr()")
        self.assertFalse(result[-1]["ok"])
        self.assertIn("PermissionError", result[-1]["stderr"])
        self.assertEqual(env.native.reward_registrations, registrations)

    def test_bad_actions_do_not_advance_or_poison_scene(self):
        env = task()
        for action in (
            {},
            {"left_arm_joint_state": [np.nan] * 6},
            {"left_arm_joint_state": [0.0] * 6},
        ):
            with self.assertRaises(ValueError):
                env.control_step(action)
        self.assertEqual(env.control_steps, 0)
        self.assertIsNone(env._failure)

    def test_native_failure_invalidates_session_even_if_program_catches_it(self):
        env = task()

        def fail(action):
            raise RuntimeError("GPU failed")

        env.native.take_action = fail
        with self.assertRaisesRegex(RuntimeError, "simulator failed"):
            env.step(
                "try:\n    s=get_observation()['state']\n"
                "    step({k:v for k,v in s.items() if k.endswith('joint_state')})\n"
                "except Exception:\n    pass"
            )

    def test_cameras_are_raw_native_color_arrays(self):
        env = task()
        self.assertEqual(list(rgb_images(env.public_observation())), ["cam_head"])
        self.assertEqual(env.render().shape, (2, 2, 3))

    def test_scene_identity_uses_native_numeric_ordinal_not_filename_suffix(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("simulators.robodojo_layouts.ROOT", Path(tmp)),
        ):
            directory = Path(tmp) / "Assets/Eval_Layout/RoboDojo/arx_x5/2"
            directory.mkdir(parents=True)
            for name in ("stack_bowls_10.json", "stack_bowls_2.json", "stack_bowls_random_0.json"):
                (directory / name).touch()
            bank, ordinal, path = decode_scene("stack_bowls", 2_000_001)
            self.assertEqual((bank, ordinal, path.name), (2, 1, "stack_bowls_10.json"))
            self.assertEqual(available_scenes("stack_bowls"), [2_000_000, 2_000_001])
            with self.assertRaises(ValueError):
                decode_scene("stack_bowls", 2_000_002)

    def test_contact_solver_profile_is_explicit_and_old_definitions_stay_compatible(self):
        cfg = load_task("RoboDojo", env_task="press_by_number")
        legacy = config_dict(cfg)
        self.assertNotIn("solve_articulation_contact_last", legacy["simulator"])
        self.assertEqual(config_from_dict(legacy), cfg)
        repaired = replace(cfg, simulator=replace(
            cfg.simulator, solve_articulation_contact_last=True,
            min_position_iterations=32, min_velocity_iterations=1,
        ))
        pinned = config_dict(repaired)
        self.assertTrue(pinned["simulator"]["solve_articulation_contact_last"])
        self.assertEqual(pinned["simulator"]["min_position_iterations"], 32)
        self.assertEqual(pinned["simulator"]["min_velocity_iterations"], 1)
        self.assertEqual(config_from_dict(pinned), repaired)
        self.assertEqual(legacy["playground"], pinned["playground"])
        self.assertEqual(legacy["test"], pinned["test"])
        self.assertEqual(legacy["agent"], pinned["agent"])

    def test_mixed_test_settings_sample_the_actual_native_layout_bank(self):
        cfg = load_task("RoboDojo", env_task="push_T")
        cfg = replace(cfg, test=replace(cfg.test, scenes=5))
        banks = {"push_T": [0, 31, 32, 33], "push_T_random": [0, 1]}
        with patch("simulators.robodojo_layouts.available_scenes", side_effect=banks.__getitem__):
            seeds = sample_test_seeds(
                cfg, settings=["standard", "standard", "standard", "random", "random"]
            )
        self.assertEqual(set(seeds[:3]), {31, 32, 33})
        self.assertEqual(set(seeds[3:]), {0, 1})
        self.assertEqual(len(set(seeds)), 5)
        with self.assertRaisesRegex(ValueError, "settings"):
            sample_test_seeds(cfg, settings=["standard"])

    def test_test_sampling_excludes_playground_and_is_finite(self):
        cfg = load_task("RoboDojo")
        cfg = replace(cfg, test=replace(cfg.test, scenes=2))
        with patch("simulators.robodojo_layouts.available_scenes", return_value=[0, 1, 2]):
            self.assertEqual(set(sample_test_seeds(cfg)), {1, 2})
            with self.assertRaises(ValueError):
                sample_test_seeds(replace(cfg, test=replace(cfg.test, scenes=3)))
