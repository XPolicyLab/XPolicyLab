"""Exercise adapter deployment entry points without importing their model environments."""
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from XPolicyLab.policy.AHA_WAM.debug_env_client import TestEnv as AhaDebugEnv
from XPolicyLab.utils.action_chunks import validate_action_batch


POLICIES = Path(__file__).resolve().parents[1] / "policy"

# These use agent RPCs, a separate runtime, or checkpoint-specific action/history formats.
CUSTOM_LOOPS = {
    "GPT_6_Astra_Direct_EEF", "JoyAI_VL", "ME_Brain_1", "Mem_0",
    "PhysicalRSI", "WallX",
}
SINGLE_ONLY = {
    "CloudRobo_WM", "LingBot_VA", "ME_Dex_1_0", "Q0_Preview", "VPP2",
    "WorldScape_Policy_2",
}


class Environment:
    deploy_cfg = {"evaluation_id": "chunk-test"}

    def __init__(self, ids=(3, 7)):
        self.ids = ids
        self.steps = []

    def is_episode_end(self):
        return len(self.steps) >= 2

    def get_running_env_idx_list(self):
        # First environment finishes one step earlier.
        return list(self.ids[1:] if len(self.ids) > 1 and self.steps else self.ids)

    def get_obs(self, env_idx=0):
        return {"env_idx": env_idx, "vision": {name: {"color": np.zeros((2, 2, 3), np.uint8)}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}}

    def get_obs_batch(self, ids):
        return [self.get_obs(i) for i in ids]

    def take_action(self, action):
        self.steps.append(action)

    def take_action_batch(self, actions, ids):
        self.steps.append((list(ids), list(actions)))


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, func_name, obs=None):
        self.calls.append(func_name)
        if func_name in {"get_action", "get_action_batch", "step"}:
            # Unexpected re-poll fails immediately, even if the loop never advances.
            return next(self.responses)


class AdapterDeployTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load deploy.py directly: some adapter package initializers also import model weights.
        cls.loops = {p.parent.name: runpy.run_path(str(p))
                     for p in sorted(POLICIES.glob("*/deploy.py"))
                     if p.parent.name not in CUSTOM_LOOPS}

    def test_empty_single_chunk_cannot_repoll_or_step(self):
        for name, module in self.loops.items():
            with self.subTest(policy=name):
                env = Environment()
                with self.assertRaises((ValueError, RuntimeError)):
                    module["eval_one_episode"](env, Client([]))
                self.assertEqual(env.steps, [])

    def test_invalid_batch_rejected_before_first_step(self):
        action = {"arm_joint_state": [0]}
        for name, module in self.loops.items():
            if name in SINGLE_ONLY:
                continue
            ids = (0,) if name == "UniWAM" else (3, 7)
            cases = [[[] for _ in ids], [[action] for _ in range(len(ids) + 1)]]
            if len(ids) == 2:
                cases.extend([[[action]], [[action, action], [action]], [[action], [action, action]]])
            if name != "EgoVLA":  # An empty outer batch ends its episode.
                cases.append([])
            for chunks in cases:
                with self.subTest(policy=name, lengths=list(map(len, chunks))):
                    env = Environment(ids)
                    with self.assertRaises((ValueError, RuntimeError)):
                        module["eval_one_episode_batch"](env, Client(chunks))
                    self.assertEqual(env.steps, [])

    def test_no_active_envs_at_entry_or_after_first_step(self):
        for name, module in self.loops.items():
            # Simate_beta and WorldLA have their own unchanged batch protocols.
            if name in SINGLE_ONLY or name in {"Simate_beta", "WorldLA"}:
                continue
            for initial_ids in ([], [3]):
                with self.subTest(policy=name, initial_ids=initial_ids):
                    env = Environment()
                    env.get_running_env_idx_list = lambda: [] if env.steps else initial_ids
                    env.get_obs_batch = Mock(wraps=env.get_obs_batch)
                    horizon = 1 if name == "HWM_CogWAM" else 2
                    client = Client([[{"step": step} for step in range(horizon)]])
                    module["eval_one_episode_batch"](env, client)
                    self.assertEqual(len(env.steps), len(initial_ids))
                    self.assertEqual(client.calls.count("get_action_batch"), len(initial_ids))
                    for args, _ in env.get_obs_batch.call_args_list:
                        self.assertEqual(args[0], [3])

    def test_invalid_late_action_never_advances_the_environment(self):
        good = {"arm_joint_state": [0., 1.], "ee_joint_state": [0.]}
        for bad in (dict(good, arm_joint_state=[0., np.nan]),
                    dict(good, ee_joint_state=[np.inf]), dict(good, arm_joint_state=[1.]),
                    dict(good, unexpected=[0]), {"arm_joint_state": [0., 1.]}):
            for batch in (False, True):
                with self.subTest(bad=bad, batch=batch):
                    env = Environment()
                    env.robot_action_dim_info = {"arm_dim": [2], "ee_dim": [1]}
                    env.action_type = "joint"
                    chunk = [good, bad]
                    if batch:
                        loop = self.loops["demo_policy"]["eval_one_episode_batch"]
                        client = Client([[good, good], chunk])
                    else:
                        loop = self.loops["demo_policy"]["eval_one_episode"]
                        client = Client(chunk)
                    with self.assertRaises((KeyError, ValueError)):
                        loop(env, client)
                    self.assertEqual(env.steps, [])

        invalid_dict = [{"arm_joint_state": [0.]}, {"arm_joint_state": ["bad"]}]
        packed = np.array([[0., 0.], [0., np.inf]])
        for chunks in ([invalid_dict] * 2, [packed.tolist()] * 2,
                       [packed] * 2, np.stack([packed, packed])):
            env = Environment()
            with self.assertRaises(ValueError):
                self.loops["demo_policy"]["eval_one_episode_batch"](env, Client(chunks))
            self.assertEqual(env.steps, [])

    def test_reference_loop_preserves_packed_numpy_batches(self):
        env = Environment(ids=(7, 3))
        chunks = np.arange(12).reshape(2, 3, 2)
        self.loops["demo_policy"]["eval_one_episode_batch"](env, Client(chunks))
        self.assertEqual([ids for ids, _ in env.steps], [[7, 3], [3]])
        np.testing.assert_array_equal(env.steps[0][1], chunks[:, 0])
        np.testing.assert_array_equal(env.steps[1][1], chunks[1:, 1])
        for array in (chunks[:1], chunks[:, :0], chunks[:, :, :0], chunks[0],
                      chunks.astype(str)):
            with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaises(ValueError):
                validate_action_batch(array, 2)

    def test_mobile_chunk_uses_complete_native_fields(self):
        env = Environment()
        env.robot_action_dim_info = {"arm_dim": [7, 7], "ee_dim": [12, 12]}
        env.deploy_cfg = {"env_cfg_type": "m92uw", "action_type": "joint"}
        good = {"root_linear_velocity": [0., 0., 0.]}
        bad = {"root_linear_velocity": [0.]}

        def validate_dict(step):
            if len(step["root_linear_velocity"]) != 3:
                raise ValueError("invalid base velocity")
        env.validate_action_dict = validate_dict
        with self.assertRaisesRegex(ValueError, "base velocity"):
            self.loops["MoPA"]["eval_one_episode"](env, Client([good, bad]))
        self.assertEqual(env.steps, [])

    def test_aha_debug_resets_the_model_once_per_episode(self):
        env = AhaDebugEnv.__new__(AhaDebugEnv)
        env.episode_step_limit = 1
        env.robot_action_dim_info = {"arm_dim": [1], "ee_dim": [1]}
        action = {"arm_joint_state": np.zeros(1), "ee_joint_state": np.zeros(1)}
        env.model_client = Client([action], [action])
        for _ in range(2):
            env.reset()
            env.eval_one_episode()
        self.assertEqual(env.model_client.calls, ["reset", "update_obs", "get_action"] * 2)

    def test_chunk_boundary_checks_episode_end_only_once(self):
        for name in ("A1", "FastWAM", "OpenWAM", "X_WAM"):
            for chunk_length in (1, 3):
                with self.subTest(policy=name, chunk_length=chunk_length):
                    env = Environment()
                    env.is_episode_end = Mock(wraps=env.is_episode_end)
                    chunk = [{"step": step} for step in range(chunk_length)]
                    self.loops[name]["eval_one_episode"](env, Client(chunk, chunk))
                    self.assertEqual(len(env.steps), 2)
                    self.assertEqual(env.is_episode_end.call_count, 3)

    def test_single_stops_mid_chunk_and_batch_keeps_active_indices_aligned(self):
        for name, module in self.loops.items():
            with self.subTest(policy=name):
                env = Environment()
                actions = [{"step": i} for i in range(3)]
                replies = [[a] for a in actions] if name == "HWM_CogWAM" else [actions]
                module["eval_one_episode"](env, Client(*replies))
                self.assertEqual(env.steps, actions[:2])
                if name in SINGLE_ONLY:
                    continue
                ids = (0,) if name == "UniWAM" else (3, 7)
                env = Environment(ids)
                horizon = 10 if name == "WorldLA" else 3
                chunks = [[{"env": i, "step": s} for s in range(horizon)] for i in ids]
                active = ids[-1]
                replies = [chunks]
                if name == "HWM_CogWAM":
                    replies = [[c[:1] for c in chunks], [chunks[-1][1:2]]]
                module["eval_one_episode_batch"](env, Client(*replies))
                self.assertEqual(env.steps, [
                    (list(ids), [c[0] for c in chunks]),
                    ([active], [chunks[-1][1]]),
                ])

    def test_wait_and_end_messages_keep_their_protocol_meaning(self):
        env = Environment()
        joy = runpy.run_path(str(POLICIES / "JoyAI_VL/deploy.py"))
        client = Client([{"hexashell": "wait"}], [])
        joy["eval_one_episode"](env, client)
        self.assertEqual(env.steps, [])
        self.assertEqual(client.calls, ["reset", "update_obs", "get_action", "get_action", "trial_end"])

        mem = runpy.run_path(str(POLICIES / "Mem_0/deploy.py"))
        action = {"step": 1}
        mem["eval_one_episode"](env, Client(None, action, action))
        self.assertEqual(env.steps, [action, action])

        env = Environment()
        self.loops["EgoVLA"]["eval_one_episode_batch"](env, Client([]))
        self.assertEqual(env.steps, [])

    def test_camera_wait_and_failure_cleanup(self):
        env = Environment()
        ready = env.get_obs()
        env.get_obs = Mock(side_effect=[{}, ready, ready])
        client = Client([{"step": 0}, {"step": 1}])
        self.loops["Awomo05"]["eval_one_episode"](env, client)
        self.assertEqual(env.get_obs.call_count, 3)
        self.assertEqual(client.calls, ["reset", "update_obs", "get_action", "update_obs"])

        for policy, cleanup in (("OpenDM", "reset_evaluation"), ("Q0_Preview", "trial_end")):
            with self.subTest(policy=policy):
                client = Client([])
                with self.assertRaises(ValueError):
                    self.loops[policy]["eval_one_episode"](Environment(), client)
                self.assertEqual(client.calls[-1], cleanup)

    def test_camera_batch_waits_for_every_requested_observation(self):
        for name in ("Awomo05", "DreamZero"):
            with self.subTest(policy=name):
                env = Environment(ids=(7, 3))
                ready = env.get_obs_batch(env.ids)
                env.get_obs_batch = Mock(side_effect=[[], ready[:1], ready])
                result = self.loops[name]["_get_valid_obs_batch"](env, env.ids, interval=0)
                self.assertIs(result, ready)
                self.assertEqual(env.get_obs_batch.call_count, 3)

    def test_mobile_debug_clients_reject_ragged_replies(self):
        action = {"step": 0}
        ragged = [[action], [action, action]]
        with patch.object(sys, "path", [str(POLICIES / "MoPA"), *sys.path]):
            mopa = runpy.run_path(str(POLICIES / "MoPA/mobile_debug_client.py"))
        with self.assertRaisesRegex(ValueError, "same length"):
            mopa["_run_batch"](Client(ragged), Mock(batch_size=2), 1, "singular")

        argv = ["debug_mobile_client.py", "--port", "1", "--eval_batch", "true", "--batch_size", "2"]
        with patch.object(sys, "path", list(sys.path)), patch.object(sys, "argv", argv):
            # This test stops at the RPC boundary, before mobile action decoding.
            with patch.dict(sys.modules, {"mmabc.embodiments": Mock()}):
                mmabc = runpy.run_path(str(POLICIES / "MM_ABC/debug_mobile_client.py"))
            source = MagicMock()
            source.__len__.return_value = 3
            with patch.dict(mmabc["main"].__globals__, {
                "WsModelClient": Mock(return_value=Client(ragged)),
                "SyntheticSource": Mock(return_value=source),
            }), self.assertRaisesRegex(ValueError, "same length"):
                mmabc["main"]()

    def test_uniwam_robotwin_shim_rejects_empty_reply(self):
        shim_module = runpy.run_path(str(POLICIES / "UniWAM/robotwin_eval_client.py"))
        client = Client([])
        env = Mock()
        observation = {"joint_action": {"vector": np.zeros(14)}}
        for camera in ("head_camera", "left_camera", "right_camera"):
            observation[camera] = {"rgb": np.zeros((2, 2, 3))}
        with self.assertRaisesRegex(ValueError, "empty action chunk"):
            shim_module["build_policy_shim"](client).eval(env, client, observation)
        env.take_action.assert_not_called()
