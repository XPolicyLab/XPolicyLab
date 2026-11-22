"""CPU routing contracts for the adapters changed in the batch-selection migration.

Run each adapter's update/get batch method bodies and its deployment loop. The
inference double echoes a value encoded in the observation; it never gets the
expected environment ID from the assertion. Isolate the methods from GPU-only
module imports and checkpoint constructors, as in test_action_contracts.py.
"""
from __future__ import annotations

import ast
import asyncio
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.utils.batch_indices import observation_env_ids, observation_rows, selected_env_ids, validate_observed_ids
from XPolicyLab.utils.process_data import unpack_robot_state


POLICIES = Path(__file__).resolve().parents[1] / "policy"
NAMES = """
Being_H05 Dexora_1B G05 GO_3_Nano_preview GR00T_N17 GalaxeaVLA GigaWorldPolicy
H_RDT InternVLA_A1 InternVLA_A1_5 LDA_1B LingBot_VLA ME_Brain_1 Meituan_Robotics_0
OpenVLA_OFT Pi_0 Pi_05 Pi_0_Fast RDT_1B Rex_M1_preview Spatial_Forcing Spirit_v15
StarVLA X_VLA Xiaomi_Robotics_0 Xiaomi_Robotics_1
""".split()


def observation(env_id, step):
    tag = 100 * env_id + step
    image = np.zeros((2, 2, 3), np.uint8)
    image[:] = [env_id, step, 0]
    return dict(env_idx=env_id, state=np.array([tag]), prompt=str(tag),
                images=dict.fromkeys(("cam_high", "cam_left_wrist", "cam_right_wrist"), image))


def predict(obs):
    value = obs["state"][0]
    if "images" in obs:
        if str(value) != obs["prompt"]:
            raise AssertionError("Instruction and state came from different observations")
        for image in obs["images"].values():
            env_id, step = map(int, image[0, 0, :2])
            if value != 100 * env_id + step:
                raise AssertionError("Camera and state came from different observations")
    return np.array([[value, value], [value + 1, value + 1]], dtype=np.float32)


def markers(chunks):
    return [[int(next(iter(step.values()))[0] if isinstance(step, dict) else step[0])
             for step in chunk] for chunk in chunks]


def identity(value, *args, **kwargs):
    return value


def make_model(name):
    path = POLICIES / name / "model.py"
    tree = ast.parse(path.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    methods = {"update_obs_batch", "get_action_batch", "_env_key_from_obs", "_current_action_vector"}
    if name in ("LDA_1B", "Rex_M1_preview"):
        methods.add("reset")
    if name == "StarVLA":
        methods.add("_infer_chunk")
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    helpers = {"stack_obs", "slice_stacked_obs", "stack_observations", "slice_stacked_observation"}
    standalone = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    torch = Mock(no_grad=nullcontext, autocast=lambda **kw: nullcontext(), Tensor=type(None))
    ns = dict(np=np, deque=deque, torch=torch, observation_env_ids=observation_env_ids,
              observation_rows=observation_rows, selected_env_ids=selected_env_ids,
              validate_observed_ids=validate_observed_ids,
              unpack_robot_state=unpack_robot_state, encode_obs=identity, _encode_observation=identity,
              encode_observation=identity, hist_from_obs=lambda obs: dict(hist_images=obs["state"], hist_mask=[1]),
              _process_data_helpers=lambda: (None, None, unpack_robot_state),
              decode_action_chunk=identity, _gr00t_action_to_env=identity,
              action_chunk_to_ee_dict_list=identity, slice_action_chunk=identity,
              resolve_prompt=lambda obs, default: obs["prompt"], _unnormalize_q99=identity)
    exec(compile(ast.Module(standalone + functions, type_ignores=[]), str(path), "exec"), ns)
    model = type("RoutingModel", (), {f.name: ns[f.name] for f in functions})()
    model.action_type = model.ACTION_TYPE = "joint"
    model.robot_action_dim_info = {"arm_dim": [1], "ee_dim": [1]}
    model.default_prompt = model.task_name = "test"
    model._prepare = model._encode_obs = model._convert_obs = model._encode_observation = identity
    model.encode_obs = identity
    model._predict_action_sequence = model._predict_action_chunk = predict
    model._predict_action_chunks = model._predict = lambda observations: [predict(o) for o in observations]
    model._infer = model.infer = predict
    model.policy = SimpleNamespace(infer=lambda obs: {"actions": predict(obs)}, get_action=predict)
    if name == "G05":
        def predict_with_history(observations, env_keys):
            if [o["env_idx"] for o in observations] != list(env_keys):
                raise AssertionError("History keys do not match selected observations")
            return [predict(o) for o in observations]
        model._predict_chunks = predict_with_history

    if name == "Dexora_1B":
        model.img_history_size, model._obs_windows = 2, {}
        model._infer_chunk = lambda env_id: predict(model._obs_windows[env_id][-1])
    elif name == "H_RDT":
        model.max_img_cache_size, model.obs_cache_by_env = 2, {}
        model._infer = lambda env_id: predict(model.obs_cache_by_env[env_id][-1])
    elif name.startswith("InternVLA_A1"):
        model._latest_obs_by_env, model._histories_by_env = {}, {}
        model._append_history = Mock()
        model.infer = lambda env_id: predict(model._latest_obs_by_env[env_id])
    elif name == "GR00T_N17":
        model.policy.get_action = lambda obs: (predict(obs), None)
    elif name == "LingBot_VLA":
        model.infer = lambda obs: {"action": predict(obs)}
    elif name == "Spirit_v15":
        model.infer = lambda observation, **kw: dict(actions=predict(observation), task_name="", action_type="joint")
        model._decode_action_chunk = identity
    elif name == "ME_Brain_1":
        model.replan_steps = 2
    elif name == "Meituan_Robotics_0":
        model.obs_by_env, model.action_stats, model.execute_horizon = {}, {}, 2
        model.model = SimpleNamespace(predict_action=lambda examples: {
            "normalized_actions": [predict(o) for o in examples]})
    elif name == "StarVLA":
        model.obs_by_env, model.step_by_env, model.action_chunks_by_env = {}, {}, {}
        model.action_dim, model.execute_horizon = 2, 2
        model.use_ddim, model.num_ddim_steps, model.unnorm_key = False, 1, None
        model.client = SimpleNamespace(predict_action=lambda request: {
            "ok": True, "data": {"actions": [predict(o) for o in request["examples"]]},
        })
    elif name == "RDT_1B":
        model._observation_windows, model._embeddings_by_prompt = {}, {}
        model.device = "cpu"
        model.config = {"camera_names": ["head", "left", "right"]}
        model._encode_instruction = model._jpeg_mapping = model._resize_img = identity
        model.infer = lambda window: predict({"state": [int(window[-1]["text_embeds"])]})
    elif name == "GalaxeaVLA":
        model._collate = Mock(side_effect=lambda samples: {"samples": samples})
        model.action_shape_meta = [{"key": "joint"}]
        model.replan_steps = model.action_horizon = 2
        model.policy.parameters = lambda: iter([SimpleNamespace(dtype="float32")])
        model.policy.predict_action = Mock(side_effect=lambda batch: {"action": {"joint": [predict(o) for o in batch["samples"]]}})
        model.processor = SimpleNamespace(postprocess=identity)
    return model


class ShrinkingBatch:
    """Env 7 ends at step 1; envs 2 and 9 change order and finish at steps 4 and 5."""
    limits = {7: 1, 2: 4, 9: 5}

    def __init__(self):
        self.steps = dict.fromkeys(self.limits, 0)
        self.applied = []

    def get_running_env_idx_list(self):
        order = (7, 2, 9) if self.steps[2] < 2 else (9, 2, 7)
        return [i for i in order if self.steps[i] < self.limits[i]]

    def is_episode_end(self):
        return not self.get_running_env_idx_list()

    def get_obs_batch(self, ids):
        return [observation(i, self.steps[i]) for i in ids]

    def take_action_batch(self, actions, ids):
        if len(actions) != len(ids):
            raise AssertionError("Action count differs from the active environment count")
        for i, action in zip(ids, actions):
            actual = markers([[action]])[0][0]
            expected = 100 * i + self.steps[i]
            if self.steps[i] >= self.limits[i] or actual != expected:
                raise AssertionError(f"env={i}, step={self.steps[i]}, expected {expected}, got {actual}")
            self.applied.append((i, self.steps[i]))
            self.steps[i] += 1


async def rollout(name, model, deploy, env):
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
    await server.start()

    def drive():
        client = WsModelClient(url=server.url, evaluation_id=name, trial_id="routing",
                               max_connect_attempts=1, request_timeout_s=3)
        try:
            deploy["eval_one_episode_batch"](env, client)
        finally:
            client.close()

    try:
        await asyncio.wait_for(asyncio.to_thread(drive), timeout=20)
    finally:
        await server.stop()


def exercise(test, name):
    # Galaxea's postprocessor normally walks GPU tensors. Here all data are already NumPy.
    module = SimpleNamespace(dict_apply=lambda batch, fn: batch)
    with patch.dict(sys.modules, {"galaxea_fm.utils.pytorch_utils": module}):
        model = make_model(name)
        model.update_obs_batch([observation(i, 0) for i in (7, 2, 9)])
        for ids in ([9, 7, 2], [2], []):
            # StarVLA consumes one cached control step on every request.
            expected = [[100 * i + int(ids == [2])] if name == "StarVLA"
                        else [100 * i, 100 * i + 1] for i in ids]
            test.assertEqual(markers(model.get_action_batch(ids)), expected)
            if name == "GalaxeaVLA" and ids:
                inputs = model.policy.predict_action.call_args.args[0]
                test.assertEqual([o["env_idx"] for o in inputs["samples"]], ids)

        if name == "GalaxeaVLA":
            model.get_action_batch()
            model._collate.assert_called_once()
            test.assertIs(model.policy.predict_action.call_args.args[0], model._sample_batch)

        model = make_model(name)
        env = ShrinkingBatch()

        model.reset = Mock(return_value=None)  # Fresh routing fixture per episode.
        if name == "ME_Brain_1":
            hist_module = SimpleNamespace(LiveHeadHistoryTracker=Mock)
            with patch.dict(sys.modules, {"routing_fixture.hist_live": hist_module}):
                deploy = runpy.run_path(str(POLICIES / name / "deploy.py"), run_name="routing_fixture.deploy")
        else:
            deploy = runpy.run_path(str(POLICIES / name / "deploy.py"))
        asyncio.run(rollout(name, model, deploy, env))
        model.reset.assert_called_once_with()
        test.assertEqual(env.steps, env.limits)
        test.assertCountEqual(env.applied, [(i, s) for i, limit in env.limits.items() for s in range(limit)])


class PolicyBatchRoutingTests(unittest.TestCase):
    def test_invalid_ids_do_not_consume_actions_and_finished_envs_cannot_be_requested(self):
        vendor = SimpleNamespace(dict_apply=lambda batch, fn: batch)
        with patch.dict(sys.modules, {"galaxea_fm.utils.pytorch_utils": vendor}):
            for name in NAMES:
                with self.subTest(policy=name):
                    model = make_model(name)
                    model.update_obs_batch([observation(i, 0) for i in (7, 2)])
                    with self.assertRaises(ValueError):
                        model.update_obs_batch([observation(2, 1), observation(2, 2)])
                    self.assertEqual(markers(model.get_action_batch([7]))[0][0], 700)
                    model.update_obs_batch([observation(2, 1)])
                    for request in ([2, 7], [2, 2]):
                        with self.assertRaises(ValueError):
                            model.get_action_batch(request)
                    self.assertEqual(markers(model.get_action_batch([2]))[0][0], 201)
                    model.update_obs_batch([])
                    self.assertEqual(model.get_action_batch(), [])
                    with self.assertRaises(ValueError):
                        model.get_action_batch([2])

    def test_encoder_failure_cannot_relabel_a_previous_observation(self):
        for name in ("Being_H05", "Pi_05", "GalaxeaVLA", "Dexora_1B", "InternVLA_A1", "RDT_1B"):
            with self.subTest(policy=name):
                model = make_model(name)
                model.update_obs_batch([observation(i, 0) for i in (7, 2)])
                def encode(obs, *args):
                    if obs["env_idx"] == 7:
                        raise ValueError("invalid second observation")
                    return obs

                vendor = SimpleNamespace(dict_apply=lambda batch, fn: batch)
                with patch.dict(sys.modules, {"galaxea_fm.utils.pytorch_utils": vendor}):
                    with patch.dict(model.update_obs_batch.__globals__, {"encode_obs": encode}), \
                            patch.object(model, "_convert_obs", encode), patch.object(model, "_encode_obs", encode):
                        with self.assertRaisesRegex(ValueError, "invalid second"):
                            model.update_obs_batch([observation(i, 1) for i in (2, 7)])
                    self.assertEqual([steps[0] for steps in markers(model.get_action_batch())], [700, 200])

    def test_g05_wrapped_batch_cannot_relabel_observations(self):
        with patch.dict(sys.modules, {"torch": Mock()}):
            module = runpy.run_path(str(POLICIES / "G05/model.py"))
        model = module["Model"].__new__(module["Model"])
        model._predict_chunks = lambda observations, env_keys: [predict(obs) for obs in observations]
        batch = [observation(i, 0) for i in (7, 2)]
        model.update_obs(batch[0])
        self.assertEqual(markers([model.get_action()]), [[700, 701]])
        envelope = {module["BATCH_OBSERVATIONS_KEY"]: batch, module["BATCH_ENV_INDICES_KEY"]: [2, 7]}
        with self.assertRaises(ValueError):
            model.update_obs(envelope)
        self.assertEqual(markers([model.get_action()]), [[700, 701]])
        model.update_obs(dict(envelope, **{module["BATCH_ENV_INDICES_KEY"]: [7, 2]}))
        self.assertEqual(markers(model.get_action()), [[700, 701], [200, 201]])
        model.update_obs_batch([])
        self.assertEqual(model.get_action(), [])

    def test_g05_failed_inference_does_not_advance_temporal_inputs_on_retry(self):
        class Tensor(int):
            def clone(self):
                return self

            def float(self):
                return self

        with patch.dict(sys.modules, {"torch": SimpleNamespace(from_numpy=lambda value: Tensor(value.item()))}):
            module = runpy.run_path(str(POLICIES / "G05/model.py"))
        model = module["Model"].__new__(module["Model"])
        model._history_buffers = {}
        model.processor = SimpleNamespace(num_obs_steps=2, shape_meta={
            "images": [{"key": "head"}], "state": [{"key": "arm"}],
        })
        model._build_obs_accepts_buffers = True
        model._to_g05_raw_obs = lambda obs: {"images": {"head": obs["step"]}, "state": {"arm": obs["step"]}}
        model._build_obs_dict = lambda obs, processor, frame_buffers, state_buffers: list(frame_buffers["head"])
        model._format_action_chunk = identity
        model.inferencer = SimpleNamespace(infer=Mock(side_effect=lambda inputs: inputs))
        model.update_obs_batch([dict(env_idx=7, step=1)])
        self.assertEqual(model.get_action_batch(), [[1, 1]])
        model.update_obs_batch([dict(env_idx=7, step=2)])
        model.inferencer.infer.side_effect = lambda inputs: []
        with self.assertRaisesRegex(ValueError, "number of action chunks"):
            model.get_action_batch()
        model.inferencer.infer.side_effect = iter
        self.assertEqual(model.get_action_batch(), [[1, 2]])

    def test_failed_stateful_encoder_discards_partial_history(self):
        for name, encoder in (("LDA_1B", "encode_obs"), ("Rex_M1_preview", "_encode_observation")):
            with self.subTest(policy=name):
                model = make_model(name)
                model._image_history, model._state_history, model._hist_by_env = {}, {}, {}
                model.update_obs_batch([observation(7, 0)])
                history = model._image_history if name == "LDA_1B" else model._hist_by_env

                def encode(obs):
                    history[obs["env_idx"]] = obs["state"]
                    if obs["env_idx"] == 2:
                        raise ValueError("incomplete camera data")
                    return obs

                with patch.object(model, encoder, encode), patch("builtins.print"):
                    with self.assertRaisesRegex(ValueError, "camera data"):
                        model.update_obs_batch([observation(i, 1) for i in (7, 2)])
                self.assertFalse(model._image_history or model._state_history or model._hist_by_env)
                with self.assertRaises((AssertionError, RuntimeError)):
                    model.get_action_batch([7])
                model.update_obs_batch([observation(7, 2)])
                self.assertEqual(markers(model.get_action_batch([7]))[0][0], 702)

    def test_legacy_batch_without_ids_keeps_separate_observations(self):
        for name in ("H_RDT", "StarVLA", "Meituan_Robotics_0"):
            with self.subTest(policy=name):
                model = make_model(name)
                observations = [observation(i, 0) for i in (7, 2)]
                for obs in observations:
                    del obs["env_idx"]
                model.update_obs_batch(observations)
                result = markers(model.get_action_batch([1, 0]))
                self.assertEqual([chunk[0] for chunk in result], [200, 700])

    def test_history_encoders_receive_the_same_ids_as_batch_selection(self):
        for name, encoder in (("LDA_1B", "encode_obs"), ("Rex_M1_preview", "_encode_observation")):
            with self.subTest(policy=name):
                model = make_model(name)
                encode = Mock(side_effect=identity)
                setattr(model, encoder, encode)
                observations = [observation(i, 0) for i in (7, 2)]
                for obs in observations:
                    del obs["env_idx"]
                observations[0]["env_id"] = 7
                model.update_obs_batch(observations)
                self.assertEqual([call.args[0]["env_idx"] for call in encode.call_args_list], [7, 1])
                self.assertEqual(markers(model.get_action_batch([1, 7])), [[200, 201], [700, 701]])
                self.assertFalse(any("env_idx" in obs for obs in observations))

    def test_starvla_failed_request_does_not_consume_a_cached_step(self):
        model = make_model("StarVLA")
        model.update_obs_batch([observation(i, 0) for i in (7, 2)])
        for ids in ([7, 99], [2, 2]):
            with self.assertRaises((KeyError, ValueError, AssertionError)):
                model.get_action_batch(ids)
            self.assertEqual(model.step_by_env, {})
        self.assertEqual(markers(model.get_action_batch([2, 7])), [[200], [700]])


# Separate unittest IDs allow running any adapter in isolation and show per-policy results in CI.
for policy in NAMES:
    setattr(PolicyBatchRoutingTests, f"test_{policy}", lambda self, name=policy: exercise(self, name))
