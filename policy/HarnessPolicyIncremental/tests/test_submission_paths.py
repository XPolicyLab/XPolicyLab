"""Offline fixtures for native termination, conveyor tracking and SAM3 wire pixels."""

import ast
import base64
import io
import math
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from XPolicyLab.policy.HarnessPolicyIncremental import bridge, deploy, sam3_http_client

POLICY = Path(__file__).resolve().parents[1]
NATIVE = POLICY.parents[2] / "src" / "eval_client" / "eval_env.py"
CONVEYOR = POLICY / "programs" / "match_and_pick_from_conveyor.py"


def load_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise AssertionError(f"expected exactly one definition of each {names} in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)


@unittest.skipUnless(NATIVE.is_file(), "requires the pinned RoboDojo evaluator source")
class NativeTerminationTests(unittest.TestCase):
    def test_final_check_uses_evaluator_reward_and_preserves_receipt(self):
        scope = {"deepcopy": deepcopy}
        load_functions(NATIVE, {"is_episode_end", "get_running_env_idx_list"}, scope)

        class Evaluator:
            is_episode_end = scope["is_episode_end"]
            get_running_env_idx_list = scope["get_running_env_idx_list"]

            def __init__(self, budget, reward):
                self.num_envs = 1
                self.success = [True]
                self.end_flag = [False]
                self.take_action_cnt = [0]
                self.step_lim = budget
                self.final_checks = []
                self.reward_manager = Mock(get_reward=self.get_reward)
                self.get_obs_batch = Mock()
                self.get_obs = Mock(return_value={"instruction": "Fold the clothes."})
                self.take_action = Mock(side_effect=self.act)
                self.final_reward = reward

            def get_reward(self, final_check):
                if final_check and not self.end_flag[0]:
                    self.final_checks.append((self.take_action_cnt[0], self.success[0]))
                return [self.final_reward if final_check else 0.0]

            def act(self, action):
                self.take_action_cnt[0] += 1

        for budget, stop_after, reward in ((3, 1, 1.0), (3, 1, 0.0), (2, 2, 1.0), (2, 2, 0.0)):
            with self.subTest(budget=budget, stop_after=stop_after, reward=reward):
                env = Evaluator(budget, reward)
                sent = {}
                verdict = {}

                def call(func_name, obs=None):
                    if func_name == "update_obs":
                        sent.update(obs["additional_info"][bridge.STATUS_KEY])
                    elif func_name == "get_action":
                        if sent["observation_step"] == stop_after:
                            return []
                        return [{"joint": [1.0], bridge.MOTION_BINDING: [sent["episode_id"], sent["observation_step"]]}]
                    elif func_name == "trial_end":
                        self.assertEqual(env.end_flag, [True])
                        verdict.update(obs)

                with patch.object(deploy, "_pictures_behind", return_value=0):
                    deploy.eval_one_episode(env, Mock(call=call))

                self.assertEqual(env.take_action.call_count, stop_after)
                self.assertEqual(env.final_checks, [(stop_after, stop_after == budget)])
                env.get_obs_batch.assert_called_once_with(env_idx_list=[0], last_frame=True)
                self.assertEqual(verdict["success"], bool(reward))
                self.assertEqual(verdict["receipt"], [sent["episode_id"], stop_after - 1])


class ConveyorTrackingTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(CONVEYOR.read_text(encoding="utf-8"))
        fallback = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "VG_FALLBACK" for target in node.targets))
        self.prior = ast.literal_eval(fallback)
        self.scope = {"np": np, "math": math, "STEP": [100], "VG_FALLBACK": self.prior, "G_D": 0.10,
                      "TORSO_OFF": (-0.017, 0.020)}
        load_functions(CONVEYOR, {"g_live_speed", "g_measure", "garage_main", "g_approach_steps", "g_lead_x",
                                  "g_intercept_arm"}, self.scope)

    def test_accepts_consistent_track_and_rejects_ambiguous_fits(self):
        speed = 0.00408
        samples = np.array([(step, -0.4 + speed * (step - 100), 0.0, 20, 0.001) for step in (100, 102, 104)])
        self.assertAlmostEqual(self.scope["g_live_speed"](samples), speed)
        for row, column, value in ((1, 0, 100), (1, 1, -0.398), (2, 1, -0.4), (2, 1, -0.39),
                                   (1, 2, 0.02), (1, 2, np.nan), (0, 3, 5), (2, 4, 0.005)):
            with self.subTest(row=row, column=column, value=value):
                bad = samples.copy()
                bad[row, column] = value
                self.assertIsNone(self.scope["g_live_speed"](bad))

    def test_observes_middle_of_existing_holds_and_fails_open(self):
        speed = 0.00408
        step = self.scope["STEP"]

        def tick(arm):
            step[0] += 2

        def base_fit(mask, tag="", image=None, frame_step=None):
            frame = step[0] if image is None else int(image[0])
            if image is not None:
                self.assertEqual(frame, frame_step)
            return -0.4 + (frame - 100) * speed, 0.0, 20, 0.001

        self.scope.update(tick=Mock(side_effect=tick), base_fit=Mock(side_effect=base_fit),
                          g_seg_near=Mock(return_value=(object(), None)), robo=Mock())
        image = self.scope["robo"].image
        image.side_effect = lambda camera: np.array([step[0]])
        for usable in (True, False):
            with self.subTest(usable=usable):
                if not usable:
                    image.side_effect = RuntimeError("middle frame unavailable")
                step[0] = 100
                self.scope["tick"].reset_mock()
                fits, live = self.scope["g_measure"](None, None)
                self.assertEqual([fit[0] for fit in fits], [100, 104])
                self.assertEqual(self.scope["tick"].call_count, 4)
                self.assertEqual(step[0], 108)
                if usable:
                    self.assertAlmostEqual(live, speed)
                else:
                    self.assertIsNone(live)

    def test_intercept_uses_live_speed_or_exact_prior(self):
        for estimate in (None, 0.00408):
            with self.subTest(estimate=estimate):
                fits = [(100, -0.4, 0.0, 20, 0.001), (104, -0.384, 0.0, 20, 0.001)]
                attempt = Mock()
                self.scope.update(g_measure=Mock(return_value=(fits, estimate)), g_attempt=attempt,
                                  log=Mock(), g_seg_near=Mock(return_value=(None, None)))
                self.scope["garage_main"]({"P": (-0.4, 0.0), "k": 100})
                intercept, lateral, attempt_number, used_speed = attempt.call_args.args
                expected = self.prior if estimate is None else estimate
                self.assertEqual(used_speed, expected)
                self.assertAlmostEqual(intercept, np.median([fit[1] - fit[0] * expected for fit in fits]))
                self.assertEqual((lateral, attempt_number), (0.0, 0))

    def test_intercept_keeps_prior_across_arm_and_reach_boundaries(self):
        estimate = 0.00408
        self.scope["STEP"][0] = 108
        for boundary in (-0.08, 0.48):
            with self.subTest(boundary=boundary):
                lead_x = self.scope["g_lead_x"]
                arm = self.scope["g_intercept_arm"]
                prior_sb = boundary - 0.0005 - lead_x(0.0, self.prior)
                fits = [(step, prior_sb + step * self.prior, 0.0, 20, 0.001) for step in (100, 104)]
                live_sb = np.median([fit[1] - fit[0] * estimate for fit in fits])
                self.assertNotEqual(arm(lead_x(prior_sb, self.prior)), arm(lead_x(live_sb, estimate)))
                attempt = Mock()
                self.scope.update(g_measure=Mock(return_value=(fits, estimate)), g_attempt=attempt,
                                  log=Mock(), g_seg_near=Mock(return_value=(None, None)))
                self.scope["garage_main"]({"P": (fits[0][1], 0.0), "k": 100})
                self.assertAlmostEqual(attempt.call_args.args[0], prior_sb)
                self.assertEqual(attempt.call_args.args[3], self.prior)


class Sam3WireTests(unittest.TestCase):
    def test_rgb_and_binary_mask_round_trip(self):
        pixels = np.array([[[253, 2, 19], [7, 37, 244]], [[211, 111, 55], [37, 186, 94]]], dtype=np.uint8)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        response = {"instances": [{"box": [0, 0, 1, 1], "score": 0.9,
                                  "mask": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}]}
        client = sam3_http_client.HttpSam3()

        def post(route, payload):
            if route == "/segment_point":
                self.assertEqual(payload["points"], [[1, 0]])
            else:
                self.assertEqual(payload["prompt"], "tile")
            raw = base64.b64decode(payload["image"].split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as decoded:
                self.assertEqual(decoded.mode, "RGB")
                np.testing.assert_array_equal(np.asarray(decoded), pixels)
            return response

        with patch.object(client, "_post", side_effect=post) as request:
            result = client.segment(pixels, text_prompt="tile")
            point_result = client.segment(pixels, point=[0, 1])
        self.assertEqual([call.args[0] for call in request.call_args_list], ["/segment", "/segment_point"])
        for item in (result, point_result):
            self.assertTrue(item.found)
            self.assertEqual(item.score, 0.9)
            np.testing.assert_array_equal(item.mask, mask > 0)
