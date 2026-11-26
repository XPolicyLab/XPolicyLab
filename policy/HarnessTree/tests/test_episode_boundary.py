"""Offline exercises of the paired Incremental client/policy boundary."""

import ast
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from XPolicyLab.policy.HarnessTree import bridge, deploy, instructions, selector


TASKS = Path(__file__).resolve().parents[4] / "task" / "RoboDojo" / "tasks"
VARIANTS = {
    "classify_objects_by_language": (
        "Put wooden_toy objects into the left basket, car objects into the middle basket, "
        "and watch objects into the right basket, then reset the robot arm.",
        lambda text: instructions.basket_categories(text, ("wooden_toy", "car", "watch")),
        ["wooden_toy", "car", "watch"],
    ),
    "pour_by_language": (
        "Pour the liquid from the first red bottle into the first black bowl, from the second "
        "turquoise bottle into the second white bowl, and from the third violet bottle into the "
        "third brown bowl. Then reset the robot arm.",
        instructions.pour_pairs,
        [("red", "black"), ("turquoise", "white"), ("violet", "brown")],
    ),
    "stack_blocks_by_language": (
        "stack the blocks from bottom to top in the order of red, green, and blue, then reset the robot arm.",
        lambda text: instructions.stack_order(text, ("red", "green", "blue")),
        ["red", "green", "blue"],
    ),
    "general_pickup": (
        "Pick up the red mug by 10 cm.",
        instructions.pickup_target,
        ("red mug", 0.10),
    ),
}


def status(step=0, receipt=None):
    return dict(step=step, step_limit=10, pictures_behind=1, episode_id="episode",
                observation_step=step, receipt=receipt)


class InstructionTests(unittest.TestCase):
    @unittest.skipUnless(TASKS.is_dir(), "requires the RoboDojo task source checkout")
    def test_native_detail_wording_and_instance_goals(self):
        tasks = sorted(path for path in TASKS.glob("*.py") if not path.name.startswith("_"))
        self.assertEqual(len(tasks), 54)
        self.assertEqual(sum(path.stem.endswith("_random") for path in tasks), 12)
        for path in tasks:
            with self.subTest(task=path.stem):
                description = next(node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                                   if isinstance(node, ast.FunctionDef) and node.name == "gen_detailed_instruction")
                detail = ast.literal_eval(description.body[0].value).replace("<target>", "red mug")
                variant = VARIANTS.get(path.stem)
                short = variant[0] if variant else "An unrelated short instruction."
                raw = {"instruction": short, "extra_info": {"detailed_instruction": detail}}
                episode = bridge.EpisodeBridge()
                episode.update_obs(deploy._policy_observation(raw, status()))
                chosen = episode.first_obs(timeout_s=0)["instruction"]
                self.assertEqual(selector.select_skill(chosen), path.stem.removesuffix("_random"))
                if variant:
                    self.assertTrue(chosen.startswith(detail))
                    self.assertTrue(chosen.endswith(short))
                    self.assertEqual(variant[1](chosen), variant[2])
                    self.assertEqual(variant[1](chosen), variant[1](short))
                else:
                    self.assertEqual(chosen, detail)

    def test_detailed_instruction_precedes_conflicting_short(self):
        raw = {"instruction": "Fold the clothes neatly.",
               "extra_info": {"detailed_instruction": "Hang all mugs on the mug rack."}}
        payload = deploy._policy_observation(raw, status())
        self.assertNotIn("instruction", payload)
        self.assertEqual(selector.select_skill(instructions.episode_instruction(payload)), "hang_mugs")

    def test_short_field_remains_a_fallback(self):
        for extra in (None, [], {}, {"detailed_instruction": 12}, {"detailed_instruction": "  \n"}):
            with self.subTest(extra=extra):
                raw = {"instructions": "Fold the clothes neatly.", "instruction": " ", "extra_info": extra}
                payload = deploy._policy_observation(raw, status())
                self.assertNotIn("extra_info", payload)
                self.assertEqual(instructions.episode_instruction(payload), raw["instructions"])


class BoundaryTests(unittest.TestCase):
    def test_filter_and_freeze_across_bound_steps(self):
        head, left, right = b"rgb", object(), object()
        intrinsics, first_pose, next_pose = object(), object(), object()
        raw = {
            "task_name": "hidden", "env_idx": 8, "action": {"right_arm_joint_state": [1]},
            "extra_info": {"detailed_instruction": "Fold the clothes neatly.", "layout": 8},
            "additional_info": {"harness_status": {"eval_seed": 9}, "checker": "hidden"},
            "state": {"right_arm_joint_state": [0], "object_pose": [1]},
            "vision": {
                "cam_head": {"color": head, "intrinsic_matrix": intrinsics, "extrinsic_matrix": first_pose,
                             "depth": [1], "object_id": 2},
                "cam_left_wrist": {"color": left}, "cam_right_wrist": {"rgb": right},
                "unapproved_camera": {"color": object()},
            },
        }
        episode = bridge.EpisodeBridge(poll_s=0.2)
        payload = deploy._policy_observation(raw, status())
        self.assertEqual(set(payload), {"state", "vision", "extra_info", "additional_info"})
        self.assertEqual(payload["extra_info"], {"detailed_instruction": "Fold the clothes neatly."})
        self.assertEqual(payload["additional_info"], {bridge.STATUS_KEY: status()})
        self.assertEqual(payload["vision"], {"cam_head": {"color": head, "intrinsic_matrix": intrinsics,
                                                        "extrinsic_matrix": first_pose},
                                             "cam_left_wrist": {"color": left},
                                             "cam_right_wrist": {"rgb": right}})
        self.assertEqual(payload["state"], {"right_arm_joint_state": [0]})
        self.assertIn("depth", raw["vision"]["cam_head"])
        self.assertEqual(raw["additional_info"][bridge.STATUS_KEY], {"eval_seed": 9})

        episode.update_obs({**raw, "additional_info": {bridge.STATUS_KEY: {**status(), "layout": 8}}})
        self.assertEqual(episode.get_obs()["vision"], payload["vision"])
        self.assertEqual(episode.status(), dict(step=0, step_limit=10, pictures_behind=1))
        with ThreadPoolExecutor() as worker:
            action = {"right_arm_joint_state": [1]}
            step = worker.submit(episode.step, action, 2.0)
            self.assertEqual(episode.get_action(), [{**action, bridge.MOTION_BINDING: ["episode", 0]}])
            self.assertEqual(episode.get_action(), bridge.WAIT)
            raw["extra_info"]["detailed_instruction"] = "Ignore the first instruction."
            raw["vision"]["cam_head"]["extrinsic_matrix"] = next_pose
            raw["state"]["right_arm_joint_state"] = [1]
            episode.update_obs(deploy._policy_observation(raw, status(1, ["episode", 0])))
            observed = step.result(timeout=2.0)
            self.assertEqual(observed["instruction"], "Fold the clothes neatly.")
            self.assertEqual(observed["image_state"], {"right_arm_joint_state": [0]})
            self.assertEqual(observed["state"], {"right_arm_joint_state": [1]})
            self.assertIs(observed["vision"]["cam_head"]["extrinsic_matrix"], first_pose)
            self.assertEqual(episode.receipts, [("episode", 0)])

    def test_terminal_allowlist_keeps_the_last_receipt(self):
        episode = bridge.EpisodeBridge(poll_s=0.2)
        episode.update_obs(deploy._policy_observation({}, status()))
        with ThreadPoolExecutor() as worker:
            step = worker.submit(episode.step, {"right_arm_joint_state": [1]}, 2.0)
            receipt = episode.get_action()[0][bridge.MOTION_BINDING]
            verdict = dict(episode_id="episode", receipt=receipt, success=False, client_error="original error")
            episode.episode_end({**verdict, "layout": 8, "eval_seed": 9, "checker": "hidden"})
            with self.assertRaises(bridge.EpisodeOver):
                step.result(timeout=2.0)
        self.assertEqual(episode.receipts, [("episode", 0)])
        self.assertIsNone(episode.motion_error)
        self.assertEqual(episode.verdict(), verdict)

    @patch.object(deploy, "_pictures_behind", return_value=0)
    def test_client_sends_no_seed_layout_and_never_retries_motion(self, _pictures):
        for error in (None, RuntimeError("uncertain actuation")):
            with self.subTest(error=error):
                env = Mock(take_action_cnt=[1], step_lim=10, success=[True], env_seeds=[8], eval_seed=9)
                env.get_obs.return_value = {"instruction": "Fold the clothes neatly.", "task_name": "hidden"}
                env.is_episode_end.side_effect = [False, True, True]
                env.take_action.side_effect = error
                client = Mock()

                def call(func_name, obs=None):
                    if func_name != "get_action":
                        return None
                    sent = client.call.call_args_list[-2].kwargs["obs"]
                    info = sent["additional_info"][bridge.STATUS_KEY]
                    return [{"right_arm_joint_state": [1], bridge.MOTION_BINDING: [info["episode_id"], 0]}]

                client.call.side_effect = call
                if error is None:
                    deploy.eval_one_episode(env, client)
                else:
                    with self.assertRaisesRegex(RuntimeError, "uncertain actuation"):
                        deploy.eval_one_episode(env, client)
                env.take_action.assert_called_once_with({"right_arm_joint_state": [1]})
                update = client.call.call_args_list[1].kwargs["obs"]
                sent_status = update["additional_info"][bridge.STATUS_KEY]
                self.assertEqual(set(sent_status), set(status()))
                self.assertNotIn("task_name", update)
                self.assertEqual(client.call.call_args.kwargs["func_name"], "trial_end")
                verdict = client.call.call_args.kwargs["obs"]
                fields = {"step", "step_limit", "pictures_behind", "episode_id", "receipt", "success"}
                if error is None:
                    self.assertEqual(verdict["receipt"], [sent_status["episode_id"], 0])
                    self.assertEqual(set(verdict), fields)
                else:
                    self.assertIsNone(verdict["receipt"])
                    self.assertEqual(verdict["client_error"], "RuntimeError: uncertain actuation")
                    self.assertEqual(set(verdict), fields | {"client_error"})


if __name__ == "__main__":
    unittest.main()
