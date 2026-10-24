"""run_policy drives a learned policy the way RoboDojo's own eval client does (G0.5 deploy.py)."""
import base64
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.runtime.reactive import ReactiveLayer  # noqa: E402
from harness_policy.services import policy_server as session_module  # noqa: E402
from harness_policy.geometry import DOWN  # noqa: E402
from harness_policy.runtime.session import session_class  # noqa: E402
from skills.vla import route_instruction  # noqa: E402

RoboDojoSession = session_class()

GROUPS = [{"name": "left_arm", "joints": [0, 1, 2, 3, 4, 5], "gripper": 6},
          {"name": "right_arm", "joints": [7, 8, 9, 10, 11, 12], "gripper": 13}]


def png_url():
    import cv2
    import numpy as np

    ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


class Worker:
    """Joints follow commands; the episode ends at step_limit."""

    def __init__(self, step_limit=40, instruction="Stack the bowls."):
        self.joints = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
        self.step, self.step_limit, self.observations = 0, step_limit, 0
        self.instruction = instruction

    def state(self):
        return {"step": self.step, "step_limit": self.step_limit, "ended": self.step >= self.step_limit,
                "joint_position": list(self.joints), "left_ee_pose": [0.0, 0.0, 1.0, *DOWN],
                "right_ee_pose": [0.0, 0.0, 1.0, *DOWN]}

    def call(self, name, **arguments):
        if name == "policy_obs":
            self.observations += 1
            return {"vision": {"cam_head": png_url()}, "state": {"left_arm_joint_state": self.joints[0:6]},
                    "instruction": self.instruction, "additional_info": {"frequency": 25},
                    "data_format_version": "v1.0", "robot_state": self.state()}
        if name == "tick":
            self.joints = list(arguments["command"])
            self.step += 1
            return {"executed": True, "shield": {"verdict": "pass", "reasons": [], "command": self.joints},
                    "state": self.state()}
        raise AssertionError(name)


class Policy:
    """An XPolicyLab client: chunks of `size` actions moving left joint 0 by 0.01 per step."""

    def __init__(self, size=16):
        self.size, self.calls, self.instructions = size, [], []

    def call(self, name, obs=None):
        self.calls.append(name)
        if name == "update_obs":
            self.instructions.append(obs["instruction"])
        if name == "get_action":
            return [{"left_arm_joint_state": [0.01 * (k + 1)] + [0.0] * 5, "left_ee_joint_state": [1.0],
                     "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]}
                    for k in range(self.size)]
        return None


class EndEffector(Policy):
    """An end-effector policy (action_type ee): link6 poses, the left arm 0.1 m further in x
    each step, the right arm at a pose with no IK solution."""

    def call(self, name, obs=None):
        self.calls.append(name)
        if name == "get_action":
            return [{"left_ee_pose": [0.1 * (k + 1), 0.0, 1.0, *DOWN], "left_ee_joint_state": [1.0],
                     "right_ee_pose": [0.6, 0.0, 1.0, *DOWN], "right_ee_joint_state": [1.0]}
                    for k in range(self.size)]
        return None


def session(worker, policy):
    s = object.__new__(RoboDojoSession)
    s.env, s.episode_id, s.hz, s.default_max_seconds = worker, "e", 25.0, 30.0
    s.reactive = ReactiveLayer(0.005)
    s.groups = {group["name"]: group for group in GROUPS}
    s.handlers, s._policy_clients = {}, {"g05": policy}
    s.policies = {"g05": {"url": "ws://127.0.0.1:19999"}}
    s._gripper_commands, s._attachments, s._events, s._live = {}, [], [], None
    s.tip_offset, s.table_z = 0.150, 0.765
    s.state = worker.state()
    return s


class RunPolicyTests(unittest.TestCase):
    def setUp(self):
        self.locks = tempfile.TemporaryDirectory()
        self.directory = mock.patch.object(session_module, "POLICY_LOCK_DIR", self.locks.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.locks.cleanup()

    def test_whole_chunks_and_an_observation_every_tick(self):
        worker, policy = Worker(), Policy(size=16)
        s = session(worker, policy)
        s.run_policy("g05", max_seconds=32 / 25.0)
        self.assertEqual(policy.calls[0], "reset")
        # 32 ticks: one update_obs each, get_action at ticks 1 and 17 only.
        self.assertEqual(policy.calls.count("update_obs"), 32)
        self.assertEqual(policy.calls.count("get_action"), 2)
        self.assertEqual(worker.observations, 32)
        self.assertEqual(set(policy.instructions), {"Stack the bowls."})  # the task's own instruction

    def test_state_kept_across_calls_with_the_same_instruction(self):
        worker, policy = Worker(), Policy()
        s = session(worker, policy)
        s.run_policy("g05", max_seconds=0.2)
        s.run_policy("g05", max_seconds=0.2)
        self.assertEqual(policy.calls.count("reset"), 1)
        s.run_policy("g05", "Pick up the bowl.", max_seconds=0.2)
        self.assertEqual(policy.calls.count("reset"), 2)
        self.assertEqual(policy.instructions[-1], "Pick up the bowl.")
        s.run_policy("g05", "Pick up the bowl.", max_seconds=0.2, reset=True)
        self.assertEqual(policy.calls.count("reset"), 3)

    def test_runs_to_the_end_of_the_episode_by_default(self):
        from harness_policy.runtime.events import EpisodeEnded

        worker, policy = Worker(step_limit=40), Policy()
        s = session(worker, policy)
        try:
            s.run_policy("g05")
        except EpisodeEnded:
            pass
        self.assertEqual(worker.step, 40)

    def test_a_fresh_run_starts_only_from_the_start_pose(self):
        from harness_policy.runtime.events import Escalation

        worker, policy = Worker(step_limit=400), Policy()
        s = session(worker, policy)
        s._start_joints = list(worker.joints)
        worker.joints[1] = 0.3  # an arm moved before the policy was handed the task
        s.state = worker.state()
        with self.assertRaises(Escalation) as raised:
            s.run_policy("g05", max_seconds=0.2)
        event = raised.exception.trace.event
        self.assertEqual((event.kind, event.detail["reason"]), ("SKILL_ERROR", "not at the start pose"))
        self.assertNotIn("reset", policy.calls)  # nothing ran
        worker.joints[1] = 0.0
        s.state = worker.state()
        s.run_policy("g05", max_seconds=0.2)  # from the start pose
        self.assertNotEqual(worker.joints[0], 0.0)
        trace = s.run_policy("g05", max_seconds=0.2)  # a continuation may start anywhere
        self.assertTrue(trace.ok)
        with self.assertRaises(Escalation):
            s.run_policy("g05", max_seconds=0.2, reset=True)  # a fresh one may not

    def test_stops_when_the_arms_stop_moving(self):
        from harness_policy.runtime.events import EpisodeEnded

        worker, policy = Worker(step_limit=400), Settling()
        s = session(worker, policy)
        trace = s.run_policy("g05")
        self.assertEqual(trace.event.kind, "COMPLETED")
        self.assertEqual(trace.event.detail, {"stopped": "stalled", "seconds": 5.0})
        self.assertEqual(trace.ticks, 126)  # 125 ticks after its one move
        worker, policy = Worker(step_limit=200), Settling()
        s = session(worker, policy)
        with self.assertRaises(EpisodeEnded):
            s.run_policy("g05", stall_seconds=None)  # runs on to the end of the episode
        worker, policy = Worker(step_limit=200), Still()
        s = session(worker, policy)
        with self.assertRaises(EpisodeEnded):  # arms still from the start are waiting, not stalled
            s.run_policy("g05")

    def test_end_effector_actions_are_solved_by_ik(self):
        worker, policy = Worker(), EndEffector()
        s = session(worker, policy)
        poses = []

        def ik(arm, pose):
            poses.append((arm, pose))
            return None if pose[0] > 0.5 else [pose[0]] + [0.0] * 5
        s.ik = ik
        s.run_policy("g05", max_seconds=4 / 25.0)
        self.assertEqual(poses[0], ("left_arm", [0.1, 0.0, 1.0, *DOWN]))
        self.assertAlmostEqual(worker.joints[0], 0.4)  # the last solved pose
        self.assertEqual(worker.joints[7], 0.0)  # no IK solution: the right arm holds

    def test_drops_are_reported_with_a_measured_gripper(self):
        worker, policy = Gripping(step_limit=400), Still(gripper=0.0)
        s = session(worker, policy)
        trace = s.run_policy("g05", max_seconds=2.0)
        self.assertEqual(trace.event.detail["drops"], [{"arm": "left_arm", "step": 21}])
        s.profile = {"gripper_feedback": "commanded"}
        worker.closed_at = None
        s._policy_runs.clear()
        trace = s.run_policy("g05", max_seconds=2.0)
        self.assertNotIn("drops", trace.event.detail)  # only the last command: nothing to measure


class StartTests(unittest.TestCase):
    """start(strategy, kind): Z2's first call, before which nothing moves."""

    def setUp(self):
        self.locks = tempfile.TemporaryDirectory()
        self.directory = mock.patch.object(session_module, "POLICY_LOCK_DIR", self.locks.name)
        self.directory.start()

    def tearDown(self):
        self.directory.stop()
        self.locks.cleanup()

    def strict(self, worker, policy):
        s = session(worker, policy)
        s._strategy_first, s.profile = True, {"return_to_start": True}
        s.vision = s.pointer = s.verifier = None
        return s

    def test_nothing_moves_before_start(self):
        from harness_policy.runtime.events import Escalation

        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        with self.assertRaises(Escalation) as raised:
            s.run_policy("g05", max_seconds=0.2)
        self.assertEqual(raised.exception.trace.event.detail["reason"], "call start(strategy, kind) first")
        self.assertEqual(worker.step, 0)
        s.hold(0.2)  # holding still is looking, not moving
        self.assertEqual(worker.step, 5)

    def test_a_and_c_hand_the_task_over_at_once_b_only_records(self):
        worker, policy = Worker(
            step_limit=400,
            instruction="Stack the three bowls together.",
        ), Settling()
        s = self.strict(worker, policy)
        trace = s.start("a", "putting objects into a bin")
        self.assertEqual(trace.skill, "run_policy")
        self.assertEqual(trace.event.detail.get("stopped"), "stalled")
        self.assertEqual(s._strategy["strategy"], "A")
        self.assertEqual(s.drain()["strategy"]["kind"], "putting objects into a bin")
        calls = len(policy.calls)
        self.assertIsNone(s.start("A", "a program run again from its top"))  # the choice stands
        self.assertEqual(len(policy.calls), calls)  # and the policy does not run again
        with self.assertRaisesRegex(ValueError, "already called"):
            s.start("B", "another strategy")
        other = Policy()
        reading = Worker(step_limit=400, instruction="Complete the equation by selecting the correct missing number or operator and placing it on the pad.")
        s = self.strict(reading, other)
        self.assertIsNone(s.start("B", "reading an equation"))
        self.assertEqual(other.calls, [])  # a read stays off the policy
        s.run_policy("g05", max_seconds=0.2)  # after B the program may move
        handed = Settling()
        s = self.strict(Worker(step_limit=400, instruction="Play tic-tac-toe as the first player and fill the board with the opponent."), handed)
        trace = s.start("B", "a board game")
        self.assertEqual(s._strategy["strategy"], "A")  # not a read: the policy keeps it
        self.assertEqual(trace.skill, "run_policy")
        self.assertGreater(len(handed.calls), 0)
        from harness_policy.runtime.events import Escalation
        with self.assertRaises(Escalation) as blocked:
            s.run_policy("g05", max_seconds=0.2, from_anywhere=True)
        self.assertEqual(blocked.exception.trace.event.detail["reason"], "the policy has this task")
        with self.assertRaisesRegex(ValueError, "strategy must be one of"):
            self.strict(Worker(), Policy()).start("D", "x")

    def test_every_instruction_routes_own_tools_or_the_simate_base(self):
        # Live gen_instruction text. A task name is not a route.
        cases = {
            "own": [
                "Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm.",
                "Complete the equation by selecting the correct missing number or operator and placing it on the pad, then reset the robot arm.",
                "Arrange the numbers from left to right to form the largest possible number, and place them on the pad.",
                "Pick up the blue race toy car by 10 cm.",
                "Sort the objects by category into the three baskets.",
                "Stack the three blocks with different textures.",
                "stack the blocks from bottom to top in the order ofblue, green, and red, then reset the robot arm.",
                "Arrange the five nesting dolls in a row from left to right, from smallest to largest.",
                "Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.",
                "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.",
                "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the objects.",
                "Wait for the opponent to discard a tile, then declare a kong with the matching tiles.",
                "Observe the object placement order, remember it, then place the corresponding objects into the basket in the same order.",
            ],
            "simate": [
                "Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm.",
                "Build a tower using the wooden blocks and wooden boards.",
                "Put fruit objects into the left basket, tool objects into the middle basket, and food objects into the right basket, then reset the robot arm.",
                "Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue.",
                "Pick up the coin from the holder and insert it precisely into the coin bank.",
                "Insert and tighten each screw into the nut of the same color.",
                "Place the four eggs from the basket into the egg holder, then close the lid.",
                "Hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.",
                "Fold the clothes neatly.",
                "Hang all the mugs on the mug rack.",
                "Insert the three tubes into the rack one by one.",
                "Pick up two slices of bread, place them into the toaster, and press the lever down.",
                "Remember the first object on the conveyor, then pick the matching object when it appears again.",
                "Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside.",
                "Place all the objects into the box with their front sides facing left.",
                "Lift the basket more than 8 cm, identify the target object on the conveyor according to the image on the board, pick it up, and place it into the basket.",
                "Pick up the mallet and strike all xylophone keys from left to right.",
                "Place all stacking toy pieces onto the correct pegs.",
                "Play tic-tac-toe as the first player and fill the board with the opponent.",
                "Plug the charger into the power strip.",
                "Pour all the balls from the cup into the vase.",
                "Pour the liquid from the first red bottle into the first black bowl, from the second turquoise bottle into the second white bowl, and from the third violet bottle into the third brown bowl. Then reset the robot arm.",
                "Pour the liquid from the bottle into the cup.",
                "Push the T-shaped block to align it precisely with the gray T-shaped pad.",
                "Pick up the bottles and throw them into the dustbin, using handover when needed.",
                "Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.",
                "Place each tool into its matching position in the toolbox, then reset the robot arm.",
                "Pick up the two T-shaped blocks, swap their positions, and place them back with the correct orientations.",
                "Swap the two blocks using the empty mat, pressing the button after each move.",
                "Stack the three bowls together.",
                "press_by_number",
                "play_tic_tac_toe",
                "stack_blocks_random",
            ],
        }
        for route, instructions in cases.items():
            for instruction in instructions:
                with self.subTest(instruction=instruction):
                    self.assertEqual(route_instruction(instruction), route)
        self.assertEqual(route_instruction(""), "simate")

    def test_the_switch_runs_the_simate_base_or_own_tools(self):
        simate, other = Settling(), Settling()
        worker = Worker(step_limit=400, instruction="Play tic-tac-toe as the first player and fill the board with the opponent.")
        s = self.strict(worker, other)
        s.policies = {"simate": {"url": "ws://simate"}, "other": {"url": "ws://other"}}
        s._policy_clients = {"simate": simate, "other": other}
        trace = s.start("B", "a board game", policy="other")
        self.assertEqual(s._strategy["strategy"], "A")
        self.assertEqual(s._strategy["policy"], "simate")
        self.assertEqual(trace.skill, "run_policy")
        self.assertGreater(len(simate.calls), 0)
        self.assertEqual(other.calls, [])

        reading = self.strict(Worker(
            step_limit=400,
            instruction="Complete the equation by selecting the correct missing number or operator and placing it on the pad.",
        ), Settling())
        reading.policies = {"simate": {"url": "ws://simate"}, "other": {"url": "ws://other"}}
        reading._policy_clients = {"simate": Settling(), "other": Settling()}
        self.assertIsNone(reading.start("A", "hand it over", policy="other"))
        self.assertEqual(reading._strategy["strategy"], "B")
        self.assertIsNone(reading._strategy["policy"])
        self.assertEqual(reading._policy_clients["simate"].calls, [])
        self.assertEqual(reading._policy_clients["other"].calls, [])

    def test_after_b_the_task_is_handed_over_after_all(self):
        worker, policy = Worker(
            step_limit=400,
            instruction="Complete the equation by selecting the correct missing number or operator and placing it on the pad.",
        ), Settling()
        s = self.strict(worker, policy)
        self.assertIsNone(s.start("B", "a box to fill"))
        trace = s.start("C", "putting objects into a box")  # own tools failed: the policy after all
        self.assertEqual(trace.skill, "run_policy")
        self.assertEqual(s._strategy["strategy"], "C")
        self.assertEqual(s._strategy["after"]["strategy"], "B")
        with self.assertRaisesRegex(ValueError, "already called"):  # back to B is refused
            s.start("B", "again")

    def test_the_program_ends_a_run_whose_tracked_objects_stopped_moving(self):
        worker, policy = Worker(step_limit=400), Policy()  # a policy that keeps moving
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0  # measured every 12 ticks, judged over 24
        s._tracked_positions = lambda: {"bowl": [[0.3, 0.1, 0.8]], "plate": [[0.5, -0.1, 0.77]]}
        trace = s.start("C", "stacking")  # a normal end: the program goes on, the model is not asked
        self.assertEqual(trace.event.kind, "COMPLETED")
        self.assertEqual(trace.event.detail["stopped"], "no_progress")
        self.assertEqual(trace.event.detail["tracked"], ["bowl", "plate"])
        self.assertEqual(trace.end_tick - trace.start_tick, 48)  # two windows to reach the first object
        self.assertEqual(trace.event.detail["progress_checks"], {"none": 3})
        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        s._tracked_positions = lambda: {"bowl": [[0.3 + 0.004 * min(worker.step, 14), 0.1, 0.8]]}
        trace = s.start("C", "stacking")  # it moved the bowl, then nothing for a window: ended at once
        self.assertEqual((trace.event.detail["stopped"], trace.end_tick), ("no_progress", 36))

    def test_objects_that_move_are_progress_hidden_ones_say_nothing(self):
        from harness_policy.runtime.events import EpisodeEnded

        worker, policy = Worker(step_limit=60), Policy()
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        s._tracked_positions = lambda: {"bowl": [[0.3 + 0.004 * worker.step, 0.1, 0.8]]}  # 10 cm a window
        with self.assertRaises(EpisodeEnded):  # left to run while it progresses
            s.start("C", "stacking")
        self.assertEqual(worker.step, 60)
        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        # The cup hidden by the arms from tick 20 says nothing; the bowl that stays put does.
        s._tracked_positions = lambda: {"bowl": [[0.3, 0.1, 0.8]],
                                        "cup": [[0.4, 0.0, 0.8]] if worker.step < 20 else []}
        trace = s.start("C", "hidden")
        self.assertEqual(trace.event.detail["stopped"], "no_progress")
        self.assertEqual(trace.end_tick, 48)

    def test_a_policy_that_waits_while_tracked_objects_move_goes_on(self):
        from harness_policy.runtime.events import EpisodeEnded

        worker, policy = Worker(step_limit=300), Still()  # arms still: waiting for the conveyor
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        s._tracked_positions = lambda: {"phone": [[0.5 - 0.004 * worker.step, 0.3, 0.8]]}  # riding the belt
        with self.assertRaises(EpisodeEnded):  # neither the stall stop (125 ticks) nor the watch ends it
            s.start("A", "picking from a conveyor")
        self.assertEqual(worker.step, 300)
        worker, policy = Worker(step_limit=300), Still()
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        s._tracked_positions = lambda: {"phone": [[0.5, 0.3, 0.8]]}  # nothing moves either: the watch ends it
        trace = s.start("A", "picking from a conveyor")
        self.assertEqual((trace.event.detail["stopped"], trace.end_tick), ("no_progress", 48))

    def test_the_policy_runs_on_to_the_end_when_the_calls_are_used_up(self):
        from harness_policy.runtime.events import EpisodeEnded

        worker, policy = Worker(step_limit=300), Settling()
        s = self.strict(worker, policy)
        s._policy_progress_seconds = 1.0
        s._tracked_positions = lambda: {"bowl": [[0.3, 0.1, 0.8]]}
        trace = s.start("C", "stacking")  # stopped: nothing moved
        self.assertEqual(trace.event.detail["stopped"], "no_progress")
        self.assertTrue(s._strategy["policy_ran"])  # the driver counts calls from here
        with self.assertRaises(EpisodeEnded):  # no stall stop, no watch: to the end
            s.__policy_to_end__()
        self.assertEqual(worker.step, 300)
        self.assertEqual(s._policy_progress_seconds, 1.0)
        s = self.strict(Worker(
            step_limit=300,
            instruction="Press the two red buttons the required number of times according to the number cards.",
        ), Policy())
        s.start("B", "reading numbers")
        self.assertIsNone(s.__policy_to_end__())  # nothing was handed over
        self.assertNotIn("policy_ran", s._strategy)

    def test_without_tracked_objects_or_the_option_only_stopping_ends_the_run(self):
        from harness_policy.runtime.events import EpisodeEnded

        for seconds, measure in ((1.0, lambda: None), (1.0, lambda: {"bowl": []}),
                                 (0.0, lambda: {"bowl": [[0.3, 0.1, 0.8]]})):
            with self.subTest(seconds=seconds):
                worker, policy = Worker(step_limit=60), Policy()
                s = self.strict(worker, policy)
                s._policy_progress_seconds, s._tracked_positions = seconds, measure
                with self.assertRaises(EpisodeEnded):  # it ran until the episode ended
                    s.start("C", "x")
                self.assertEqual(worker.step, 60)

    def test_the_benchmarks_wrap_up_is_not_refused(self):
        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        s.home = lambda arms: [s._execute("home", {}, lambda state, tick: None, 1.0, None, None)] * 2
        worker.joints[6] = worker.joints[13] = 0.0  # jaws left closed (after a push, say)
        s.state = worker.state()
        s._gripper_commands = {"left_arm": 0.0, "right_arm": 0.0}
        kinds = s._wrap_up()
        # RoboDojo scores most tasks only with every gripper open: both open, then home.
        self.assertEqual(len(kinds), 4)
        self.assertEqual(kinds[-2:], ["COMPLETED", "COMPLETED"])
        self.assertEqual((worker.joints[6], worker.joints[13]), (1.0, 1.0))

    def test_the_wrap_up_keeps_a_held_object(self):
        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        s.home = lambda arms: [s._execute("home", {}, lambda state, tick: None, 1.0, None, None)] * 2
        worker.joints[6] = 0.5  # the left jaws closed on an object (measured opening)
        worker.joints[13] = 0.0  # the right jaws closed on nothing
        s.state = worker.state()
        s._gripper_commands = {"left_arm": 0.0, "right_arm": 0.0}
        kinds = s._wrap_up()
        # A lifted object stays in the air (general_pickup scores it there); the empty jaws open.
        self.assertEqual(kinds[0], "HOLDING")
        self.assertLess(worker.joints[6], 0.8)
        self.assertEqual(worker.joints[13], 1.0)

    def test_the_wrap_up_asks_the_vision_model_without_a_measured_opening(self):
        worker, policy = Worker(step_limit=400), Policy()
        s = self.strict(worker, policy)
        s.home = lambda arms: [s._execute("home", {}, lambda state, tick: None, 1.0, None, None)] * 2
        s.profile = {**s.profile, "gripper_feedback": "commanded"}
        worker.joints[6] = worker.joints[13] = 0.0
        s.state = worker.state()
        s._gripper_commands = {"left_arm": 0.0, "right_arm": 0.0}
        looked = []
        s._looked_holding = lambda arm, look: looked.append(arm) or arm == "right_arm"
        kinds = s._wrap_up()
        self.assertEqual(looked, ["left_arm", "right_arm"])
        self.assertEqual(kinds[:2], ["COMPLETED", "HOLDING"])
        self.assertEqual(worker.joints[6], 1.0)
        self.assertLess(worker.joints[13], 0.8)


class Still(Policy):
    """A policy that keeps the arms where they are (the left gripper at `gripper`)."""

    def __init__(self, gripper=1.0):
        super().__init__()
        self.gripper = gripper

    def call(self, name, obs=None):
        self.calls.append(name)
        if name == "get_action":
            return [{"left_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [self.gripper],
                     "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]}] * 8
        return None


class Settling(Still):
    """A policy that moves the left arm's first joint to 0.1 rad, then holds it there."""

    def call(self, name, obs=None):
        chunk = super().call(name, obs)
        if name == "get_action":
            return [{**action, "left_arm_joint_state": [0.1] + [0.0] * 5} for action in chunk]
        return chunk


class Gripping(Worker):
    """Jaws commanded closed read 0.5 (an object between them) for 20 steps, then 0.185
    (closed on nothing: it slipped out)."""

    def __init__(self, step_limit=40):
        super().__init__(step_limit)
        self.closed_at = None

    def call(self, name, **arguments):
        result = super().call(name, **arguments)
        if name == "tick":
            if arguments["command"][6] <= 0.5:
                self.closed_at = self.step if self.closed_at is None else self.closed_at
                self.joints[6] = 0.5 if self.step - self.closed_at < 20 else 0.185
            result["state"] = self.state()
        return result


if __name__ == "__main__":
    unittest.main()
