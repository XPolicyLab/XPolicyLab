"""Offline contract/failure tests. These fixtures are NOT closed-loop scores."""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from harness_policy.config import load_layered
from harness_policy.geometry import TABLE_Z
from harness_policy.runtime.events import EpisodeEnded, Escalation, Event, ExecutionTrace, ProgramFinished
from harness_policy.runtime.prompts import assemble_prompt
from harness_policy.runtime.skills import load_skills, namespace, resolve, skills_for_layers
from skills.complementary import TOOLS, instruction_route, turn_report
from skills.vla import route_instruction
from test_eef import item
from test_run_policy import Worker, Policy, session
from test_session_skills import FakeWorker, pressing_session

T = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
ALIGN = "Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm."
PRESS = "Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm."
BASKET = "Put car objects into the left basket, watch objects into the middle basket, and wooden toy objects into the right basket."
SWAP = "Swap the two blocks using the empty mat, pressing the button after each move."
XYLOPHONE = "Pick up the mallet and strike all xylophone keys from left to right."
SWEEP = "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks."


def comp_session(instruction=T):
    worker, policy = Worker(step_limit=200, instruction=instruction), Policy()
    s = session(worker, policy)
    s.skills = ["core", "eef", "scene", "vla", "complementary"]
    s._policy_span, s._policy_action_mode, s._fixed_task_route = "cell", "joint", None
    s.policies = {"wall-oss": {"url": "ws://127.0.0.1:19999"}}
    s._policy_clients = {"wall-oss": policy}
    s._strategy_first = True
    s._start_joints = list(worker.joints)
    s.profile = {"return_to_start": True, "arm_bases": {"left_arm": [-.3, -.45, .765], "right_arm": [.3, -.45, .765]}}
    s._eef_grip, s._eef_scene, s._eef_measured, s._eef_measure_serial = {}, [], {"step": 0}, 1
    s._held, s._policy_progress_seconds = {}, 0
    return s


def evidence(s, **extra):
    return {"source": "fixture RGB interpretation", "episode_id": s.episode_id,
            "observed_step": s.state["step"], **extra}


def boundary(s, **extra):
    return evidence(s, empty_hands=True, stable=True, exposed=True, reachable=True, measured=True, **extra)


def trace(s, kind="COMPLETED"):
    step = s.state["step"]
    s.state["step"] += 1
    return ExecutionTrace("motion", {}, step, step+1, Event(kind, "motion", {}, step+1))


def measured(s, objects):
    s._eef_scene = objects
    s._eef_measured = {"step": s.state["step"]}
    s._eef_measure_serial += 1


class RouteTests(unittest.TestCase):
    def test_native_instructions_use_capabilities(self):
        for text in (T, ALIGN, PRESS, BASKET, SWAP, SWEEP):
            with self.subTest(text=text):
                self.assertEqual(instruction_route(text)["owner"], "B")
        self.assertFalse(instruction_route(SWAP)["memory"])
        self.assertTrue(instruction_route(SWAP)["identity_sensitive"])
        self.assertFalse(instruction_route(T)["hard"])
        self.assertTrue(instruction_route(PRESS)["hard"])
        self.assertFalse(instruction_route(PRESS)["memory"])

    def test_route_owners_and_semantic_hard_flags(self):
        for text, owner, rule in (("Fasten screws with nuts.", "A", "wall_preserve"),
                                  ("Lift the named car by 10 cm.", "A", "wall_preserve"),
                                  (BASKET, "B", "semantic_read_binding"),
                                  (PRESS, "B", "semantic_read_binding"),
                                  (T, "B", "measured_complement"),
                                  ("Make a kong after the opponent discards.", "A", "wall_preserve")):
            with self.subTest(text=text):
                route = instruction_route(text)
                self.assertEqual((route["owner"], route["rule"]), (owner, rule))
        for text in ("Solve the equation.", "Arrange digits into the largest number.",
                     "Pour from the blue bottle into the red bowl."):
            self.assertTrue(instruction_route(text)["hard"])

    def test_named_10cm_pickup_preserves_wall_without_own_motion(self):
        for text in ("Pick up the blue race toy car by 10 cm.",
                     "Pick up the speckled ceramic dove by 10 cm.",
                     "Lift the named car BY 10 CM."):
            with self.subTest(instruction=text):
                route = instruction_route(text)
                self.assertEqual(route["owner"], "A")
                self.assertFalse(route["hard"])
                s = comp_session(text)
                s.start("B", "model requested pickup complement")
                self.assertEqual(s._strategy["strategy"], "A")
                self.assertEqual(s._comp()["route"]["instruction"], text)
                self.assertEqual(s._ownership_guard("eef_pick"), "policy_owner_no_own_motion")
                self.assertEqual(s._policy_clients["wall-oss"].calls, [])

    def test_generic_swap_is_not_kfmem(self):
        self.assertFalse(instruction_route("Swap the bowls.")["memory"])
        self.assertFalse(instruction_route("Exchange positions of the cups.")["memory"])
        self.assertTrue(instruction_route("Swap the two T-shaped objects.")["memory"])
        for text in ("Swap the positions of the two T-shaped blocks using the empty mat.",
                     "Exchange positions of the T-shaped blocks on their aligned pads."):
            route = instruction_route(text)
            self.assertEqual(route["owner"], "A")
            self.assertTrue(route["memory"])
            self.assertFalse(route["identity_sensitive"])

    def test_xylophone_and_mallet_strikes_preserve_wall(self):
        for text in (XYLOPHONE, "Play the xylophone using the mallet.",
                     "Play the xylophone, striking each key with the mallet.",
                     "PLAY THE XYLOPHONE.", "Use the mallet to strike each bar.",
                     "Strike the bars with the mallet.", "The mallet strikes each bar.",
                     "Use the mallet,\nstriking the bars in order."):
            with self.subTest(instruction=text):
                route = instruction_route(text)
                self.assertEqual((route["owner"], route["rule"]), ("A", "wall_preserve"))
                for flag in ("hard", "memory", "identity_sensitive", "precision", "tail"):
                    self.assertFalse(route[flag], flag)
                self.assertIsNone(TOOLS.search(text))
                self.assertEqual(route_instruction(text, wall_only=True), "simate")

    def test_striking_does_not_open_measured_or_tail_ownership(self):
        for suffix in (SWEEP, ALIGN, T, SWAP, "Pack objects into the box."):
            with self.subTest(suffix=suffix):
                route = instruction_route(XYLOPHONE + " " + suffix)
                self.assertEqual((route["owner"], route["rule"]), ("A", "wall_preserve"))
                self.assertFalse(route["identity_sensitive"])
                self.assertFalse(route["tail"])

    def test_striking_keeps_explicit_hard_and_memory_precedence(self):
        for prefix in (XYLOPHONE, "Use the mallet to strike each bar."):
            for suffix in (PRESS, BASKET, "Solve the equation.",
                           "Arrange digits into the largest possible number.",
                           "Pour from the blue bottle into the red bowl."):
                with self.subTest(prefix=prefix, suffix=suffix):
                    route = instruction_route(prefix + " " + suffix)
                    self.assertEqual((route["owner"], route["rule"]), ("B", "semantic_read_binding"))
                    self.assertTrue(route["hard"])
            for suffix in ("Remember the striking sequence.", "Imitate the sequence.",
                           "Swap the two T-shaped blocks."):
                with self.subTest(prefix=prefix, suffix=suffix):
                    route = instruction_route(prefix + " " + suffix)
                    self.assertEqual((route["owner"], route["rule"]), ("A", "wall_preserve"))
                    self.assertTrue(route["memory"])
                    self.assertFalse(route["identity_sensitive"])
            route = instruction_route(prefix + " Remember the sequence from the number cards.")
            self.assertTrue(route["hard"])
            self.assertFalse(route["memory"])
            self.assertEqual(route["owner"], "B")

    def test_broom_and_dustpan_sweeping_keep_measured_ownership(self):
        for text in (SWEEP, "Sweep the blocks with the broom.",
                     "Use the broom to sweep the objects.", "Sweep the objects into the dustpan."):
            with self.subTest(instruction=text):
                route = instruction_route(text)
                self.assertIsNotNone(TOOLS.search(text))
                self.assertEqual((route["owner"], route["rule"]), ("B", "measured_complement"))
                self.assertFalse(route["hard"])

    def test_xylophone_starts_wall_without_tool_measurement_or_failure_handover(self):
        for text in (XYLOPHONE, "Use the mallet to strike each bar."):
            with self.subTest(instruction=text):
                s = comp_session(text)
                with mock.patch.object(s, "tool_contact") as contact, \
                     mock.patch.object(s, "failure_route") as failure:
                    s.start("B", "model requested tool contact")
                    status = s.ownership_status()["status"]
                    self.assertEqual((status["owner"], status["phase"], status["rule"]),
                                     ("A", "policy", "wall_preserve"))
                    self.assertEqual(s._strategy["policy"], "wall-oss")
                    self.assertEqual(status["wall_returns"], 0)
                    self.assertEqual(status["transitions"], [])
                    self.assertEqual(s._ownership_guard("eef_pick"), "policy_owner_no_own_motion")
                    self.assertIsNone(s._ownership_guard("run_policy"))
                    for _ in range(2):
                        s.run_policy("wall-oss", max_seconds=.2, stall_seconds=None)
                    contact.assert_not_called()
                    failure.assert_not_called()
                self.assertEqual(set(s._policy_clients["wall-oss"].instructions), {text})
                self.assertEqual(s._comp()["failures"], {})
                self.assertEqual(s._comp()["route"]["wall_returns"], 0)

    def test_precision_never_opens_motion_even_with_an_own_keyword(self):
        for text in ("Fasten the screws by 10 cm.", "Build a tower with the largest number on top.",
                     "Insert the key into the hole.", "Plug in the charger.", "Insert tubes.",
                     "Place the stacking toy over its pegs."):
            route = instruction_route(text)
            self.assertTrue(route["precision"])
            self.assertEqual(route["owner"], "A")

    def test_keepers_precision_dynamic_are_not_own(self):
        for text in ("Stack bowls.", "Sort objects by category.", "Stack blocks bottom to top in the order ofblue, red.",
                     "Pour liquid into the cup.", "Fasten screws with nuts.", "Insert the key.",
                     "Build a tower.", "Swap the two T-shaped objects.", "Remember and cover blocks.",
                     "Imitate the sorting sequence.", "Match and pick from the conveyor.", "make_kong"):
            with self.subTest(text=text):
                self.assertEqual(instruction_route(text)["owner"], "A")

    def test_static_tail_only_no_memory_precision(self):
        for text in ("Fill the pen holder.", "Pack objects into the box.", "Store laptop and headphones.",
                     "Place all the objects into the box with their front sides facing left.",
                     "Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand."):
            self.assertEqual(instruction_route(text)["owner"], "C")
        self.assertEqual(instruction_route("Remember objects and pack them into the box.")["owner"], "A")
        self.assertEqual(instruction_route("Insert key into box.")["owner"], "A")

    def test_benchmark_ids_do_not_select_routes(self):
        for text in ("push_T", "align_blocks", "press_by_number", "swap_blocks", "classify_objects_by_language"):
            self.assertEqual(instruction_route(text)["owner"], "A")

    def test_skill_optional_namespace_complete_default_unchanged(self):
        defaults = skills_for_layers({"learned": True})
        self.assertNotIn("complementary", defaults)
        resolution = resolve(defaults + ["complementary"], {"segmenter", "vlm", "pointer", "policy", "verifier", "world_model"}, strict=True)
        s = comp_session()
        exports = namespace(s, ["complementary"])
        self.assertEqual(set(exports), set(load_skills()["complementary"].exports))
        self.assertIn("complementary", resolution.enabled)
        self.assertIn("swap_facing_correct", exports)
        self.assertNotIn("_orient_held", exports)
        self.assertNotIn("_swap_motion_unknown", exports)
        wall = load_layered(ROOT / "configs/ablations/wall-oss.toml")
        self.assertIn("complementary", wall["skills"]["enable"])

    def test_counted_success_and_measured_swap_recovery_prompt_contract(self):
        prompt = assemble_prompt(ROOT / "prompts/robodojo", ["core", "scene", "eef", "vla", "complementary"])
        prompt += (ROOT / "prompts/robodojo/wall_cell.md").read_text()
        for rule in ("Do NOT invent", "count_complete and arm_clear are independent",
                     "that leg is CONFIRMED", "ONE measured correction per leg",
                     "Two failed checks alone", "NO nearest-oN relabeling", "no motion retry"):
            self.assertIn(rule, prompt)
        task_prompt = (ROOT / "prompts/robodojo_tasks/swap_blocks.md").read_text()
        for text in (prompt, task_prompt):
            for rule in ("swap_facing_correct", 'stage="picked"', "CURRENT held yaw",
                         "held_entry", "reobserve_rebind_then_swap_confirm", "BEFORE"):
                self.assertIn(rule, text)
            self.assertIn("does NOT consume", text)
            self.assertIn("BEFORE orient", text)
            self.assertNotIn("BEFORE pick", text)

    def test_xylophone_prompt_preserves_wall_without_contact_gate(self):
        text = (ROOT / "prompts/robodojo/wall_cell.md").read_text()
        for rule in ("Xylophone / mallet striking sequences are A / wall_preserve",
                     "unless explicit hard", "or memory rules match", "do NOT first call tool_contact",
                     "do NOT require", "handover to begin", "Sweeping requires actual broom"):
            self.assertIn(rule, text)
        self.assertNotIn("Xylophone requires held mallet and actual mallet-head offset", text)
        self.assertNotIn("Xylophone tool_contact_measurement_missing", text)

    def test_three_confirmed_swap_legs_prompt_requires_bounded_closeout(self):
        for path in ("prompts/robodojo/wall_cell.md", "prompts/robodojo_tasks/swap_blocks.md"):
            with self.subTest(path=path):
                text = (ROOT / path).read_text()
                for rule in ("confirmed_legs >= 3", 'next_hop="complete_task"',
                             'next_hop="observe_swap_completion"', "at most ONE bounded non-motion",
                             "complete_task, NOT another transfer or facing correction",
                             "move_tip", "unbounded browsing", "swap_completion_unverified"):
                    self.assertIn(rule, text)

    def test_swap_prompt_certifies_completeness_without_a_verifier_veto(self):
        for path in ("prompts/robodojo/wall_cell.md", "prompts/robodojo_tasks/swap_blocks.md"):
            with self.subTest(path=path):
                text = (ROOT / path).read_text()
                for rule in ("three swap_confirm receipts + ledger done", "verify() to veto closeout",
                             "evidence.complete=False", "fresh_visual_evidence", "safe_to_finish",
                             "not_failed", "motion_resolved", "identity_items_done", "cards_confirmed",
                             "SAME cell", "complete_task -> finish", "do NOT open another observation cell"):
                    self.assertIn(rule, text)
                self.assertNotIn('"complete":True', text)
                self.assertNotIn("complete:True", text)
        wall = (ROOT / "prompts/robodojo/wall_cell.md").read_text()
        self.assertIn("For non-swap work, a verifier exception", wall)
        self.assertIn("retain receipt/ledger completeness despite verifier failure", wall)


class ReadHelperTests(unittest.TestCase):
    def setUp(self):
        self.s = comp_session("Solve the equation by placing the missing number on the pad.")
        self.expression = "? + 2 = 5"
        self.candidates = [{"key": "three", "token": "3", "value": 3},
                           {"key": "four", "token": "4", "value": 4}]

    def solve(self, **extra):
        return self.s.equation_solve(self.expression, self.candidates,
                    evidence(self.s, expression=self.expression, candidates=self.candidates,
                             read_confident=True, identity_unique=True, **extra))

    def test_equation_exact_no_motion_uniform_receipt(self):
        r = self.solve()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["answer_token"], "3")
        self.assertEqual(self.s.state["step"], 0)
        self.assertFalse(r["motion_executed"])
        for key in ("ok", "reason", "event", "next_hop", "receipt"):
            self.assertIn(key, r)

    def test_missing_operator_and_fraction_check(self):
        self.expression = "6 ? 4 = 3 / 2"
        self.candidates = [{"key": "div", "token": "÷", "value": "/"},
                           {"key": "times", "token": "×", "value": "*"}]
        self.assertEqual(self.solve()["answer_token"], "÷")

    def test_equation_ambiguous_duplicate_or_no_solution(self):
        self.assertFalse(self.solve(ambiguous=True)["ok"])
        self.candidates.append({"key": "other-three", "token": "3", "value": 3})
        self.assertEqual(self.solve()["reason"], "equation_no_unique_answer")
        self.candidates = [{"key": "four", "token": "4", "value": 4}]
        self.assertFalse(self.solve()["ok"])

    def test_equation_grammar_never_executes_python(self):
        for text in ("__import__('os').system('?') = 5", "? ** 2 = 9", "? + x = 5", "?? = 3", "? = 3 = 3"):
            self.expression = text
            self.assertFalse(self.solve()["ok"], text)
        self.assertEqual(self.s.state["step"], 0)

    def test_equation_stale_mismatched_or_missing_evidence(self):
        self.assertFalse(self.s.equation_solve(self.expression, self.candidates, None)["ok"])
        e = evidence(self.s, expression="? + 3 = 5", candidates=self.candidates,
                     read_confident=True, identity_unique=True)
        self.assertFalse(self.s.equation_solve(self.expression, self.candidates, e)["ok"])
        e["expression"], e["observed_step"] = self.expression, -1
        self.assertFalse(self.s.equation_solve(self.expression, self.candidates, e)["ok"])

    def test_equation_token_cannot_disagree_with_digit_value(self):
        self.candidates[0]["token"] = "4"
        self.assertEqual(self.solve()["reason"], "equation_invalid")

    def test_largest_preserves_duplicate_digit_keys_and_zero(self):
        digits = [{"key": "zero", "value": 0}, {"key": "nine-a", "value": 9}, {"key": "nine-b", "value": 9}]
        r = self.s.largest_order(digits, evidence(self.s, digits=digits, read_confident=True, identity_unique=True))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["ordered_keys"], ["nine-a", "nine-b", "zero"])
        self.assertEqual(r["ordered_values"], [9, 9, 0])
        self.assertFalse(r["motion_executed"])

    def test_largest_invalid_ocr_duplicate_identity_or_stale(self):
        for digits in ([], [{"key": "a", "value": True}], [{"key": "a", "value": "9"}],
                       [{"key": "a", "value": 10}], [{"key": "a", "value": 9}, {"key": "a", "value": 8}]):
            self.assertFalse(self.s.largest_order(digits, evidence(self.s, digits=digits,
                             read_confident=True, identity_unique=True))["ok"])
        self.assertFalse(self.s.largest_order([{"key": "a", "value": 9}], None)["ok"])

    def cards(self, left=2, **extra):
        return evidence(self.s, cards=[{"side": "middle", "value": 3, "target": "o2"},
                                     {"side": "left", "value": left, "target": "o1"}],
                        blue_confirm="o3", read_confident=True, target_unique=True, **extra)

    def test_read_cards_order_and_press_binding(self):
        self.s = comp_session(PRESS)
        self.s.start("B", "press")
        r = self.s.read_cards(self.cards())
        self.assertTrue(r["ok"], r)
        self.assertEqual([row["count"] for row in r["cards"]], [2, 3])
        self.assertEqual(r["blue_confirm"], "o3")
        self.assertFalse(r["motion_executed"])
        self.assertFalse(self.s.press_card(0, 3, "o1", evidence=evidence(self.s, read_confident=True, target_unique=True))["ok"])
        self.assertEqual(self.s._comp()["cards"], [])

    def test_read_cards_ambiguity_and_changed_count_after_attempt(self):
        self.assertFalse(self.s.read_cards(self.cards(ambiguous=True))["ok"])
        self.assertFalse(self.s.read_cards(self.cards(True))["ok"])
        self.assertTrue(self.s.read_cards(self.cards())["ok"])
        self.s._comp()["cards"].append({"status": "attempted"})
        self.assertEqual(self.s.read_cards(self.cards(4))["reason"], "card_read_changed_after_motion")
        e = self.cards()
        e["blue_confirm"] = "o1"
        self.assertFalse(self.s.read_cards(e)["ok"])

    def test_blue_binding_rejected_without_consuming_confirm(self):
        self.s = comp_session(PRESS)
        self.s.read_cards(self.cards())
        self.s._comp()["cards"].append({"status": "pressed_unconfirmed"})
        self.assertEqual(self.s.confirm_card(0, "wrong", evidence=evidence(self.s, count_matches=True,
                         target_unique=True))["reason"], "blue_confirm_binding_mismatch")
        self.assertEqual(self.s._comp()["cards"][0]["status"], "pressed_unconfirmed")


class PadTests(unittest.TestCase):
    def setUp(self):
        self.s = comp_session("Solve the equation with a missing number.")
        self.s.start("B", "equation")
        source, pad = item("o1", -.1, -.2), item("o2", .1, -.2, name="pad")
        source["reach"] = pad["reach"] = {"left_arm": "down"}
        pad["container"] = False
        measured(self.s, [source, pad])
        self.entries = [{"key": "answer", "identity": "unique three", "object": "o1",
                         "destination": "answer pad", "destination_object": "o2", "yaw_deg": None}]
        self.bind()

    def bind(self):
        self.assertTrue(self.s.identity_ledger(self.entries, evidence(self.s, identity_unique=True, directed_axis=True))["ok"])

    def e(self, **extra):
        return evidence(self.s, identity_unique=True, pad_measured=True, path_clear=True, **extra)

    def pick(self, *args, **kwargs):
        self.s.state["step"] += 1
        return {"ok": True, "raw": "pick"}

    def release(self, *args, **kwargs):
        self.s.state["step"] += 1
        return {"ok": True, "raw": "release"}

    def test_place_receipts_then_fresh_inside_and_seated_commit(self):
        with mock.patch.object(self.s, "eef_pick", side_effect=self.pick), mock.patch.object(self.s, "eef_release", side_effect=self.release) as release:
            r = self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())
        self.assertTrue(r["ok"], r)
        release.assert_called_once_with("left_arm", onto="o2", withdraw=False)
        self.assertEqual(r["receipt"]["pick"]["raw"], "pick")
        self.assertEqual(self.s._comp()["ledger"]["answer"]["status"], "placed_unverified")
        self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["ok"])
        e = evidence(self.s, identity_unique=True, destination_correct=True, holding=False, inside=True, seated=True)
        self.assertFalse(self.s.placement_check("answer", e)["ok"])
        measured(self.s, self.s._eef_scene)
        self.bind()
        self.assertFalse(self.s.placement_check("answer", {**e, "inside": False})["ok"])
        self.assertTrue(self.s.placement_check("answer", e)["ok"])

    def test_missing_reach_pad_or_orientation_never_moves(self):
        with mock.patch.object(self.s, "eef_pick") as pick:
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=None)["ok"])
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "wrong", evidence=self.e())["ok"])
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", yaw_deg=20, evidence=self.e())["ok"])
            self.s._eef_scene[1]["container"] = True
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["ok"])
            pick.assert_not_called()

    def test_changed_locate_serial_requires_rebind_even_same_step(self):
        measured(self.s, self.s._eef_scene)
        with mock.patch.object(self.s, "eef_pick") as pick:
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["ok"])
            pick.assert_not_called()

    def test_failed_grasp_no_release_unknown_exception_no_retry(self):
        with mock.patch.object(self.s, "eef_pick", return_value={"ok": False}), mock.patch.object(self.s, "eef_release") as release:
            self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["ok"])
            release.assert_not_called()
        self.assertEqual(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["reason"], "same_failed_action_no_retry")
        self.setUp()
        with mock.patch.object(self.s, "eef_pick", side_effect=TimeoutError("motion connection lost")):
            r = self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())
        self.assertEqual(r["motion_executed"], "unknown")
        self.assertIn("exception", r["receipt"])
        self.assertFalse(self.s.place_on_pad("answer", "left_arm", "o2", evidence=self.e())["ok"])

    def test_held_yaw_receipt_and_required_facing(self):
        self.entries[0]["yaw_deg"] = 20
        self.bind()
        self.s._eef_scene[0]["held_by"] = "left_arm"
        self.s._eef_grip["left_arm"] = (0, 0, .04)
        with mock.patch.object(self.s, "eef_pick") as pick, mock.patch.object(self.s, "eef_release", side_effect=self.release), mock.patch.object(self.s, "move_eef", side_effect=lambda *a, **k: trace(self.s)):
            r = self.s.place_on_pad("answer", "left_arm", "o2", yaw_deg=20,
                                   evidence=self.e(held_identity=True, orientation_measured=True, yaw_deg=0))
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["receipt"]["orient"]["ok"])
        pick.assert_not_called()
        measured(self.s, self.s._eef_scene)
        self.bind()
        e = evidence(self.s, identity_unique=True, destination_correct=True, holding=False, inside=True, seated=True)
        self.assertFalse(self.s.placement_check("answer", e)["ok"])
        self.assertTrue(self.s.placement_check("answer", {**e, "facing_correct": True})["ok"])

    def test_nested_orientation_unknown_stays_unknown_and_no_release(self):
        self.entries[0]["yaw_deg"] = 20
        self.bind()
        self.s._eef_scene[0]["held_by"] = "left_arm"
        self.s._eef_grip["left_arm"] = (0, 0, .04)
        with mock.patch.object(self.s, "move_eef", side_effect=TimeoutError("possibly executed yaw")), mock.patch.object(self.s, "eef_release") as release:
            r = self.s.place_on_pad("answer", "left_arm", "o2", yaw_deg=20,
                                   evidence=self.e(held_identity=True, orientation_measured=True, yaw_deg=0))
        self.assertEqual(r["motion_executed"], "unknown")
        self.assertTrue(self.s._comp()["uncertain"])
        release.assert_not_called()

    def test_nested_orientation_preserves_outer_receipt_guard(self):
        self.entries[0]["yaw_deg"] = 20
        self.bind()
        self.s._eef_scene[0]["held_by"] = "left_arm"
        self.s._eef_grip["left_arm"] = (0, 0, .04)
        def guarded_release(*args, **kwargs):
            self.assertTrue(self.s._comp_active)
            return self.release(*args, **kwargs)
        with mock.patch.object(self.s, "move_eef", side_effect=lambda *a, **k: trace(self.s)), mock.patch.object(self.s, "eef_release", side_effect=guarded_release):
            self.assertTrue(self.s.place_on_pad("answer", "left_arm", "o2", yaw_deg=20,
                            evidence=self.e(held_identity=True, orientation_measured=True, yaw_deg=0))["ok"])
        self.assertFalse(self.s._comp_active)


class OwnershipTests(unittest.TestCase):
    def test_missing_initial_instruction_cannot_grant_wall_motion(self):
        s = comp_session("")
        with self.assertRaisesRegex(ValueError, "native_instruction_unavailable"):
            s.run_policy("wall-oss", max_seconds=.1)
        self.assertEqual(s._policy_clients["wall-oss"].calls, [])
        self.assertEqual(s.state["step"], 0)

    def test_explicit_instruction_remeasure_before_first_motion_recovers_gate(self):
        s = comp_session("")
        s.start("A", "observation unavailable")
        s.env.instruction = PRESS
        self.assertTrue(s.ownership_status()["ok"])
        self.assertEqual(s._strategy["strategy"], "B")
        with self.assertRaises(ValueError):
            s.run_policy("wall-oss", max_seconds=.1)

    def test_a_is_real_lock_before_any_motion(self):
        s = comp_session("Stack the bowls.")
        s.start("B", "requested own")
        self.assertEqual(s._strategy["strategy"], "A")
        with self.assertRaises(Escalation) as raised:
            s._execute("move_eef", {}, lambda *_: None, 1, None, None)
        self.assertIn("policy_owner", raised.exception.trace.event.detail["failure_reason"])
        self.assertEqual(s.state["step"], 0)

    def test_hard_direct_call_and_rewrite_rejected_before_client(self):
        for text in (PRESS, BASKET, "Pour liquid from the violet bottle into the black bowl."):
            s = comp_session(text)
            with self.assertRaises(ValueError):
                s.run_policy("wall-oss", instruction="Stack bowls.")
            self.assertEqual(s._policy_clients["wall-oss"].calls, [])
            self.assertEqual(s._strategy["strategy"], "B")

    def test_original_route_latched_on_missing_later_instruction(self):
        s = comp_session(BASKET)
        s.start("A", "first")
        s.env.instruction = ""
        s.start("C", "escape")
        self.assertEqual(s._comp()["route"]["owner"], "B")
        self.assertEqual(s._comp()["route"]["instruction"], BASKET)

    def test_wall_argument_red_lines(self):
        s = comp_session("Fasten the screws.")
        s.start("A", "screws")
        for kwargs in ({"instruction": "subtask"}, {"reset": True}, {"arms": ["left_arm"]},
                       {"until": lambda _: True}, {"from_anywhere": True}):
            with self.assertRaises(ValueError):
                s.run_policy("wall-oss", max_seconds=.1, **kwargs)
        self.assertEqual(s._policy_clients["wall-oss"].calls, [])

    def test_wall_continuous_slices_disable_watch_and_keep_original(self):
        s = comp_session("Stack bowls.")
        s.start("A", "stack")
        s._progress_watch = mock.Mock(side_effect=AssertionError("watch forbidden"))
        s.run_policy("wall-oss", max_seconds=.2)
        s.run_policy("wall-oss", max_seconds=.2)
        self.assertEqual(s._policy_clients["wall-oss"].calls.count("reset"), 1)
        self.assertEqual(set(s._policy_clients["wall-oss"].instructions), {"Stack bowls."})
        s._progress_watch.assert_not_called()

    def test_memory_forbids_time_advancing_watch(self):
        s = comp_session("Remember and cover the blocks.")
        s.start("A", "memory")
        self.assertIn("memory_owner", s._ownership_guard("watch_scene"))

    def test_a_slice_timeout_never_exhausts_local_failure_budget(self):
        for text in ("Fasten screws.", "Make a kong.", "Build a tower.", "Cover the blocks.",
                     "Match and pick from the conveyor.", "Swap the T-shaped objects."):
            s = comp_session(text)
            s.start("A", "continuous")
            for _ in range(4):
                result = s.failure_route("policy_timeout", request="finish")
                self.assertEqual(result["next_hop"], "continue_same_wall")
            self.assertNotEqual(s._comp()["completion"], "failed")

    def test_failed_finish_no_automatic_home(self):
        s = comp_session()
        s.start("B", "push")
        s.home = mock.Mock(side_effect=AssertionError("home rescue forbidden"))
        with self.assertRaises(ProgramFinished) as raised:
            s.finish("not complete")
        self.assertIn("FAILED", raised.exception.summary)
        self.assertEqual(s._wrap_up(), ["RECOVERY_FORBIDDEN"])
        s.home.assert_not_called()

    def test_only_observed_completion_allows_final_wrap(self):
        s = comp_session()
        s.start("B", "push")
        self.assertFalse(s.complete_task(evidence(s, complete=True))["ok"])
        self.assertTrue(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])
        s._wrapping_up = True
        self.assertIsNone(s._ownership_guard("home"))

    def test_non_swap_still_requires_explicit_visual_complete_true(self):
        s = comp_session()
        s.start("B", "push")
        for fields in ({}, {"complete": False}, {"complete": None}, {"complete": 1}):
            with self.subTest(fields=fields):
                result = s.complete_task(evidence(s, safe_to_finish=True, **fields))
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "completion_unverified")
                self.assertEqual(s._comp()["completion"], "unknown")
        self.assertTrue(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])


class FailureTests(unittest.TestCase):
    def test_uniform_observable_event_and_bounded_remeasure(self):
        s = comp_session()
        for i in range(3):
            result = s.failure_route("perception_uncertain")
            self.assertEqual(result["failure_count"], i+1)
        self.assertEqual(result["next_hop"], "finish_honest_failure")
        for key in ("episode_id", "observed_step", "motion_executed", "receipt", "event", "ownership", "failure_reason"):
            self.assertIn(key, result)
        self.assertEqual(result["event"]["kind"], "SKILL_ERROR")
        json.dumps(result)
        self.assertIn("perception_uncertain", turn_report(s)["text"])

    def test_uncertain_motion_freezes_then_changed_signature_only(self):
        s = comp_session()
        s.start("B", "push")
        action = mock.Mock(side_effect=TimeoutError("possibly sent"))
        result = s._comp_motion("test", ["same"], action, evidence(s))
        self.assertEqual(result["motion_executed"], "unknown")
        self.assertEqual(result["receipt"]["exception_type"], "TimeoutError")
        self.assertIn("uncertain", s._ownership_guard("move_eef"))
        s.failure_route("inspection", evidence=evidence(s, resolved_motion=True))
        result = s._comp_motion("test", ["same"], action, evidence(s))
        self.assertEqual(result["reason"], "same_failed_action_no_retry")
        self.assertEqual(action.call_count, 1)
        self.assertTrue(s._comp_motion("test", ["different"], lambda: trace(s), evidence(s))["ok"])

    def test_nested_failed_receipt_keeps_motion_uncertain(self):
        s = comp_session()
        s.start("B", "push")
        result = s._comp_motion("test", [1], lambda: {"ok": False, "reason": "lift failed",
                    "receipt": trace(s, "TIMEOUT").to_dict()}, evidence(s))
        self.assertEqual(result["motion_executed"], "unknown")

    def test_safe_wall_return_once_no_reset_or_fake_memory_restart(self):
        s = comp_session()
        s.start("B", "push")
        s._policy_rests = {"wall-oss": {"actions": ["stale"]}}
        result = s.failure_route("planar_capability_missing", request="return_wall", evidence=boundary(s))
        self.assertEqual(result["next_hop"], "return_wall_original_instruction_both_arms")
        self.assertEqual(s._comp()["route"]["owner"], "A")
        self.assertNotIn("wall-oss", s._policy_rests)
        self.assertTrue(s._comp()["route"]["transitions"][-1]["memory_restart_pending"])
        self.assertNotIn("memory_restarts", result)
        again = s.failure_route("again", request="return_wall", evidence=boundary(s))
        self.assertEqual(again["next_hop"], "wall_fallback_denied")

    def test_hard_memory_precision_never_return_wall(self):
        for text in (PRESS, BASKET, SWAP, "Solve the equation.", "Arrange digits into the largest number.",
                     "Remember and cover the blocks.", "Insert key into hole."):
            s = comp_session(text)
            s.start("B", "own")
            result = s.failure_route("failed", request="return_wall", evidence=boundary(s))
            self.assertEqual(result["next_hop"], "wall_fallback_denied")

    def test_tool_planar_failures_return_wall_with_original_instruction_both_arms(self):
        cases = ((SWEEP, "tool_contact_measurement_missing", False),
                 (ALIGN, "tool_contact_measurement_missing", False),
                 (ALIGN, "grasp_failed", True), (T, "planar_contact_unknown", False))
        for text, reason, moved in cases:
            with self.subTest(instruction=text, reason=reason):
                s = comp_session(text)
                s.start("B", "measured contact")
                receipt = {"ok": False, "reason": reason, "motion_executed": moved}
                r = s.failure_route(reason, motion_executed=moved, receipt=receipt,
                                    evidence=boundary(s), request="return_wall")
                self.assertEqual(r["next_hop"], "return_wall_original_instruction_both_arms")
                self.assertEqual(r["receipt"], receipt)
                self.assertEqual(r["motion_executed"], moved)
                self.assertEqual(s._comp()["completion"], "unknown")
                self.assertEqual(s._policy_clients["wall-oss"].calls, [])
                for kwargs in ({"instruction": "rewritten subtask"}, {"arms": ["left_arm"]}, {"reset": True}):
                    with self.assertRaises(ValueError):
                        s.run_policy("wall-oss", max_seconds=.2, from_anywhere=True, **kwargs)
                s.run_policy("wall-oss", max_seconds=.2, stall_seconds=None, from_anywhere=True)
                self.assertEqual(set(s._policy_clients["wall-oss"].instructions), {text})

    def test_tool_return_wall_denied_for_unknown_motion_or_missing_boundary(self):
        for text in (ALIGN, SWEEP, T):
            with self.subTest(instruction=text):
                s = comp_session(text)
                r = s.failure_route("tool_contact_measurement_missing", motion_executed="unknown",
                                    request="return_wall", evidence=boundary(s))
                self.assertEqual(r["next_hop"], "wall_fallback_denied")
                self.assertTrue(s._comp()["uncertain"])
                self.assertEqual(s._comp()["route"]["owner"], "B")
                for key in ("empty_hands", "stable", "exposed", "reachable", "measured"):
                    s = comp_session(text)
                    e = boundary(s)
                    del e[key]
                    r = s.failure_route("tool_contact_measurement_missing", request="return_wall", evidence=e)
                    self.assertEqual(r["next_hop"], "wall_fallback_denied")
                    self.assertEqual(s._comp()["route"]["owner"], "B")

    def test_stale_handover_and_holding_rejected(self):
        s = comp_session()
        s.start("B", "push")
        e = boundary(s)
        s.state["step"] += 1
        self.assertEqual(s.failure_route("failed", request="return_wall", evidence=e)["next_hop"], "wall_fallback_denied")
        s._eef_grip["left_arm"] = (0, 0, .05)
        self.assertEqual(s.failure_route("failed", request="return_wall", evidence=boundary(s))["next_hop"], "wall_fallback_denied")

    def test_irreversible_count_spill_and_memory_finish_even_unknown(self):
        for reason in ("count_uncertain", "confirm_uncertain", "spill", "memory_identity_lost"):
            s = comp_session(PRESS)
            result = s.failure_route(reason, motion_executed="unknown")
            self.assertEqual(result["next_hop"], "finish_honest_failure")
            self.assertEqual(s._comp()["completion"], "failed")

    def test_measured_failures_remeasure_before_requested_finish_but_stay_bounded(self):
        for instruction, reason in ((T, "planar_contact_unknown"), (ALIGN, "tool_contact_measurement_missing"),
                                    (SWAP, "staging_measurement_unknown"), (SWAP, "unreachable")):
            s = comp_session(instruction)
            for _ in range(2):
                r = s.failure_route(reason, request="finish")
                self.assertEqual(r["next_hop"], "remeasure_no_motion")
                self.assertNotEqual(s._comp()["completion"], "failed")
            with self.assertRaises(ProgramFinished):
                s.failure_route(reason, request="finish")

    def test_measured_contact_failed_motion_changes_action_not_identity_checks(self):
        s = comp_session(ALIGN)
        r = s.failure_route("tool_contact_measurement_missing", motion_executed=True)
        self.assertEqual(r["next_hop"], "change_action_no_silent_retry")
        s = comp_session(SWAP)
        self.assertEqual(s.failure_route("memory_identity_lost")["next_hop"], "finish_honest_failure")

    def test_state_not_shared_between_episodes(self):
        a, b = comp_session(), comp_session()
        a.failure_route("failed")
        self.assertEqual(b._comp()["failures"], {})


class BoundaryTests(unittest.TestCase):
    def test_c_requires_fresh_complete_boundary_one_correction(self):
        s = comp_session("Fill the pen holder.")
        s.start("C", "pens")
        self.assertIn("policy_owner", s._ownership_guard("move_eef"))
        self.assertFalse(s.safe_boundary(evidence(s, empty_hands=True))["ok"])
        self.assertTrue(s.safe_boundary(boundary(s))["ok"])
        self.assertEqual(s._ownership_guard("move_eef"), "C_requires_one_tail_correct_call")
        s.eef_transfer = mock.Mock(return_value={"ok": True})
        self.assertTrue(s.tail_correct("left_arm", "o1", into="o2", evidence=boundary(s))["ok"])
        self.assertFalse(s.tail_correct("left_arm", "o1", into="o2", evidence=boundary(s))["ok"])
        self.assertFalse(s.safe_boundary(boundary(s))["ok"])
        self.assertEqual(s.eef_transfer.call_count, 1)

    def test_c_cannot_bypass_budget_through_other_complements(self):
        s = comp_session("Pack items into the box.")
        s.start("C", "packing")
        s.safe_boundary(boundary(s))
        action = mock.Mock(return_value={"ok": True})
        result = s._comp_motion("tool_contact", [1], action, evidence(s))
        self.assertEqual(result["reason"], "C_requires_one_tail_correct_call")
        action.assert_not_called()

    def test_c_failure_does_not_automatically_attempt_again(self):
        s = comp_session("Fill the pen holder.")
        s.start("C", "pens")
        s.safe_boundary(boundary(s))
        s.eef_transfer = mock.Mock(return_value={"ok": False, "reason": "not held"})
        s.tail_correct("left_arm", "o1", into="o2", evidence=boundary(s))
        self.assertEqual(s.failure_route("grasp_failed", motion_executed=True)["next_hop"], "finish_honest_failure")
        self.assertEqual(s.eef_transfer.call_count, 1)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.s = comp_session(BASKET)
        self.s.start("B", "sort")
        a, b = item("o1", -.1, -.2), item("o2", .1, -.2, name="basket")
        a["reach"], b["reach"] = {"left_arm": "down"}, {"left_arm": "down"}
        measured(self.s, [a, b])
        self.entries = [{"key": "car-1", "object": "o1", "identity": "unique car",
                         "destination": "left basket", "destination_object": "o2", "yaw_deg": None}]
        self.e = evidence(self.s, identity_unique=True)

    def bind(self):
        self.assertTrue(self.s.identity_ledger(self.entries, self.e)["ok"])

    def test_fresh_identity_and_alias_guards(self):
        self.s._eef_scene[0]["aliases"] = ["watch"]
        self.assertFalse(self.s.identity_ledger(self.entries, self.e)["ok"])
        self.e["aliases_resolved"] = True
        self.bind()
        self.s.state["step"] += 1
        self.assertEqual(self.s.identity_ledger(self.entries, evidence(self.s, identity_unique=True))["reason"], "fresh_locate_required")

    def test_destination_and_identity_cannot_silently_change(self):
        self.bind()
        bad = [dict(self.entries[0], identity="different object")]
        self.assertFalse(self.s.identity_ledger(bad, self.e)["ok"])
        bad = [dict(self.entries[0], destination="right basket")]
        self.assertFalse(self.s.identity_ledger(bad, self.e)["ok"])

    def test_failed_pick_never_release_and_same_retry_refused(self):
        self.bind()
        self.s.eef_pick = mock.Mock(return_value={"ok": False, "reason": "empty"})
        self.s.eef_release = mock.Mock(side_effect=AssertionError("must not release"))
        self.assertFalse(self.s.transfer_item("car-1", "left_arm", into="o2", evidence=self.e)["ok"])
        self.assertFalse(self.s.transfer_item("car-1", "left_arm", into="o2", evidence=self.e)["ok"])
        self.assertEqual(self.s.eef_pick.call_count, 1)
        self.s.eef_release.assert_not_called()

    def test_release_ok_is_not_done_needs_new_measurement(self):
        self.bind()
        self.s.eef_pick = mock.Mock(return_value={"ok": True})
        self.s.eef_release = mock.Mock(return_value={"ok": True})
        result = self.s.transfer_item("car-1", "left_arm", into="o2", evidence=self.e)
        self.assertTrue(result["ok"])
        self.assertEqual(self.s._comp()["ledger"]["car-1"]["status"], "placed_unverified")
        self.assertEqual(self.s.transfer_item("car-1", "left_arm", into="o2", evidence=self.e)["reason"],
                         "placement_unverified_no_replay")
        self.assertEqual(self.s.eef_pick.call_count, 1)
        check = evidence(self.s, identity_unique=True, destination_correct=True, holding=False, inside=True, below_rim=True)
        self.assertFalse(self.s.placement_check("car-1", check)["ok"])
        measured(self.s, self.s._eef_scene)
        self.s.identity_ledger(self.entries, self.e)
        self.assertFalse(self.s.placement_check("car-1", evidence(self.s, identity_unique=True,
                           destination_correct=True, holding=False, seated=True))["ok"])
        self.assertTrue(self.s.placement_check("car-1", check)["ok"])

    def test_directed_facing_not_unoriented_footprint(self):
        self.entries[0]["yaw_deg"] = 90
        self.assertFalse(self.s.identity_ledger(self.entries, self.e)["ok"])
        self.e["directed_axis"] = True
        self.bind()
        check = evidence(self.s, identity_unique=True, destination_correct=True, holding=False, seated=True)
        self.assertFalse(self.s.placement_check("car-1", check)["ok"])
        check["facing_correct"] = True
        self.assertTrue(self.s.placement_check("car-1", check)["ok"])

    def test_relabel_current_destination_id_keeps_semantics(self):
        self.bind()
        self.s._eef_scene[1]["id"] = "o9"
        self.entries[0]["destination_object"] = "o9"
        measured(self.s, self.s._eef_scene)
        self.assertTrue(self.s.identity_ledger(self.entries, self.e)["ok"])
        self.assertEqual(self.s._comp()["ledger"]["car-1"]["destination"], "left basket")

    def test_global_reach_intersection_and_explicit_staging(self):
        entries = [{"key": "car-1", "source": "o1", "destination": "o2", "category": "car"}]
        self.assertEqual(self.s.transfer_plan(entries, evidence=self.e)["plan"][0]["legs"][0]["arm"], "left_arm")
        self.s._eef_scene[1]["reach"] = {"right_arm": "down"}
        self.assertEqual(self.s.transfer_plan(entries, evidence=self.e)["reason"], "staging_required")
        self.s._eef_reach = mock.Mock(return_value={"left_arm": "down", "right_arm": "down"})
        plan = self.s.transfer_plan(entries, {"x": 0, "y": -.3}, evidence(self.s, staging_clear=True))
        self.assertEqual(plan["plan"][0]["legs"][1]["source"], "RELOCATE_SAME_IDENTITY")

    def test_same_arm_direct_ignores_optional_unreachable_staging(self):
        entries = [{"key": "car-1", "source": "o1", "destination": "o2"}]
        self.s._eef_reach = mock.Mock(side_effect=AssertionError("staging unnecessary"))
        for staging in (None, {"x": 0, "y": -.3}):
            r = self.s.transfer_plan(entries, staging, evidence(self.s, staging_clear=True))
            self.assertTrue(r["ok"])
            self.assertEqual(r["plan"][0]["legs"], [{"arm": "left_arm", "source": "o1", "into": "o2"}])
            self.assertEqual(r["blocked"], [])
        self.s._eef_reach.assert_not_called()

    def mixed_entries(self):
        cross = item("o3", .2, -.2, name="other basket")
        cross["reach"] = {"right_arm": "down"}
        source = item("o4", -.2, -.2, name="other car")
        source["reach"] = {"left_arm": "down"}
        measured(self.s, self.s._eef_scene + [cross, source])
        # Blocked entry FIRST used to discard the reachable entry that followed it.
        return [{"key": "cross-1", "source": "o4", "destination": "o3", "category": "a"},
                {"key": "car-1", "source": "o1", "destination": "o2", "category": "z"}]

    def test_missing_staging_retains_direct_work_and_cannot_finish_at_zero(self):
        s = self.s
        r = s.transfer_plan(self.mixed_entries(), evidence=self.e)
        self.assertFalse(r["ok"])
        self.assertTrue(r["recoverable"])
        self.assertEqual(r["reason"], "staging_required")
        self.assertEqual([row["key"] for row in r["plan"]], ["car-1"])
        self.assertEqual(r["blocked"][0]["key"], "cross-1")
        self.assertEqual(r["next_hop"], "continue_direct_transfers")
        r = s.failure_route("capability_missing", request="finish")
        self.assertEqual(r["next_hop"], "continue_direct_transfers")
        self.assertNotEqual(s._comp()["completion"], "failed")
        self.assertEqual(s.state["step"], 0)
        self.bind()
        s.eef_pick, s.eef_release = mock.Mock(return_value={"ok": True}), mock.Mock(return_value={"ok": True})
        self.assertTrue(s.transfer_item("car-1", "left_arm", into="o2", evidence=self.e)["ok"])

    def test_unreachable_staging_retains_directs_and_new_spot_recovers(self):
        s = self.s
        entries = self.mixed_entries()
        s._eef_reach = mock.Mock(side_effect=[{"left_arm": "down"},
                                             {"left_arm": "down", "right_arm": "down"}])
        e = evidence(s, staging_clear=True)
        r = s.transfer_plan(entries, {"x": 0, "y": -.3}, e)
        self.assertEqual(r["reason"], "unreachable_staging")
        self.assertEqual(r["next_hop"], "continue_direct_transfers")
        self.assertEqual(s.failure_route(r["reason"])["next_hop"], "continue_direct_transfers")
        r = s.transfer_plan(entries, {"x": 0, "y": -.35}, e)
        self.assertTrue(r["ok"])
        self.assertEqual([len(row["legs"]) for row in r["plan"]], [1, 2])
        self.assertEqual(s._comp()["completion"], "unknown")

    def test_staging_only_failure_is_recoverable_not_capability_missing(self):
        s = self.s
        entries = self.mixed_entries()[:1]
        s._eef_reach = mock.Mock(return_value={"left_arm": "down"})
        r = s.transfer_plan(entries, {"x": 0, "y": -.3}, evidence(s, staging_clear=True))
        self.assertEqual(r["reason"], "unreachable_staging")
        self.assertEqual(r["next_hop"], "remeasure_no_motion")
        self.assertTrue(r["recoverable"])
        self.assertEqual(s.failure_route(r["reason"], request="finish")["next_hop"], "remeasure_no_motion")
        self.assertEqual(s.failure_route(r["reason"], request="return_wall", evidence=boundary(s))["next_hop"],
                         "wall_fallback_denied")

    def test_staging_budget_is_shared_across_reason_names(self):
        s = self.s
        entries = self.mixed_entries()[:1]
        r = s.transfer_plan(entries, evidence=self.e)
        self.assertEqual(s.failure_route("capability_missing", request="finish")["next_hop"], "remeasure_no_motion")
        self.assertEqual(s.failure_route(r["reason"])["next_hop"], "remeasure_no_motion")
        result = s.failure_route("unreachable_staging")
        self.assertEqual(result["next_hop"], "finish_honest_failure")
        self.assertEqual(result["recovery_budget_count"], 3)

    def test_exhausted_staging_budget_does_not_erase_unmoved_directs(self):
        s = self.s
        s.transfer_plan(self.mixed_entries(), evidence=self.e)
        for reason in ("staging_required", "unreachable_staging", "capability_missing"):
            result = s.failure_route(reason, request="finish")
            self.assertEqual(result["next_hop"], "continue_direct_transfers")
        self.assertTrue(result["recovery_budget_exhausted"])
        self.assertNotEqual(s._comp()["completion"], "failed")

    def test_staging_checks_do_not_finish_item_and_allow_second_arm_leg(self):
        s = self.s
        self.bind()
        s.eef_pick = mock.Mock(return_value={"ok": True})
        s.eef_release = mock.Mock(return_value={"ok": True})
        self.assertTrue(s.transfer_item("car-1", "left_arm", x=0, y=-.3, staging=True,
                                       evidence=evidence(s, staging_clear=True))["ok"])
        measured(s, s._eef_scene)
        s.identity_ledger(self.entries, self.e)
        self.assertTrue(s.placement_check("car-1", evidence(s, identity_unique=True, staging_matches=True,
                                                         holding=False, seated=True))["ok"])
        self.assertEqual(s._comp()["ledger"]["car-1"]["status"], "staged")
        self.assertTrue(s.transfer_item("car-1", "right_arm", into="o2", evidence=self.e)["ok"])
        self.assertEqual(s.eef_pick.call_count, 2)


class CountTests(unittest.TestCase):
    def setUp(self):
        self.s = comp_session(PRESS)
        self.s.start("B", "count")
        self.s.press = mock.Mock(side_effect=lambda target, times, arm: {"ok": True, "presses": times})

    def test_two_cards_interleaved_confirm_once_then_read_only_commit(self):
        s = self.s
        self.assertTrue(s.press_card(0, 3, "red-left", evidence=evidence(s, read_confident=True, target_unique=True))["ok"])
        self.assertFalse(s.press_card(1, 4, "red-middle", evidence=evidence(s, read_confident=True, target_unique=True))["ok"])
        self.assertTrue(s.confirm_card(0, "blue", evidence=evidence(s, count_matches=True, target_unique=True))["ok"])
        self.assertTrue(s.confirm_card(0, "blue", evidence=evidence(s, confirmed=True))["ok"])
        self.assertEqual(s.press.call_count, 2)
        self.assertTrue(s.press_card(1, 4, "red-middle", evidence=evidence(s, read_confident=True, target_unique=True))["ok"])
        self.assertEqual(s.press.call_count, 3)

    def test_partial_or_timedout_count_never_repressed(self):
        s = self.s
        s.press.side_effect = TimeoutError("already sent")
        e = evidence(s, read_confident=True, target_unique=True)
        result = s.press_card(0, 3, "red", evidence=e)
        self.assertEqual(result["next_hop"], "finish_honest_failure")
        self.assertFalse(s.press_card(0, 3, "red", evidence=e)["ok"])
        self.assertFalse(s.confirm_card(0, "blue", evidence=e)["ok"])
        self.assertEqual(s.press.call_count, 1)

    def test_no_confirm_without_count_evidence(self):
        s = self.s
        s.press_card(0, 3, "red", evidence=evidence(s, read_confident=True, target_unique=True))
        self.assertFalse(s.confirm_card(0, "blue", evidence=evidence(s, target_unique=True))["ok"])
        self.assertEqual(s.press.call_count, 1)

    def test_zero_count_and_invalid_counts_no_motion(self):
        s = self.s
        for count in (True, -1, 30, 1.5):
            self.assertFalse(s.press_card(0, count, "red", evidence=evidence(s, read_confident=True, target_unique=True))["ok"])
        self.assertTrue(s.press_card(0, 0, "red", evidence=evidence(s, read_confident=True, target_unique=True))["ok"])
        s.press.assert_not_called()

    def test_real_press_cycles_lift_without_home(self):
        s = pressing_session(FakeWorker(floor=.80+.15-.004))
        s._counted_press = True
        s.home = mock.Mock(side_effect=AssertionError("counted presses never home"))
        result = s.press([-.1, -.1, .80], times=3)
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(result["cycles"]), 3)
        self.assertTrue(all(cycle["released"] for cycle in result["cycles"]))
        s.home.assert_not_called()

    def test_raw_press_cannot_bypass_counted_route(self):
        s = comp_session(PRESS)
        s.start("B", "press")
        self.assertEqual(s.press("o1", times=3)["reason"], "counted_press_api_required")
        self.assertEqual(s.state["step"], 0)

    def test_completed_count_with_failed_clearance_is_not_count_uncertain(self):
        s = self.s
        raw = {"ok": True, "presses": 3, "count_complete": True, "arm_clear": False,
               "contacts": [.8], "clearance_receipt": {"event": {"kind": "TIMEOUT"}}}
        s.press.side_effect = lambda *_args, **_kwargs: raw
        result = s.press_card(0, 3, "red", evidence=evidence(s, read_confident=True, target_unique=True))
        self.assertTrue(result["ok"])
        self.assertEqual(s._comp()["cards"][0]["status"], "pressed_unconfirmed")
        self.assertTrue(s._comp()["uncertain"])
        self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
        self.assertFalse(s.press_card(0, 3, "red", evidence=evidence(s))["ok"])
        self.assertEqual(s.press.call_count, 1)
        self.assertNotIn("count_uncertain", s._comp()["failures"])
        blocked = s.confirm_card(0, "blue", evidence=evidence(s, count_matches=True, target_unique=True))
        self.assertEqual(blocked["next_hop"], "inspect_receipt_no_retry")
        self.assertEqual(s._comp()["cards"][0]["status"], "pressed_unconfirmed")
        self.assertEqual(s.press.call_count, 1)

    def test_confirm_card_rejects_partial_or_false_complete_receipt(self):
        for raw in ({"ok": True, "presses": 0}, {"ok": True, "presses": 1, "count_complete": False}):
            with self.subTest(raw=raw):
                self.setUp()
                s = self.s
                s.press_card(0, 3, "red", evidence=evidence(s, read_confident=True, target_unique=True))
                s.press.side_effect = lambda *_args, **_kwargs: raw
                result = s.confirm_card(0, "blue", evidence=evidence(s, count_matches=True, target_unique=True))
                self.assertEqual(result["reason"], "confirm_uncertain")
                self.assertFalse(s.confirm_card(0, "blue", evidence=evidence(s))["ok"])
                self.assertEqual(s.press.call_count, 2)


class PlanarToolTests(unittest.TestCase):
    def setUp(self):
        self.s = comp_session()
        self.s.start("B", "planar")
        obj = item("o1", 0, -.2, TABLE_Z+.02, name="T block")
        obj["reach"] = {"left_arm": "down"}
        measured(self.s, [obj])
        self.s.close_gripper = mock.Mock(side_effect=lambda *_: trace(self.s))
        self.s.move_eef = mock.Mock(side_effect=lambda *_args, **_kwargs: trace(self.s))

    def e(self):
        return evidence(self.s, directed_axis=True, yaw_deg=0, contact=[-.03, -.2, TABLE_Z+.015], approach_clear=True)

    def test_planar_bounded_translation_does_not_fabricate_pose(self):
        position = list(self.s._eef_scene[0]["position"])
        r = self.s.planar_push("left_arm", "o1", {"x": .1, "y": -.2, "yaw_deg": 0}, evidence=self.e())
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "translation")
        self.assertEqual(self.s._eef_scene[0]["position"], position)
        self.assertEqual(self.s.move_eef.call_args_list[2].kwargs["max_seconds"], 2.0)

    def test_tangent_rotation_with_visible_lever(self):
        r = self.s.planar_push("left_arm", "o1", {"x": 0, "y": -.2, "yaw_deg": 30}, evidence=self.e())
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "tangential_rotation")

    def test_contact_height_and_directed_axis_guard_no_motion(self):
        e = self.e()
        e["directed_axis"] = False
        self.assertFalse(self.s.planar_push("left_arm", "o1", {"x": .1, "y": -.2, "yaw_deg": 0}, evidence=e)["ok"])
        self.s.move_eef.assert_not_called()

    def test_planar_no_progress_bounded(self):
        for i in range(3):
            r = self.s.planar_push("left_arm", "o1", {"x": .1, "y": -.2, "yaw_deg": 0}, evidence=self.e())
        self.assertEqual(r["reason"], "planar_no_progress")

    def test_align_cannot_skip_required_tool(self):
        s = comp_session(ALIGN)
        s.start("B", "align")
        measured(s, self.s._eef_scene)
        r = s.planar_push("left_arm", "o1", {"x": .1, "y": -.2, "yaw_deg": 0}, evidence=evidence(s))
        self.assertEqual(r["reason"], "align_requires_tool_contact")

    def test_no_pick_or_unmeasured_push_bypasses_planar_T(self):
        s = self.s
        self.assertEqual(s.eef_pick("left_arm", "o1")["reason"], "planar_T_no_lift")
        self.assertEqual(s.eef_push("left_arm", "o1", .1, -.2)["reason"], "measured_planar_or_tool_contact_required")
        s.move_eef.assert_not_called()

    def test_tool_offset_and_holding_required_stroke_bounded(self):
        s = self.s
        e = evidence(s, tool_held=True, contact_visible=True, offset_measured=True, path_clear=True)
        args = ("left_arm", "o1", [0, -.2, .84])
        kwargs = {"tip_offset": [0, 0, -.03], "direction": [1, 0, 0], "distance": .02, "evidence": e}
        self.assertEqual(s.tool_contact(*args, **kwargs)["reason"], "tool_contact_lost")
        s._eef_grip["left_arm"] = (0, 0, .03)
        s._eef_scene[0]["held_by"] = "left_arm"
        self.assertTrue(s.tool_contact(*args, **kwargs)["ok"])
        self.assertEqual(s.move_eef.call_count, 2)
        self.assertFalse(s.tool_contact(*args, **{**kwargs, "distance": .1, "evidence": evidence(s)})["ok"])

    def test_orientation_keeps_current_tilt_and_requires_held_identity(self):
        s = self.s
        s.identity_ledger([{"key": "T", "identity": "T block", "object": "o1", "destination": None, "yaw_deg": 0}],
                          evidence(s, identity_unique=True, directed_axis=True))
        s._eef_grip["left_arm"] = (0, 0, .04)
        r = s.orient_held("T", "left_arm", 20, evidence(s, orientation_measured=True, path_clear=True, yaw_deg=0))
        self.assertTrue(r["ok"], r)


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.entries = [{"key": key, "object": obj, "identity": key,
                         "destination": "swapped pad", "yaw_deg": 0}
                        for key, obj in (("first", "o1"), ("second", "o2"))]
        self.s = self.make_session()

    def make_session(self, *, bind=True):
        s = comp_session(SWAP)
        s.start("B", "swap")
        measured(s, [item("o1", -.12, -.2), item("o2", .12, -.2),
                     item("o4", 0, -.35, name="empty mat")])
        if bind:
            self.assertTrue(s.identity_ledger(self.entries,
                            evidence(s, identity_unique=True, directed_axis=True))["ok"])
        s._eef_reach = mock.Mock(return_value={"left_arm": "down", "right_arm": "down"})
        return s

    def swap_evidence(self, s, **extra):
        return evidence(s, **{"identity_unique": True, "directed_axis": True,
                             "staging_clear": True, "mat_observed": True, **extra})

    def test_entry_dicts_bind_without_prior_ledger_and_plan_correct_three_legs(self):
        s = self.make_session(bind=False)
        first = {"key": "original-left", "object": "o1", "identity": "unique left block", "yaw_deg": -55.6}
        second = {"key": "original-right", "object": "o2", "identity": "unique right block", "yaw_deg": 67.5}
        staging = {"object": "o4", "position": [-.0008, -.2066, .785], "empty": True}
        self.assertEqual(s._comp()["ledger"], {})
        plan = s.swap_plan(first, second, staging, self.swap_evidence(s))
        self.assertTrue(plan["ok"], plan)
        self.assertIsNone(plan["reason"])
        self.assertEqual(plan["legs"], [
            {"key": "original-left", "xy": [-.0008, -.2066], "yaw_deg": -55.6},
            {"key": "original-right", "xy": [-.12, -.2], "yaw_deg": 67.5},
            {"key": "original-left", "xy": [.12, -.2], "yaw_deg": -55.6}])
        self.assertEqual(s._comp()["ledger"]["original-left"]["object"], "o1")
        self.assertEqual(s._comp()["ledger"]["original-right"]["identity"], "unique right block")
        self.assertEqual(s.state["step"], 0)
        self.assertFalse(plan["motion_executed"])
        self.assertEqual(plan["completion"], "plan_only")
        self.assertEqual(plan["event"]["source"], "swap_plan")

    def test_staging_position_xy_object_and_tuple_forms(self):
        for staging in ([0, -.35], (0, -.35), {"position": [0, -.35, .785]},
                        {"xy": [0, -.35]}, {"xy": (0, -.35, .785)}, "o4", {"object": "o4"}):
            with self.subTest(staging=staging):
                s = self.make_session()
                plan = s.swap_plan("first", "second", staging, self.swap_evidence(s))
                self.assertTrue(plan["ok"], plan)
                self.assertEqual(plan["legs"][0]["xy"], [0, -.35])

    def test_entry_dicts_also_accept_object_only_staging(self):
        s = self.make_session(bind=False)
        plan = s.swap_plan(*self.entries, {"object": "o4"}, evidence=self.swap_evidence(s))
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(plan["legs"][0]["xy"], [0, -.35])

    def test_mixed_string_key_and_entry_dict(self):
        s = self.s
        plan = s.swap_plan("first", self.entries[1], {"position": [0, -.35, .785]}, self.swap_evidence(s))
        self.assertTrue(plan["ok"], plan)
        self.assertEqual([leg["key"] for leg in plan["legs"]], ["first", "second", "first"])

    def test_entry_dict_identity_not_unique_remains_honest_failure(self):
        for unique in (False, None, 1):
            with self.subTest(identity_unique=unique):
                s = self.make_session(bind=False)
                plan = s.swap_plan(*self.entries, {"position": [0, -.35, .785]},
                                   evidence=self.swap_evidence(s, identity_unique=unique))
                self.assertEqual(plan["reason"], "memory_identity_lost")
                self.assertEqual(plan["next_hop"], "finish_honest_failure")
                self.assertEqual(s._comp()["ledger"], {})
                self.assertEqual(s.failure_route(plan["reason"])["next_hop"], "finish_honest_failure")

    def test_missing_ledger_or_malformed_entry_remains_fail_closed(self):
        for first in ("missing-key", {"object": "o1"}, {"key": "first", "object": "o1"},
                      {"key": "first", "object": "o1", "identity": ""}, None, ["first"]):
            with self.subTest(first=first):
                s = self.make_session(bind=False)
                plan = s.swap_plan(first, self.entries[1], [0, -.35], self.swap_evidence(s))
                self.assertEqual(plan["reason"], "memory_identity_lost")
                self.assertEqual(plan["next_hop"], "finish_honest_failure")
                self.assertEqual(s._comp()["ledger"], {})

    def test_entry_binding_requires_resolved_aliases(self):
        s = self.make_session(bind=False)
        s._eef_scene[0]["aliases"] = ["other block"]
        plan = s.swap_plan(*self.entries, [0, -.35], evidence=self.swap_evidence(s))
        self.assertEqual(plan["reason"], "identity_ambiguous")
        self.assertEqual(s._comp()["ledger"], {})
        self.assertNotIn("swap", s._comp())
        plan = s.swap_plan(*self.entries, [0, -.35], evidence=self.swap_evidence(s, aliases_resolved=True))
        self.assertTrue(plan["ok"], plan)

    def test_duplicate_entry_keys_or_object_bindings_are_ambiguous(self):
        for second in (dict(self.entries[1], key="first"), dict(self.entries[1], object="o1")):
            with self.subTest(second=second):
                s = self.make_session(bind=False)
                plan = s.swap_plan(self.entries[0], second, [0, -.35], self.swap_evidence(s))
                self.assertEqual(plan["reason"], "identity_ambiguous")
                self.assertEqual(s._comp()["ledger"], {})
                self.assertNotIn("swap", s._comp())

    def test_unknown_object_entry_does_not_raise_or_guess_nearest(self):
        s = self.make_session(bind=False)
        second = dict(self.entries[1], object="o99")
        plan = s.swap_plan(self.entries[0], second, [0, -.35], self.swap_evidence(s))
        self.assertEqual(plan["reason"], "identity_ambiguous")
        self.assertEqual(plan["receipt"]["exception_type"], "ValueError")
        self.assertIn("o99", plan["receipt"]["exception"])
        self.assertEqual(s._comp()["ledger"], {})

    def test_entry_binding_requires_directed_axis_and_yaw(self):
        for first, extra in ((self.entries[0], {"directed_axis": False}),
                             (dict(self.entries[0], yaw_deg=None), {}),
                             ({k: v for k, v in self.entries[0].items() if k != "yaw_deg"}, {}),
                             (dict(self.entries[0], yaw_deg=float("nan")), {})):
            with self.subTest(first=first, extra=extra):
                s = self.make_session(bind=False)
                plan = s.swap_plan(first, self.entries[1], [0, -.35], self.swap_evidence(s, **extra))
                self.assertEqual(plan["reason"], "directed_orientation_unknown")
                self.assertNotIn("swap", s._comp())

    def test_entry_binding_cannot_change_identity_or_destination(self):
        for first in (dict(self.entries[0], identity="other identity"),
                      dict(self.entries[0], destination="other destination")):
            with self.subTest(first=first):
                s = self.make_session()
                before = s.identity_ledger()["ledger"]
                plan = s.swap_plan(first, self.entries[1], [0, -.35], self.swap_evidence(s))
                self.assertEqual(plan["reason"], "identity_or_destination_changed")
                self.assertEqual(s.identity_ledger()["ledger"], before)

    def test_entry_binding_requires_fresh_evidence_and_locate(self):
        for stale_evidence in (False, True):
            with self.subTest(stale_evidence=stale_evidence):
                s = self.make_session(bind=False)
                s.state["step"] += 1
                e = self.swap_evidence(s)
                if stale_evidence:
                    e["observed_step"] = 0
                plan = s.swap_plan(*self.entries, [0, -.35], evidence=e)
                self.assertEqual(plan["reason"], "stale_or_missing_measurement" if stale_evidence
                                 else "fresh_locate_required")
                self.assertEqual(s._comp()["ledger"], {})

    def test_staging_invalid_geometry_or_unknown_object_fails_without_plan(self):
        for staging in ({"position": [0]}, {"position": [float("nan"), -.35, .785]},
                        {"xy": "0,-.35"}, {"object": "o99"}, {}, [0, -.35, .785],
                        {"position": None, "object": "o4"}):
            with self.subTest(staging=staging):
                s = self.make_session()
                plan = s.swap_plan("first", "second", staging, self.swap_evidence(s))
                self.assertEqual(plan["reason"], "staging_measurement_unknown")
                self.assertNotIn("swap", s._comp())
                if staging == {"object": "o99"}:
                    self.assertEqual(plan["receipt"]["exception_type"], "ValueError")
                    self.assertIn("o99", plan["receipt"]["exception"])

    def test_object_staging_requires_current_locate_and_resolved_aliases(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                s = self.make_session()
                if stale:
                    s.state["step"] += 1
                else:
                    s._eef_scene[2]["aliases"] = ["occupied pad"]
                plan = s.swap_plan("first", "second", {"object": "o4"}, self.swap_evidence(s))
                self.assertEqual(plan["reason"], "staging_measurement_unknown")
                self.assertNotIn("swap", s._comp())

    def test_staging_clearance_and_reach_checks_still_apply_to_dicts(self):
        for extra, staging, reach, reason in (
                ({"mat_observed": False}, {"object": "o4"}, "down", "staging_measurement_unknown"),
                ({"staging_clear": False}, {"position": [0, -.35, .785]}, "down", "staging_measurement_unknown"),
                ({}, {"position": [-.12, -.2, .785]}, "down", "staging_overlaps_protected_object"),
                ({}, {"object": "o4"}, None, "unreachable")):
            with self.subTest(extra=extra, staging=staging, reach=reach):
                s = self.make_session(bind=False)
                s._eef_reach.return_value = {"left_arm": reach, "right_arm": reach}
                plan = s.swap_plan(*self.entries, staging, evidence=self.swap_evidence(s, **extra))
                self.assertEqual(plan["reason"], reason)
                self.assertNotIn("swap", s._comp())
                self.assertEqual(s.state["step"], 0)

    def test_actual_empty_mat_three_frozen_legs(self):
        s = self.s
        e = evidence(s, identity_unique=True, staging_clear=True, mat_observed=True)
        plan = s.swap_plan("first", "second", [0, -.35], e)
        self.assertTrue(plan["ok"])
        self.assertEqual(len(plan["legs"]), 3)
        self.assertEqual(plan["legs"][1]["xy"], [-.12, -.2])
        s._comp()["ledger"]["first"]["position"] = [9,9,9]
        frozen = s.swap_plan("first", "second", [0, -.4], e)
        self.assertEqual(frozen["swap"]["legs"], plan["legs"])

    def test_missing_mat_or_identity_does_not_plan(self):
        s = self.s
        plan = s.swap_plan("first", "second", [0, -.35], evidence(s, identity_unique=True, staging_clear=True))
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["reason"], "staging_measurement_unknown")
        self.assertEqual(plan["next_hop"], "remeasure_no_motion")

    def test_identity_loss_remains_fail_closed_separate_from_staging(self):
        s = self.s
        plan = s.swap_plan("first", "second", [0, -.35], evidence(s, identity_unique=False,
                           staging_clear=True, mat_observed=True))
        self.assertEqual(plan["reason"], "memory_identity_lost")
        self.assertEqual(plan["next_hop"], "finish_honest_failure")

    def test_confirmation_after_each_leg_once_only(self):
        s = self.s
        s.swap_plan("first", "second", [0, -.35], evidence(s, identity_unique=True, staging_clear=True, mat_observed=True))
        s.press = mock.Mock(return_value={"ok": True, "presses": 1})
        e = evidence(s, identity_unique=True, at_destination=True, facing_correct=True, released=True, target_unique=True)
        self.assertFalse(s.swap_confirm(1, "blue", evidence=e)["ok"])
        self.assertTrue(s.swap_confirm(0, "blue", evidence=e)["ok"])
        self.assertFalse(s.swap_confirm(0, "blue", evidence=e)["ok"])
        self.assertEqual(s.press.call_count, 1)

    def plan_and_press(self, raw):
        s = self.s
        s.swap_plan("first", "second", [0, -.35], self.swap_evidence(s))
        s.press = mock.Mock(return_value=raw)
        e = evidence(s, identity_unique=True, at_destination=True, facing_correct=True,
                     released=True, target_unique=True)
        return s, e

    def test_downstroke_timeout_confirmed_leg_continues_and_rejects_false_failure(self):
        raw = {"ok": True, "presses": 1, "count_complete": True, "arm_clear": True,
               "contacts": [.8], "cycles": [{"down": {"event": {"kind": "TIMEOUT"}}, "released": True}]}
        s, e = self.plan_and_press(raw)
        result = s.swap_confirm(0, "blue", evidence=e)
        self.assertTrue(result["ok"])
        self.assertEqual(result["confirmation"], "counted_press_complete")
        self.assertEqual(result["confirmed_legs"], 1)
        self.assertEqual(result["next_hop"], "rebind_next_swap_leg")
        self.assertFalse(s._comp()["uncertain"])
        self.assertEqual(result["receipt"], raw)  # original TIMEOUT retained
        self.assertEqual(json.loads(turn_report(s)["text"].split(": ", 1)[1])["swap"]["confirmed_legs"], 1)
        for receipt in (raw, result):
            reroute = s.failure_route("confirmation_downstroke_timeout", receipt=receipt, request="finish")
            self.assertEqual(reroute["reason"], "counted_press_already_complete")
            self.assertEqual(reroute["next_hop"], "continue_after_counted_press")
        self.assertEqual(s._comp()["completion"], "unknown")
        self.assertFalse(s.swap_confirm(0, "blue", evidence=e)["ok"])
        self.assertFalse(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])
        self.assertTrue(s.swap_confirm(1, "blue", evidence=e)["ok"])
        self.assertTrue(s.swap_confirm(2, "blue", evidence=e)["ok"])
        self.assertEqual(s._comp()["swap"]["confirmed_legs"], 3)
        self.assertEqual(s.press.call_count, 3)
        self.assertTrue(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])

    def test_clearance_failure_keeps_confirm_but_blocks_next_motion(self):
        raw = {"ok": True, "presses": 1, "count_complete": True, "arm_clear": False,
               "clearance_receipt": {"event": {"kind": "TIMEOUT"}}}
        s, e = self.plan_and_press(raw)
        result = s.swap_confirm(0, "blue", evidence=e)
        self.assertTrue(result["ok"])
        self.assertEqual(result["confirmed_legs"], 1)
        self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
        self.assertEqual(s._ownership_guard("move_eef"), "motion_uncertain_no_retry")
        route = s.failure_route("confirm_uncertain", receipt=result, request="finish")
        self.assertEqual(route["next_hop"], "inspect_receipt_no_retry")
        self.assertNotEqual(s._comp()["completion"], "failed")
        self.assertFalse(s.swap_confirm(0, "blue", evidence=e)["ok"])
        blocked = s.swap_confirm(1, "blue", evidence=e)
        self.assertEqual(blocked["next_hop"], "inspect_receipt_no_retry")
        self.assertEqual(s._comp()["swap"]["attempted"], [0])
        self.assertEqual(s.press.call_count, 1)
        s.failure_route("final_lift_unverified", receipt=raw,
                        evidence=evidence(s, resolved_motion=True))
        self.assertFalse(s._comp()["uncertain"])
        self.assertEqual(s._comp()["swap"]["confirmed_legs"], 1)
        s.press.return_value = {"ok": True, "presses": 1, "count_complete": True, "arm_clear": True}
        self.assertTrue(s.swap_confirm(1, "blue", evidence=e)["ok"])
        self.assertEqual(s.press.call_count, 2)

    def confirm_all_legs(self, *, final_arm_clear=True):
        raw = {"ok": True, "presses": 1, "count_complete": True, "arm_clear": True,
               "cycles": [{"down": {"event": {"kind": "TIMEOUT"}}, "released": True}]}
        s, e = self.plan_and_press(raw)
        for index in range(3):
            if index == 2:
                s.press.return_value = {**raw, "arm_clear": final_arm_clear}
            result = s.swap_confirm(index, "blue", evidence=e)
            self.assertTrue(result["ok"], result)
        return s, result

    def test_third_confirm_requires_complete_task_without_more_motion(self):
        s, result = self.confirm_all_legs()
        self.assertEqual(result["confirmed_legs"], 3)
        self.assertTrue(result["count_complete"])
        self.assertTrue(result["completion_required"])
        self.assertEqual(result["reason"], "swap_complete_task_required")
        self.assertEqual(result["next_hop"], "complete_task")
        for rule in ("MUST call complete_task next", "at most ONE", "No transfer", "facing correction",
                     "three swap_confirm receipts + ledger done", "do NOT use verify() to veto closeout",
                     "evidence.complete=False or missing", "all safety checks still apply"):
            self.assertIn(rule, result["message"])
        self.assertEqual(result["receipt"], s.press.return_value)  # retain original nested TIMEOUT
        self.assertEqual(result["event"]["detail"]["next_hop"], "complete_task")
        report = json.loads(turn_report(s)["text"].split(": ", 1)[1])
        self.assertEqual(report["last_event"]["next_hop"], "complete_task")
        self.assertEqual(report["completion"], "unknown")  # not auto-success
        completion = s.complete_task(evidence(s, complete=True, safe_to_finish=True))
        self.assertTrue(completion["ok"], completion)
        self.assertEqual(completion["reason"], "swap_completion_observed")
        self.assertEqual(completion["confirmed_legs"], 3)
        self.assertEqual(completion["next_hop"], "finish")
        self.assertFalse(completion["completion_required"])
        with self.assertRaises(ProgramFinished) as finished:
            s.finish("observed swap complete")
        self.assertNotIn("FAILED", str(finished.exception))
        s._wrapping_up = True
        self.assertIsNone(s._ownership_guard("home"))
        self.assertEqual(s.press.call_count, 3)

    def test_confirmed_swap_accepts_false_or_missing_complete_and_preserves_receipts(self):
        for fields in ({}, {"complete": False}, {"complete": None}, {"complete": True}):
            with self.subTest(fields=fields):
                self.setUp()
                s, _ = self.confirm_all_legs()
                before = json.loads(json.dumps(s._comp()["swap"]))
                ledger = json.loads(json.dumps(s._comp()["ledger"]))
                self.assertEqual(len(before["receipts"]), 3)
                self.assertTrue(all(item["status"] == "done" for item in ledger.values()))
                e = evidence(s, safe_to_finish=True, **fields)
                step = s.state["step"]
                result = s.complete_task(e)
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["reason"], "swap_completion_observed")
                self.assertTrue(all(result["checks"].values()))
                self.assertEqual(result["next_hop"], "finish")
                self.assertEqual(result["evidence"], e)  # never overwrite False / fabricate complete
                self.assertEqual(result["event"]["detail"]["evidence"], e)
                self.assertEqual(s.state["step"], step)
                self.assertEqual(s._comp()["swap"], before)
                self.assertEqual(s._comp()["ledger"], ledger)
                self.assertEqual(s._comp()["completion"], "observed")
                with self.assertRaises(ProgramFinished) as finished:
                    s.finish("three confirmations and ledger done; safety observed")
                self.assertNotIn("FAILED", str(finished.exception))
                self.assertEqual(s.press.call_count, 3)

    def test_optional_vlm_verify_false_does_not_veto_confirmed_swap_closeout(self):
        # r8b episode-016 regression: original NO and complete=False stay diagnostic.
        s, _ = self.confirm_all_legs()
        s.verifier = mock.Mock()
        s.verifier.ask.return_value = "NO\nBlocks are occluded by the close-up grippers."
        s.get_observation = mock.Mock(return_value=mock.Mock(images={}))
        final_ok, answer = s.verify("The two blocks have been swapped.")
        self.assertFalse(final_ok)
        e = evidence(s, complete=final_ok, safe_to_finish=True, verifier_answer=answer)
        result = s.complete_task(e)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["checks"]["complete"])
        self.assertIs(result["evidence"]["complete"], False)
        self.assertEqual(result["evidence"]["verifier_answer"], answer)
        with self.assertRaises(ProgramFinished) as finished:
            s.finish("confirmed swap; safety observed")
        self.assertNotIn("FAILED", str(finished.exception))
        self.assertEqual(s.press.call_count, 3)
        s.verifier.ask.assert_called_once()

    def test_swap_with_fewer_than_three_confirmations_cannot_complete(self):
        s, e = self.plan_and_press({"ok": True, "presses": 1, "count_complete": True, "arm_clear": True})
        for count in range(3):
            self.assertEqual(s._comp()["swap"]["confirmed_legs"], count)
            for fields in ({}, {"complete": False}, {"complete": True}):
                with self.subTest(count=count, fields=fields):
                    result = s.complete_task(evidence(s, safe_to_finish=True, **fields))
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["reason"], "completion_unverified")
                    self.assertEqual(s._comp()["completion"], "unknown")
            self.assertTrue(s.swap_confirm(count, "blue", evidence=e)["ok"])
        self.assertTrue(s.complete_task(evidence(s, safe_to_finish=True))["ok"])
        self.assertEqual(s.press.call_count, 3)

    def test_swap_prompt_requires_same_cell_closeout_and_home_reserve(self):
        text = (ROOT / "prompts/robodojo_tasks/swap_blocks.md").read_text()
        for rule in ("SAME cell", "complete_task -> finish", "do NOT open another observation cell",
                     "reserve at least 50 host steps", "finish by step 650", "690+",
                     "completion observed is NOT environment success", "not a guarantee"):
            self.assertIn(rule, text)

    def test_third_confirm_closeout_budget_and_same_cell_guidance(self):
        for step, limit, remaining, urgent in ((649, 700, 51, False), (650, 700, 50, True),
                                              (691, 700, 9, True), (701, 700, 0, True),
                                              (40, 80, 40, True), (649, None, None, False),
                                              (691, None, None, True)):
            with self.subTest(step=step, limit=limit):
                self.setUp()
                s, e = self.plan_and_press({"ok": True, "presses": 1, "count_complete": True,
                                           "arm_clear": True})
                s.state.update(step=step)
                if limit is None:
                    s.state.pop("step_limit", None)
                else:
                    s.state["step_limit"] = limit
                e["observed_step"] = step
                for index in range(3):
                    result = s.swap_confirm(index, "blue", evidence=e)
                    self.assertTrue(result["ok"], result)
                self.assertTrue(result["same_cell_closeout"])
                self.assertEqual(result["wrap_up_reserve_steps"], 50)
                self.assertEqual(result["remaining_steps"], remaining)
                self.assertEqual(result["closeout_urgent"], urgent)
                for rule in ("SAME cell", "complete_task -> finish", "do NOT open another observation cell",
                             "finish by step 650", "690+", "never fabricate evidence"):
                    self.assertIn(rule, result["message"])
                self.assertEqual("URGENT:" in result["message"], urgent)
                self.assertEqual(result["event"]["detail"]["remaining_steps"], remaining)
                self.assertEqual(s.state["step"], step)  # guidance does not observe or advance time
                self.assertEqual(s._comp()["completion"], "unknown")
                self.assertEqual(s.press.call_count, 3)
                completion = s.complete_task(evidence(s, complete=False, safe_to_finish=True))
                self.assertTrue(completion["ok"], completion)
                self.assertTrue(completion["same_cell_closeout"])
                self.assertEqual(completion["remaining_steps"], remaining)
                self.assertIn("NO new observation cell", completion["message"])
                self.assertIn("NOT environment success", completion["message"])

    def test_completion_observed_then_finish_really_homes_both_arms_with_low_budget(self):
        # Synthetic short return, NOT replay/eval or a claim that ep040's real path fits.
        # At step 691/700 the 3.0 rad/s return takes 8 ticks; default 2.5 takes 9
        # and hits truncation. Run actual _wrap_up/home/_execute, not a mocked home.
        for fast in (True, False):
            with self.subTest(fast=fast):
                self.setUp()
                s, _ = self.confirm_all_legs()
                worker = s.env
                worker.step, worker.step_limit = 691, 700
                worker.joints[0], worker.joints[7] = .48, -.48
                s.state = worker.state()
                s.plan_path = lambda arm, pose=None, joints=None: [list(joints)]
                home = s.home
                s.home = mock.Mock(wraps=home if fast else lambda arms, speed: home(arms))
                completion = s.complete_task(evidence(s, complete=False, safe_to_finish=True))
                self.assertTrue(completion["ok"])
                self.assertEqual(completion["reason"], "swap_completion_observed")
                with self.assertRaises(ProgramFinished) as finished:
                    s.finish("observed swap complete")
                self.assertNotIn("FAILED", str(finished.exception))
                if fast:
                    self.assertEqual(s._wrap_up(), ["COMPLETED"] * 4)
                    self.assertEqual(worker.step, 699)
                    self.assertFalse(s.state["ended"])
                    self.assertEqual([e["skill"] for e in s._events[-3:]],
                                     ["open_gripper", "open_gripper", "home"])
                    self.assertEqual(worker.joints[:6] + worker.joints[7:13], [0.0] * 12)
                else:
                    with self.assertRaises(EpisodeEnded):
                        s._wrap_up()
                    self.assertEqual(worker.step, 700)
                s.home.assert_called_once_with(("left_arm", "right_arm"), speed=3.0)
                self.assertEqual(s.press.call_count, 3)  # no motion or press replay

    def test_confirmed_swap_blocks_transfer_facing_press_and_time_advancing_browse(self):
        s, _ = self.confirm_all_legs()
        before = json.loads(json.dumps(s._comp()["swap"]))
        ledger = json.loads(json.dumps(s._comp()["ledger"]))
        action = mock.Mock()
        calls = (lambda: s.transfer_item("first", "left_arm", evidence=evidence(s)),
                 lambda: s.transfer_plan([], evidence=evidence(s)),
                 lambda: s.swap_plan("first", "second", [0, -.35], self.swap_evidence(s)),
                 lambda: s.swap_facing_correct("first", "left_arm", evidence(s)),
                 lambda: s.orient_held("first", "left_arm", 0, evidence(s)),
                 lambda: s.swap_confirm(2, "blue", evidence=evidence(s)),
                 lambda: s._comp_motion("new_motion", ["new"], action, evidence(s)))
        for call in calls:
            result = call()
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "swap_complete_task_required")
            self.assertEqual(result["next_hop"], "complete_task")
            self.assertFalse(result["motion_executed"])
        for skill in ("move_eef", "home", "run_policy", "hold", "watch_scene"):
            self.assertEqual(s._ownership_guard(skill), "swap_complete_task_required")
        # Exercise the execution guard too, not only its private decision function.
        step = s.state["step"]
        with self.assertRaises(Escalation) as blocked:
            s._execute("move_eef", {}, action, 1, None, None)
        result = blocked.exception.trace
        self.assertFalse(result.ok)
        self.assertEqual(result.event.detail["next_hop"], "complete_task")
        self.assertEqual(s.state["step"], step)
        action.assert_not_called()
        self.assertEqual(s.press.call_count, 3)
        self.assertEqual(s._comp()["swap"], before)
        self.assertEqual(s._comp()["ledger"], ledger)
        with self.assertRaisesRegex(ValueError, "swap_complete_task_required"):
            s._complement_policy_arguments(None, ("left_arm", "right_arm"), False, None, None, False)
        self.assertEqual(s._comp()["last"]["next_hop"], "complete_task")
        self.assertEqual(s._policy_clients["wall-oss"].calls, [])
        reroute = s.failure_route("confirmation_downstroke_timeout", receipt=s._comp()["last_counted_press"])
        self.assertEqual(reroute["next_hop"], "complete_task")

    def test_swap_completion_rejections_are_explicit_and_do_not_silently_downgrade(self):
        for changes, failed_check in (({"source": ""}, "fresh_visual_evidence"),
                                      ({"observed_step": -1}, "fresh_visual_evidence"),
                                      ({"episode_id": "other"}, "fresh_visual_evidence"),
                                      ({"safe_to_finish": False}, "safe_to_finish"),
                                      ({"safe_to_finish": None}, "safe_to_finish")):
            with self.subTest(changes=changes):
                self.setUp()
                s, _ = self.confirm_all_legs()
                result = s.complete_task({**evidence(s, complete=False, safe_to_finish=True), **changes})
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "swap_completion_unverified")
                self.assertEqual(result["next_hop"], "complete_task")
                self.assertEqual(result["failed_checks"], [failed_check])
                self.assertTrue(result["checks"]["complete"])
                self.assertEqual(result["confirmed_legs"], 3)
                self.assertEqual(s._comp()["completion"], "unknown")
                self.assertEqual(s.press.call_count, 3)
        for invalid_state, failed_check in (("ledger", "identity_items_done"),
                                            ("cards", "cards_confirmed"), ("failed", "not_failed")):
            with self.subTest(invalid_state=invalid_state):
                self.setUp()
                s, _ = self.confirm_all_legs()
                if invalid_state == "ledger":
                    s._comp()["ledger"]["first"]["status"] = "pending"
                elif invalid_state == "cards":
                    s._comp()["cards"].append({"status": "pressed_unconfirmed"})
                else:
                    s._comp()["completion"] = "failed"
                result = s.complete_task(evidence(s, complete=False, safe_to_finish=True))
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "swap_completion_unverified")
                self.assertEqual(result["next_hop"], "failure_route")
                self.assertEqual(result["failed_checks"], [failed_check])
                self.assertTrue(result["checks"]["complete"])
                self.assertEqual(result["confirmed_legs"], 3)

    def test_confirmed_swap_missing_evidence_or_safety_is_explicitly_rejected(self):
        s, _ = self.confirm_all_legs()
        for e, failed in ((None, ["fresh_visual_evidence", "safe_to_finish"]),
                          ([], ["fresh_visual_evidence", "safe_to_finish"]),
                          ({}, ["fresh_visual_evidence", "safe_to_finish"]),
                          (evidence(s, complete=False), ["safe_to_finish"])):
            with self.subTest(evidence=e):
                result = s.complete_task(e)
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "swap_completion_unverified")
                self.assertEqual(result["failed_checks"], failed)
                self.assertTrue(result["checks"]["complete"])
                self.assertEqual(result["next_hop"], "complete_task")
                self.assertEqual(s._comp()["completion"], "unknown")
        self.assertEqual(s.press.call_count, 3)

    def test_confirmed_swap_reports_all_six_safety_failures_without_downgrading(self):
        s, _ = self.confirm_all_legs()
        before = json.loads(json.dumps(s._comp()["swap"]))
        s._comp().update(completion="failed", uncertain=True)
        s._comp()["ledger"]["first"]["status"] = "pending"
        s._comp()["cards"].append({"status": "pressed_unconfirmed"})
        e = {**evidence(s, complete=False, safe_to_finish=False), "observed_step": -1}
        result = s.complete_task(e)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "swap_completion_unverified")
        self.assertEqual(result["failed_checks"], ["fresh_visual_evidence", "safe_to_finish", "not_failed",
                                                  "motion_resolved", "identity_items_done", "cards_confirmed"])
        self.assertEqual(result["next_hop"], "failure_route")
        self.assertTrue(result["checks"]["complete"])
        self.assertEqual(s._comp()["swap"], before)
        self.assertEqual(s._comp()["completion"], "failed")
        self.assertEqual(s.press.call_count, 3)

    def test_third_confirm_uncertain_clearance_requires_inspect_then_complete_no_retry(self):
        s, result = self.confirm_all_legs(final_arm_clear=False)
        self.assertEqual(result["confirmed_legs"], 3)
        self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
        self.assertTrue(result["completion_required"])
        self.assertIn("resolve uncertainty before complete_task", result["message"])
        completion = s.complete_task(evidence(s, complete=False, safe_to_finish=True))
        self.assertFalse(completion["ok"])
        self.assertIn("motion_resolved", completion["failed_checks"])
        self.assertEqual(completion["next_hop"], "inspect_receipt_no_retry")
        blocked = s.swap_confirm(2, "blue", evidence=evidence(s))
        self.assertEqual(blocked["next_hop"], "inspect_receipt_no_retry")
        route = s.failure_route("final_lift_unverified", receipt=result["receipt"],
                                evidence=evidence(s, resolved_motion=True))
        self.assertEqual(route["next_hop"], "complete_task")
        self.assertTrue(s.complete_task(evidence(s, complete=False, safe_to_finish=True))["ok"])
        self.assertEqual(s.press.call_count, 3)

    def test_confirmed_swap_cannot_finish_success_without_complete_task(self):
        s, _ = self.confirm_all_legs()
        with self.assertRaises(ProgramFinished) as finished:
            s.finish("three legs done")
        self.assertIn("FAILED/UNKNOWN", str(finished.exception))
        self.assertEqual(s._wrap_up(), ["RECOVERY_FORBIDDEN"])

    def test_unverified_final_safety_can_finish_honestly_without_reopening_legs(self):
        s, _ = self.confirm_all_legs()
        result = s.complete_task(evidence(s, complete=False, safe_to_finish=False))
        self.assertEqual(result["reason"], "swap_completion_unverified")
        self.assertEqual(result["failed_checks"], ["safe_to_finish"])
        with self.assertRaises(ProgramFinished) as finished:
            s.failure_route(result["reason"], receipt=result, request="finish")
        self.assertIn("FAILED: swap_completion_unverified", str(finished.exception))
        self.assertEqual(s._comp()["swap"]["confirmed_legs"], 3)
        self.assertEqual(s._wrap_up(), ["RECOVERY_FORBIDDEN"])
        self.assertEqual(s.press.call_count, 3)

    def test_post_third_press_fresh_visual_needs_no_new_ledger_rebind(self):
        s, _ = self.plan_and_press({"ok": True, "presses": 1, "count_complete": True, "arm_clear": True})
        def press(*args, **kwargs):
            s.state["step"] += 1
            return {"ok": True, "presses": 1, "count_complete": True, "arm_clear": True}
        s.press.side_effect = press
        for index in range(3):
            pre_press = evidence(s, identity_unique=True, at_destination=True, facing_correct=True,
                                 released=True, target_unique=True)
            self.assertTrue(s.swap_confirm(index, "blue", evidence=pre_press)["ok"])
        stale = s.complete_task({**pre_press, "complete": False, "safe_to_finish": True})
        self.assertEqual(stale["failed_checks"], ["fresh_visual_evidence"])
        # No fresh locate/identity_ledger/placement_check required after the final press.
        s.identity_ledger = mock.Mock()
        self.assertTrue(s.complete_task(evidence(s, safe_to_finish=True))["ok"])
        s.identity_ledger.assert_not_called()
        self.assertEqual(s.press.call_count, 3)

    def test_unknown_or_partial_confirmation_never_represses(self):
        for raw in ({"ok": False, "presses": 0, "event": {"kind": "TIMEOUT"}},
                    {"ok": True, "presses": 1, "count_complete": False}):
            with self.subTest(raw=raw):
                self.setUp()
                s, e = self.plan_and_press(raw)
                result = s.swap_confirm(0, "blue", evidence=e)
                self.assertEqual(result["reason"], "confirm_uncertain")
                self.assertEqual(result["next_hop"], "finish_honest_failure")
                self.assertFalse(s.swap_confirm(0, "blue", evidence=e)["ok"])
                self.assertEqual(s._comp()["swap"]["confirmed_legs"], 0)
                self.assertEqual(s.press.call_count, 1)
                self.assertEqual(s.failure_route("confirmation_downstroke_timeout", receipt=raw)["next_hop"],
                                 "inspect_receipt_no_retry")

    def test_facing_small_measured_error_survives_two_checks_and_one_correction(self):
        s, e = self.plan_and_press({"ok": True, "presses": 1})
        facing = evidence(s, key="first", identity_unique=True, at_destination=True, released=True,
                          facing_correct=False, directed_axis=True, orientation_measured=True, yaw_deg=12, path_clear=True)
        for _ in range(3):
            result = s.failure_route("placement_unverified", request="finish", evidence=facing)
            self.assertEqual(result["next_hop"], "pick_observe_orient_reseat_then_confirm")
        self.assertFalse(s.swap_confirm(0, "blue", evidence=facing)["ok"])
        s.press.assert_not_called()
        s._eef_grip["left_arm"] = (0, 0, .04)
        s._eef_scene[0]["held_by"] = "left_arm"
        measured(s, s._eef_scene)
        s.identity_ledger(self.entries, self.swap_evidence(s))
        s.move_eef = mock.Mock(side_effect=lambda *_args, **_kwargs: trace(s))
        held = evidence(s, identity_unique=True, held_identity=True, directed_axis=True,
                        orientation_measured=True, path_clear=True, yaw_deg=12)
        self.assertTrue(s.orient_held("first", "left_arm", 0, held)["ok"])
        self.assertEqual(s._comp()["swap"]["facing_corrections"], [0])
        self.assertFalse(s.orient_held("first", "left_arm", 0, evidence(s, **{
            k: v for k, v in held.items() if k not in {"source", "episode_id", "observed_step"}}))["ok"])
        self.assertEqual(s.move_eef.call_count, 1)
        self.assertEqual(s._comp()["swap"]["confirmed_legs"], 0)
        s._eef_grip.clear()  # observed explicit reseating/release, not the turn's ok flag
        s._eef_scene[0].pop("held_by")
        self.assertTrue(s.swap_confirm(0, "blue", evidence=evidence(s, identity_unique=True, at_destination=True,
                        facing_correct=True, released=True, target_unique=True))["ok"])

    def test_facing_recovery_denies_unknown_identity_large_error_and_uncertain_motion(self):
        for overrides in ({"identity_unique": False}, {"directed_axis": False}, {"yaw_deg": 31}, {"key": "second"},
                          {"released": False}, {"observed_step": -1}):
            with self.subTest(overrides=overrides):
                self.setUp()
                s, e = self.plan_and_press({"ok": True, "presses": 1})
                facing = evidence(s, key="first", identity_unique=True, at_destination=True, released=True,
                                  facing_correct=False, directed_axis=True, orientation_measured=True, yaw_deg=12, path_clear=True)
                facing.update(overrides)
                with self.assertRaises(ProgramFinished):
                    s.failure_route("placement_unverified", request="finish", evidence=facing)
        self.setUp()
        s, e = self.plan_and_press({"ok": True, "presses": 1})
        facing = evidence(s, key="first", identity_unique=True, at_destination=True, released=True,
                          facing_correct=False, directed_axis=True, orientation_measured=True, yaw_deg=12, path_clear=True)
        self.assertEqual(s.failure_route("placement_unverified", motion_executed="unknown",
                         request="finish", evidence=facing)["next_hop"], "inspect_receipt_no_retry")
        self.assertEqual(s.failure_route("memory_identity_lost", evidence=facing)["next_hop"], "finish_honest_failure")

    def test_facing_unknown_held_identity_and_failed_turn_never_retry(self):
        s, e = self.plan_and_press({"ok": True, "presses": 1})
        s._eef_grip["left_arm"] = (0, 0, .04)
        s.move_eef = mock.Mock(side_effect=lambda *_args, **_kwargs: trace(s, "TIMEOUT"))
        held = evidence(s, identity_unique=True, held_identity=True, directed_axis=True,
                        orientation_measured=True, path_clear=True, yaw_deg=12)
        self.assertEqual(s.orient_held("first", "left_arm", 0, held)["reason"],
                         "swap_held_identity_or_facing_unverified")
        s.move_eef.assert_not_called()
        s._eef_scene[0]["held_by"] = "left_arm"
        result = s.orient_held("first", "left_arm", 0, held)
        self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
        s.failure_route("inspection", evidence=evidence(s, resolved_motion=True))
        self.assertFalse(s.orient_held("first", "left_arm", 0, evidence(s, identity_unique=True,
                         held_identity=True, directed_axis=True, orientation_measured=True, path_clear=True, yaw_deg=10))["ok"])
        self.assertEqual(s.move_eef.call_count, 1)

    def facing_api_session(self, index=0):
        s, _ = self.plan_and_press({"ok": True, "presses": 1, "count_complete": True})
        for previous in range(index):
            self.assertTrue(s.swap_confirm(previous, "blue", evidence=evidence(s, identity_unique=True,
                            at_destination=True, facing_correct=True, released=True, target_unique=True))["ok"])
        leg = s._comp()["swap"]["legs"][index]
        obj = s._eef_object(s._comp()["ledger"][leg["key"]]["object"])
        obj["position"][:2] = leg["xy"]
        measured(s, s._eef_scene)
        self.assertTrue(s.identity_ledger(self.entries, self.swap_evidence(s))["ok"])
        def pick(arm, object):
            self.assertNotIn(index, s._comp()["swap"].get("facing_corrections", []))
            trace(s)
            s._eef_object(object)["held_by"] = arm
            s._eef_grip[arm] = (0, 0, .04)
            return {"ok": True, "check": {"held": True}}
        def release(arm, **kwargs):
            self.assertEqual(s._comp()["swap"]["facing_recovery"]["phase"], "reseat_attempted")
            trace(s)
            s._eef_grip.pop(arm)
            for source in s._eef_scene:
                if source.get("held_by") == arm:
                    source["held_by"] = None
            return {"ok": True, "where": kwargs}
        s.eef_pick = mock.Mock(side_effect=pick)
        s.move_eef = mock.Mock(side_effect=lambda *_a, **_kw: trace(s))
        s.eef_release = mock.Mock(side_effect=release)
        facing = evidence(s, key=leg["key"], identity_unique=True, at_destination=True, released=True,
                          facing_correct=False, directed_axis=True, orientation_measured=True, yaw_deg=12, path_clear=True)
        return s, leg, facing

    def held_api_evidence(self, s, leg, *, bind=False, object=None, **extra):
        measured(s, s._eef_scene)  # NEW held observation; not the pick's held_by mutation.
        entry = dict(next(row for row in self.entries if row["key"] == leg["key"]))
        if object is not None:
            entry["object"] = object
        held = evidence(s, identity_unique=True, held_identity=True, directed_axis=True,
                        orientation_measured=True, path_clear=True, yaw_deg=12)
        held.update(extra)
        if bind:
            self.assertTrue(s.identity_ledger([entry], held)["ok"])
        return held, entry

    def test_facing_api_rebinds_and_completes_each_frozen_leg_without_confirming_by_motion(self):
        for index in range(3):
            with self.subTest(index=index):
                self.setUp()
                s, leg, facing = self.facing_api_session(index)
                before = s._comp()["swap"]["confirmed_legs"]
                for _ in range(3):
                    self.assertEqual(s.failure_route("placement_unverified", request="finish", evidence=facing)
                                     ["next_hop"], "pick_observe_orient_reseat_then_confirm")
                picked = s.swap_facing_correct(leg["key"], "left_arm", facing)
                self.assertTrue(picked["ok"], picked)
                self.assertEqual(picked["stage"], "picked")
                self.assertEqual(picked["next_hop"], "observe_rebind_held_then_swap_facing_correct")
                self.assertFalse(picked["correction_consumed"])
                self.assertNotIn(index, s._comp()["swap"].get("facing_corrections", []))
                self.assertFalse(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])
                s.move_eef.assert_not_called()
                s.eef_release.assert_not_called()
                held, entry = self.held_api_evidence(s, leg)
                original_orient = s._orient_held
                def orient(*args, **kwargs):
                    self.assertEqual(s._comp()["swap"]["facing_corrections"], [index])
                    self.assertEqual(s._comp()["swap"]["facing_recovery"]["phase"], "orient_reserved")
                    return original_orient(*args, **kwargs)
                s._orient_held = mock.Mock(side_effect=orient)
                reseated = s.swap_facing_correct(leg["key"], "left_arm", held, held_entry=entry)
                self.assertTrue(reseated["ok"], reseated)
                self.assertEqual(reseated["stage"], "placed_unverified")
                self.assertEqual(reseated["next_hop"], "reobserve_rebind_then_swap_confirm")
                self.assertTrue(reseated["correction_consumed"])
                s.eef_pick.assert_called_once_with("left_arm", entry["object"])
                s.move_eef.assert_called_once()
                self.assertEqual(s._orient_held.call_args.args, (leg["key"], "left_arm", leg["yaw_deg"], held))
                s.eef_release.assert_called_once_with("left_arm", x=leg["xy"][0], y=leg["xy"][1], withdraw=False)
                self.assertEqual(s._comp()["swap"]["confirmed_legs"], before)
                self.assertEqual(s._comp()["swap"]["facing_corrections"], [index])
                self.assertFalse(s.swap_confirm(index, "blue", evidence=held)["ok"])
                # Even fresh booleans cannot stand in for a NEW released-identity binding.
                confirm = evidence(s, identity_unique=True, at_destination=True, facing_correct=True,
                                   released=True, target_unique=True)
                self.assertFalse(s.swap_confirm(index, "blue", evidence=confirm)["ok"])
                self.assertNotIn(index, s._comp()["swap"]["attempted"])
                self.assertEqual(s.swap_facing_correct(leg["key"], "left_arm", evidence(s))["reason"],
                                 "swap_facing_correction_exhausted")
                measured(s, s._eef_scene)
                self.assertFalse(s.swap_confirm(index, "blue", evidence=confirm)["ok"])
                s.identity_ledger([entry], self.swap_evidence(s))
                self.assertTrue(s.swap_confirm(index, "blue", evidence=evidence(s, identity_unique=True,
                                at_destination=True, facing_correct=True, released=True, target_unique=True))["ok"])
                self.assertEqual(s._comp()["swap"]["confirmed_legs"], before + 1)
                self.assertEqual(s.press.call_count, index + 1)
                self.assertEqual(len(s._comp()["swap"]["receipts"]), index + 3)

    def test_facing_api_accepts_only_explicit_same_identity_current_id_and_prebound_path(self):
        s, leg, facing = self.facing_api_session()
        self.assertTrue(s.swap_facing_correct("first", "left_arm", facing)["ok"])
        s._eef_scene[0]["id"] = "o9"  # Labels change; no auto-matching by position/name.
        held, entry = self.held_api_evidence(s, leg, object="o9", bind=True)
        self.assertTrue(s.swap_facing_correct("first", "left_arm", held)["ok"])
        self.assertEqual(s._comp()["ledger"]["first"]["object"], "o9")

    def test_facing_api_three_corrections_and_counted_confirmations_keep_frozen_sequence(self):
        s, _, _ = self.facing_api_session()
        frozen = json.loads(json.dumps(s._comp()["swap"]["legs"]))
        def pick(arm, object):
            index = s._comp()["swap"]["confirmed_legs"]
            self.assertEqual(s._comp()["swap"].get("facing_corrections", []), list(range(index)))
            trace(s)
            s._eef_object(object)["held_by"] = arm
            s._eef_grip[arm] = (0, 0, .04)
            return {"ok": True, "check": "held after lift"}
        s.eef_pick.side_effect = pick
        for index, leg in enumerate(frozen):
            s._eef_object(s._comp()["ledger"][leg["key"]]["object"])["position"][:2] = leg["xy"]
            measured(s, s._eef_scene)
            s.identity_ledger(self.entries, self.swap_evidence(s))
            facing = evidence(s, key=leg["key"], identity_unique=True, at_destination=True, released=True,
                              facing_correct=False, directed_axis=True, orientation_measured=True, yaw_deg=30, path_clear=True)
            self.assertTrue(s.swap_facing_correct(leg["key"], "left_arm", facing)["ok"])
            held, entry = self.held_api_evidence(s, leg, yaw_deg=-30)
            self.assertTrue(s.swap_facing_correct(leg["key"], "left_arm", held, held_entry=entry)["ok"])
            self.assertEqual(s._comp()["swap"]["legs"], frozen)
            measured(s, s._eef_scene)
            s.identity_ledger([entry], self.swap_evidence(s))
            confirmed = s.swap_confirm(index, "blue", evidence=evidence(s, identity_unique=True,
                                     at_destination=True, facing_correct=True, released=True, target_unique=True))
            self.assertTrue(confirmed["ok"], confirmed)
            self.assertEqual(confirmed["confirmation"], "counted_press_complete")
            self.assertEqual(confirmed["confirmed_legs"], index + 1)
        self.assertEqual(s._comp()["swap"]["facing_corrections"], [0, 1, 2])
        self.assertEqual(s.eef_pick.call_count, 3)
        self.assertEqual(s.move_eef.call_count, 3)
        self.assertEqual(s.eef_release.call_count, 3)
        self.assertEqual(s.press.call_count, 3)
        self.assertTrue(s.complete_task(evidence(s, complete=True, safe_to_finish=True))["ok"])
        new_episode = self.make_session()
        self.assertNotIn("swap", new_episode._comp())

    def test_facing_api_rejects_ineligible_pre_pick_without_consuming_or_moving(self):
        for changes in ({"yaw_deg": 31}, {"yaw_deg": -31}, {"yaw_deg": 0}, {"yaw_deg": float("nan")},
                        {"identity_unique": False}, {"directed_axis": False}, {"at_destination": False},
                        {"released": False}, {"facing_correct": True}, {"path_clear": False},
                        {"observed_step": -1}, {"episode_id": "wrong"}):
            with self.subTest(changes=changes):
                self.setUp()
                s, leg, facing = self.facing_api_session()
                facing.update(changes)
                self.assertFalse(s.swap_facing_correct("first", "left_arm", facing)["ok"])
                self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
                s.eef_pick.assert_not_called()
                s.move_eef.assert_not_called()
                s.eef_release.assert_not_called()

    def test_facing_api_rejects_wrong_key_held_source_owner_uncertainty_and_reach(self):
        for case in ("wrong_key", "wrong_arm", "held", "uncertain", "terminal", "policy", "source_reach", "goal_reach"):
            with self.subTest(case=case):
                self.setUp()
                s, leg, facing = self.facing_api_session()
                key, arm = "first", "left_arm"
                if case == "wrong_key":
                    key = "second"
                elif case == "wrong_arm":
                    arm = "left"
                elif case == "held":
                    s._eef_scene[0]["held_by"] = arm
                elif case == "uncertain":
                    s._comp()["uncertain"] = True
                elif case == "terminal":
                    s._comp()["completion"] = "failed"
                elif case == "policy":
                    s._comp()["route"]["phase"] = "policy"
                elif case == "source_reach":
                    s._eef_scene[0]["reach"][arm] = "tilted"
                else:
                    s._eef_reach.return_value = {arm: "tilted"}
                self.assertFalse(s.swap_facing_correct(key, arm, facing)["ok"])
                s.eef_pick.assert_not_called()

    def test_facing_api_failed_or_unknown_grasp_spends_no_correction_and_never_orients_or_retries(self):
        for raw in ({"ok": False, "reason": "not held after the lift"},
                    {"ok": True},  # A bare success without grip/held_by is not held evidence.
                    {"ok": False, "reason": "could not get above it: TIMEOUT"},
                    {"ok": False, "receipt": {"event": {"kind": "TIMEOUT"}}},
                    {"ok": True, "motion_executed": "unknown"}):
            with self.subTest(raw=raw):
                self.setUp()
                s, leg, facing = self.facing_api_session()
                def fail(*args):
                    self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
                    trace(s)
                    return raw
                s.eef_pick.side_effect = fail
                result = s.swap_facing_correct("first", "left_arm", facing)
                self.assertFalse(result["ok"])
                self.assertEqual(result["receipt"]["pick"], raw)
                self.assertFalse(result["correction_consumed"])
                self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
                if s._swap_motion_unknown(raw):
                    self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
                    self.assertTrue(s._comp()["uncertain"])
                    s.failure_route("inspection", evidence=evidence(s, resolved_motion=True))
                else:
                    self.assertEqual(result["next_hop"], "failure_route")
                    self.assertEqual(result["reason"], "grasp_failed" if not raw["ok"] else
                                     "swap_held_identity_or_facing_unverified")
                # Fresh, qualified released evidence cannot silently repeat even a
                # known failed grasp; its attempt receipt is independent of yaw budget.
                measured(s, s._eef_scene)
                s.identity_ledger(self.entries, self.swap_evidence(s))
                facing["observed_step"] = s.state["step"]
                rejected = s.swap_facing_correct("first", "left_arm", facing)
                self.assertEqual(rejected["reason"], "same_failed_action_no_retry")
                self.assertEqual(s._comp()["swap"]["receipts"][-1],
                                 {"facing_pick": 0, "receipt": result["receipt"]})
                self.assertFalse(rejected["correction_consumed"])
                self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
                s.eef_pick.assert_called_once()
                s.move_eef.assert_not_called()
                s.eef_release.assert_not_called()

    def test_facing_api_failed_grasp_preserves_prior_legs_correction_budget(self):
        for index in (1, 2):
            with self.subTest(index=index):
                self.setUp()
                s, leg, facing = self.facing_api_session(index)
                swap = s._comp()["swap"]
                swap["facing_corrections"] = list(range(index))
                raw = {"ok": False, "reason": "not held after the lift"}
                s.eef_pick.side_effect = lambda *_args: (trace(s), raw)[1]
                result = s.swap_facing_correct(leg["key"], "left_arm", facing)
                self.assertEqual(result["reason"], "grasp_failed")
                self.assertEqual(result["next_hop"], "failure_route")
                self.assertFalse(result["correction_consumed"])
                self.assertEqual(swap["facing_corrections"], list(range(index)))
                self.assertEqual(swap["confirmed_legs"], index)
                s.eef_pick.assert_called_once()
                s.move_eef.assert_not_called()
                s.eef_release.assert_not_called()

    def test_facing_api_requires_new_measurement_and_same_fresh_held_binding(self):
        for case in ("no_locate", "stale_ledger", "stale_step", "stale_episode", "held_identity", "directed_axis",
                     "missing_yaw", "not_held", "wrong_held_arm", "wrong_entry", "changed_identity",
                     "changed_destination", "alias", "large_held_error"):
            with self.subTest(case=case):
                self.setUp()
                s, leg, facing = self.facing_api_session()
                self.assertTrue(s.swap_facing_correct("first", "left_arm", facing)["ok"])
                held, entry = self.held_api_evidence(s, leg)
                supplied_entry = entry
                if case == "no_locate":
                    s._eef_measure_serial = s._comp()["swap"]["facing_recovery"]["pick_measure_serial"]
                elif case == "stale_ledger":
                    supplied_entry = None
                elif case == "stale_step":
                    held["observed_step"] = -1
                elif case == "stale_episode":
                    held["episode_id"] = "wrong"
                elif case in {"held_identity", "directed_axis"}:
                    held[case] = False
                elif case == "missing_yaw":
                    held.pop("yaw_deg")
                elif case == "not_held":
                    s._eef_grip.clear()
                elif case == "wrong_held_arm":
                    s._eef_scene[0]["held_by"] = "right_arm"
                elif case == "wrong_entry":
                    entry["object"] = "o2"
                elif case == "changed_identity":
                    entry["identity"] = "nearest block"
                elif case == "changed_destination":
                    entry["destination"] = "different mat"
                elif case == "alias":
                    s._eef_scene[0]["aliases"] = ["unresolved twin"]
                else:
                    held["yaw_deg"] = 31
                original_object = s._comp()["ledger"]["first"]["object"]
                self.assertFalse(s.swap_facing_correct("first", "left_arm", held, held_entry=supplied_entry)["ok"])
                self.assertEqual(s._comp()["ledger"]["first"]["object"], original_object)
                s.eef_pick.assert_called_once()
                s.move_eef.assert_not_called()
                s.eef_release.assert_not_called()
                self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])

    def test_facing_api_no_motion_rebind_rejection_can_be_resolved_without_second_pick(self):
        s, leg, facing = self.facing_api_session()
        self.assertTrue(s.swap_facing_correct("first", "left_arm", facing)["ok"])
        self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
        self.assertEqual(s.swap_facing_correct("first", "left_arm", facing)["next_hop"],
                         "observe_rebind_held_then_swap_facing_correct")
        self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
        held, entry = self.held_api_evidence(s, leg)
        self.assertTrue(s.swap_facing_correct("first", "left_arm", held, held_entry=entry)["ok"])
        self.assertEqual(s._comp()["swap"]["facing_corrections"], [0])
        s.eef_pick.assert_called_once()

    def test_facing_api_pending_and_consumed_budget_cannot_escape_through_public_orient_held(self):
        s, leg, facing = self.facing_api_session()
        s.swap_facing_correct("first", "left_arm", facing)
        held, entry = self.held_api_evidence(s, leg, bind=True)
        self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
        rejected = s.orient_held("first", "left_arm", 0, held)
        self.assertEqual(rejected["reason"], "swap_facing_staged_recovery_required")
        self.assertEqual(rejected["next_hop"], "observe_rebind_held_then_swap_facing_correct")
        self.assertEqual(s._comp()["swap"].get("facing_corrections", []), [])
        s.move_eef.assert_not_called()
        self.assertTrue(s.swap_facing_correct("first", "left_arm", held)["ok"])
        self.assertEqual(s._comp()["swap"]["facing_corrections"], [0])
        self.assertFalse(s.orient_held("first", "left_arm", 0, evidence(s, **{
                         k: v for k, v in held.items() if k != "observed_step"}))["ok"])
        s.move_eef.assert_called_once()

    def test_facing_api_pick_cannot_be_confirmed_from_booleans_or_pre_pick_locate_serial(self):
        s, leg, facing = self.facing_api_session()
        original_pick = s.eef_pick.side_effect
        def pick_with_internal_measure(*args):
            result = original_pick(*args)
            measured(s, s._eef_scene)  # Even an internal pick-check locate is not new caller evidence.
            return result
        s.eef_pick.side_effect = pick_with_internal_measure
        self.assertTrue(s.swap_facing_correct("first", "left_arm", facing)["ok"])
        held = evidence(s, identity_unique=True, held_identity=True, directed_axis=True,
                        orientation_measured=True, path_clear=True, yaw_deg=12)
        self.assertFalse(s.swap_facing_correct("first", "left_arm", held, held_entry=self.entries[0])["ok"])
        self.assertFalse(s.swap_confirm(0, "blue", evidence=evidence(s, identity_unique=True,
                         at_destination=True, facing_correct=True, released=True, target_unique=True))["ok"])
        self.assertEqual(s._comp()["swap"]["attempted"], [])
        s.press.assert_not_called()
        s.move_eef.assert_not_called()

    def test_facing_api_turn_or_release_failure_never_retries_even_after_resolution(self):
        for case in ("failed_turn", "unknown_turn", "failed_release", "unknown_release", "pick_exception"):
            with self.subTest(case=case):
                self.setUp()
                s, leg, facing = self.facing_api_session()
                if case == "pick_exception":
                    s.eef_pick.side_effect = RuntimeError("lost motion response")
                    result = s.swap_facing_correct("first", "left_arm", facing)
                    self.assertEqual(result["receipt"]["exception"], "lost motion response")
                else:
                    s.swap_facing_correct("first", "left_arm", facing)
                    held, entry = self.held_api_evidence(s, leg)
                    if case.endswith("turn"):
                        s.move_eef.side_effect = lambda *_a, **_kw: trace(s, "TIMEOUT" if case == "unknown_turn" else "UNREACHABLE")
                    else:
                        raw = {"ok": False, "reason": "release failed"}
                        if case == "unknown_release":
                            raw["receipt"] = {"event": {"kind": "TIMEOUT"}}
                        s.eef_release.side_effect = lambda *_a, **_kw: (trace(s), raw)[1]
                    result = s.swap_facing_correct("first", "left_arm", held, held_entry=entry)
                self.assertFalse(result["ok"])
                self.assertEqual(s._comp()["swap"].get("facing_corrections", []),
                                 [] if case == "pick_exception" else [0])
                self.assertEqual(result["correction_consumed"], case != "pick_exception")
                if case.startswith("unknown") or case == "pick_exception":
                    self.assertEqual(result["next_hop"], "inspect_receipt_no_retry")
                    s.failure_route("inspection", evidence=evidence(s, resolved_motion=True))
                else:
                    self.assertEqual(result["next_hop"], "failure_route")
                self.assertFalse(s.swap_facing_correct("first", "left_arm", evidence(s))["ok"])
                s.eef_pick.assert_called_once()
                self.assertEqual(s.move_eef.call_count, 0 if case == "pick_exception" else 1)
                self.assertEqual(s.eef_release.call_count, 1 if case.endswith("release") else 0)


if __name__ == "__main__":
    unittest.main()
