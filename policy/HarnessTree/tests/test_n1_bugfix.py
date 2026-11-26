"""Offline checks for the digit pad undercount and the unverified egg exit."""

import ast
import math
import unittest
from pathlib import Path
from unittest.mock import Mock, call

import cv2
import numpy as np

from XPolicyLab.policy.HarnessTree import skill_helpers


PROGRAMS = Path(__file__).resolve().parents[1] / "programs"
DIGIT_SKILLS = ("arrange_largest_number", "arrange_largest_number_random")
PAD_SPACING = 0.085


def skill_tree(name):
    return ast.parse((PROGRAMS / f"{name}.py").read_text(encoding="utf-8"))


def exec_nodes(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<skill fixture>", "exec"), scope)


def digit_scope(name):
    names = {"Grid", "gradient_map", "certain", "merge", "read_scene", "in_range", "dedupe", "face_filter", "assign", "run"}
    scope = dict(np=np, cv2=cv2, math=math, skill_helpers=skill_helpers, PAD_SPACING=PAD_SPACING)
    exec_nodes([node for node in skill_tree(name).body if getattr(node, "name", None) in names], scope)
    return scope


def pad_fixture(scope, strengths):
    grid = scope["Grid"]((-0.46, 0.46, -0.34, 0.0), 0.001)
    rows, cols = np.indices((grid.H, grid.W))
    x, y = grid.xy(rows, cols)
    image = np.full((grid.H, grid.W, 3), 40, dtype=np.uint8)
    for index, strength in enumerate(strengths):
        center = (index - (len(strengths) - 1) / 2) * PAD_SPACING
        image[np.hypot(x - center, y + 0.078) < 0.040] = strength
    return grid, image, np.ones((grid.H, grid.W), dtype=bool)


class DigitPlanningTests(unittest.TestCase):
    def test_combined_rim_support_finds_outer_pads_and_does_not_invent_extra_ones(self):
        scope = digit_scope(DIGIT_SKILLS[0])
        for strengths in ((200, 220), (170, 210, 220, 173), (140, 180, 183, 178, 145)):
            with self.subTest(count=len(strengths)):
                grid, image, valid = pad_fixture(scope, strengths)
                gradient = scope["gradient_map"](image, valid)
                score, count, y, radius, dx = skill_helpers.fit_circular_pad_row(gradient, grid, PAD_SPACING)
                self.assertGreater(score, 0)
                self.assertEqual(count, len(strengths))
                self.assertAlmostEqual(y, -0.078, delta=0.002)
                self.assertAlmostEqual(radius, 0.040, delta=0.002)
                self.assertAlmostEqual(dx, 0.0, delta=0.002)

    def test_no_rim_support_does_not_produce_pads(self):
        scope = digit_scope(DIGIT_SKILLS[0])
        grid, image, valid = pad_fixture(scope, ())
        self.assertIsNone(skill_helpers.fit_circular_pad_row(scope["gradient_map"](image, valid), grid, PAD_SPACING))

    def test_both_skills_attempt_motion_only_for_supported_plan(self):
        for name in DIGIT_SKILLS:
            for values, expected in (((8, 7, 2, 0), [8, 7, 2, 0]), ((8, 8, 2), [8, 8, 2]),
                                     ((9, 8, 7, 2, 0), [])):
                with self.subTest(skill=name, values=values):
                    scope = digit_scope(name)
                    grid, image, valid = pad_fixture(scope, (170, 210, 220, 173))
                    digits = [dict(best=value, score=0.95, margin=0.15, src="left", partial=False,
                                   x=-0.36 + 0.08 * index, bx=-0.36 + 0.08 * index,
                                   y=-0.22, by=-0.22, yaw=0, angles={value: 0})
                              for index, value in enumerate(values)]
                    robo = Mock(image=Mock(return_value=image))
                    place = Mock(return_value="placed")
                    scope.update(robo=robo, say=Mock(), PAD_TOP=0.770, PAD_GEOM={}, STEPS=[54],
                                 ACTIVE=[None], PLAN=[], PLAN_PADS=[], overview=Mock(return_value=(digits, [], grid)),
                                 camera=Mock(return_value=(None, None)), rectify=Mock(return_value=(image, valid)),
                                 rep_face=Mock(return_value=None), search_look=Mock(return_value=[]),
                                 edge_candidates=Mock(return_value=[]), sam_dets=Mock(return_value=[]),
                                 occupancy=Mock(), home=Mock(), left_budget=Mock(return_value=600),
                                 plan_job=Mock(return_value=[("left", 90)]), use=Mock(),
                                 pick=Mock(side_effect=lambda arm, x, y, value, **kwargs: ("held", value)),
                                 place_and_check=place, edge_at=Mock())
                    scope["run"]()
                    robo.image.assert_called_once_with("cam_head")
                    scope["rectify"].assert_called_once_with(image, None, None, 0.770, grid, False)
                    self.assertEqual([args[1] for args, _ in place.call_args_list], expected)
                    self.assertEqual([args[2] for args, _ in place.call_args_list], list(range(len(expected))))
                    self.assertEqual(bool(scope["PLAN"]), len(expected) == 4)
                    if len(values) > 4:
                        scope["say"].assert_any_call("ambiguous scene: 5 plausible digits for 4 supported pads")


class EggFinalizationTests(unittest.TestCase):
    def test_unverified_loading_finishes_open_and_real_errors_propagate(self):
        tail = skill_tree("fill_egg_holder").body[-2:]
        args = (0.0, -0.2, 0.3, -0.1, "right", 1, "cam_right_wrist")
        for outcome in (False, True, RuntimeError("motion error")):
            with self.subTest(outcome=outcome):
                actions = Mock()
                actions.home.return_value.reached = True
                fill = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                scope = dict(robo=actions, grip=actions.grip, P=Mock(), ENV=[548], ATT=[], note_steps=Mock(),
                             perceive_holder_and_basket=Mock(return_value=args), act_fill_cups=fill,
                             act_close_lid_and_home=actions.close_lid)
                if isinstance(outcome, Exception):
                    with self.assertRaisesRegex(RuntimeError, "motion error"):
                        exec_nodes(tail, scope)
                    self.assertEqual(actions.mock_calls, [])
                else:
                    exec_nodes(tail, scope)
                    expected = [call.close_lid(*args[:2], args[4])] if outcome else [
                        call.grip("right", actions.OPEN), call.home("both")]
                    self.assertEqual(actions.mock_calls, expected)
                    if not outcome:
                        scope["note_steps"].assert_called_once_with(actions.home.return_value)
                        scope["P"].assert_any_call("egg loading not verified; leaving the lid open")
                        scope["P"].assert_any_call("lid", False, "home", True, "env", 548)

    def test_grasp_crossing_loading_cutoff_cannot_reach_lid_verification(self):
        steps = [508]

        def grasp(*args):
            steps[0] = 548
            return True

        scope = dict(ENV=steps, CUPS=[], CUP_DX=0.0271, CUP_DY=0.0275, BC=[], P=Mock(),
                     egg_blobs=Mock(return_value=[dict(cup=index, seated=True) for index in range(3)]),
                     report_blobs=Mock(), grasp_egg=Mock(side_effect=grasp),
                     place=Mock(), look_eggs=Mock(), hover_basket=Mock())
        exec_nodes([node for node in skill_tree("fill_egg_holder").body
                    if getattr(node, "name", None) == "act_fill_cups"], scope)
        self.assertFalse(scope["act_fill_cups"](0.0, -0.2, 0.3, -0.1, "right", 1, "cam_right_wrist"))
        scope["egg_blobs"].assert_called_once()
        scope["place"].assert_not_called()
        scope["look_eggs"].assert_not_called()
