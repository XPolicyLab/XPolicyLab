from copy import deepcopy
import importlib
import math
from harness.execution import ExecutionError
from skills.shared.motion.observed_start_phase_execution.implementation import PublicRobotAPI, PhasedProgram, validate_phase_feedback
ACTION_BINDINGS = {'cover_scene_view': 'tasks/cover_blocks/skills/cover_scene_view/implementation.py', 'cover_cup_transfer': 'tasks/cover_blocks/skills/cover_cup_transfer/implementation.py', 'equation_scene_view': 'tasks/solve_equation/skills/equation_scene_view/implementation.py', 'equation_token_transfer': 'tasks/solve_equation/skills/equation_token_transfer/implementation.py', 'swap_scene_view': 'tasks/swap_blocks/skills/swap_scene_view/implementation.py', 'swap_mat_move': 'tasks/swap_blocks/skills/swap_mat_move/implementation.py', 'swap_confirm_move': 'tasks/swap_blocks/skills/swap_confirm_move/implementation.py', 'swap_return_both_home': 'tasks/swap_blocks/skills/return_both_home/implementation.py', 'button_view': 'tasks/press_by_number/skills/button_view/implementation.py', 'press_card_button': 'tasks/press_by_number/skills/press_card_button/implementation.py', 'active_surface_view': 'skills/shared/perception/active_surface_view/implementation.py', 'stack_colour_pair': 'tasks/stack_blocks_by_language/skills/stack_colour_pair/implementation.py', 'stage_stack_foundation': 'tasks/stack_blocks_by_language/skills/stage_stack_foundation/implementation.py', 'return_episode_home': 'skills/shared/motion/return_episode_home_observed_start/implementation.py', 'counted_press': 'skills/shared/motion/counted_press/implementation.py', 'grasp_test_lift': 'tasks/stack_blocks_by_language/skills/grasp_test_lift/implementation.py', 'grasp_visible_block': 'skills/shared/motion/visible_block_grasp/implementation.py', 'stack_place': 'tasks/stack_blocks_by_language/skills/stack_place/implementation.py','pour_segment': 'skills/shared/motion/incremental_tool_tilt/implementation.py'}
PHASE_EXECUTOR = 'skills/shared/motion/observed_start_phase_execution/implementation.py'
ACTION_LIMITS = {'counted_press': 256, 'grasp_test_lift': 128, 'stack_place': 128, 'align_yaw': 64, 'pour_segment': 192}
ACTION_LIMITS.update(stack_colour_pair=192, return_episode_home=96)
ACTION_LIMITS['stage_stack_foundation'] = 192
ACTION_LIMITS['active_surface_view'] = 32
ACTION_LIMITS['grasp_visible_block'] = 128
ACTION_LIMITS.update(button_view=32, press_card_button=256)
ACTION_LIMITS.update(swap_scene_view=32, swap_mat_move=256, swap_confirm_move=96)
ACTION_LIMITS['swap_return_both_home'] = 96
ACTION_LIMITS.update(equation_scene_view=32, equation_token_transfer=256)
ACTION_LIMITS.update(align_scene_view=32, dual_bar_stage=192)
ACTION_LIMITS['align_paired_view'] = 32
ACTION_BINDINGS.update(t_scene_view='tasks/swap_T/skills/t_scene_view/implementation.py', t_transfer_stage='tasks/swap_T/skills/t_transfer_stage/implementation.py')
ACTION_LIMITS.update(t_scene_view=32, t_transfer_stage=256)
ACTION_LIMITS.update(cover_scene_view=32, cover_cup_transfer=256)
ACTION_BINDINGS.update(board_scene_view='tasks/play_tic_tac_toe/skills/board_scene_view/implementation.py', board_turn_stage='tasks/play_tic_tac_toe/skills/board_turn_stage/implementation.py', board_wait_opponent='tasks/play_tic_tac_toe/skills/wait_opponent/implementation.py')
ACTION_LIMITS.update(board_scene_view=32, board_turn_stage=256, board_wait_opponent=250)
ACTION_BINDINGS['board_return_idle_wrist'] = 'tasks/play_tic_tac_toe/skills/board_return_idle_wrist/implementation.py'
ACTION_LIMITS['board_return_idle_wrist'] = 96
ACTION_MINIMUM_STEPS = {'grasp_test_lift': 22, 'stack_place': 14, 'align_yaw': 5, 'stack_colour_pair': 22, 'return_episode_home': 4, 'active_surface_view': 9}
ACTION_MINIMUM_STEPS['stage_stack_foundation'] = 22
ACTION_MINIMUM_STEPS.update(cover_scene_view=9, cover_cup_transfer=2)
ACTION_MINIMUM_STEPS.update(board_scene_view=9, board_turn_stage=2, board_wait_opponent=12)
ACTION_MINIMUM_STEPS['board_return_idle_wrist'] = 2
ACTION_MINIMUM_STEPS['grasp_visible_block'] = 22
ACTION_MINIMUM_STEPS.update(button_view=9, press_card_button=16)
ACTION_MINIMUM_STEPS.update(swap_scene_view=9, swap_mat_move=23, swap_confirm_move=8)
ACTION_MINIMUM_STEPS['swap_return_both_home'] = 2
ACTION_MINIMUM_STEPS.update(equation_scene_view=9, equation_token_transfer=10)
ACTION_MINIMUM_STEPS.update(align_scene_view=9, dual_bar_stage=2)
ACTION_MINIMUM_STEPS['align_paired_view'] = 9
ACTION_MINIMUM_STEPS.update(t_scene_view=9, t_transfer_stage=2)
COLOURS = ('blue', 'green', 'red', 'yellow', 'orange', 'cyan')
SIDE = {'type': 'string', 'enum': ['left', 'right']}
DELTA = {'type': 'array', 'items': {'type': 'number', 'minimum': -0.15, 'maximum': 0.15}, 'minItems': 3, 'maxItems': 3}
COLOUR = {'type': 'string', 'enum': list(COLOURS)}
CLEARANCE = {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 0.06}
MAT_ROLE = {'type': 'string', 'enum': ['mat_a', 'mat_b', 'mat_c']}
MOVE_INDEX = {'type': 'integer', 'minimum': 1, 'maximum': 3}
ACTION_PARAMETERS = {'board_return_idle_wrist': {'side': SIDE, 'turn': {'type': 'integer', 'minimum': 0, 'maximum': 3}, 'cell': {'type': 'integer', 'minimum': 0, 'maximum': 8}, 'source': {'type': 'integer', 'minimum': 0, 'maximum': 4}}, 'board_scene_view': {'side': SIDE, 'turn': {'type': 'integer', 'minimum': 0, 'maximum': 4}}, 'board_turn_stage': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'stage': {'type': 'string', 'enum': ['place', 'home']}, 'turn': {'type': 'integer', 'minimum': 0, 'maximum': 4}, 'cell': {'type': 'integer', 'minimum': 0, 'maximum': 8}, 'source': {'type': 'integer', 'minimum': 0, 'maximum': 4}}, 'board_wait_opponent': {'side': SIDE, 'turn': {'type': 'integer', 'minimum': 0, 'maximum': 3}, 'cell': {'type': 'integer', 'minimum': 0, 'maximum': 8}, 'source': {'type': 'integer', 'minimum': 0, 'maximum': 4}}, 'cover_scene_view': {'side': SIDE, 'stage': {'type': 'string', 'enum': ['cover_first', 'cover_second', 'cover_third', 'uncover_red', 'uncover_green', 'uncover_blue']}}, 'cover_cup_transfer': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'stage': {'type': 'string', 'enum': ['cover_first', 'cover_second', 'cover_third', 'uncover_red', 'uncover_green', 'uncover_blue', 'home_left', 'home_right', 'settle']}, 'colour': {'type': 'string', 'enum': ['red', 'green', 'blue']}}, 'align_paired_view': {'side': {'type': 'string', 'enum': ['left']}}, 't_scene_view': {'side': SIDE}, 't_transfer_stage': {'side': SIDE, 'stage': {'type': 'string', 'enum': ['pick_left', 'park_left', 'pick_right', 'place_right', 'clear_right', 'place_left', 'clear_left', 'settle']}}, 'equation_scene_view': {'side': SIDE, 'target': {'type': 'string', 'enum': ['source', 'answer_pad']}}, 'equation_token_transfer': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'stage': {'type': 'string', 'enum': ['initial', 'after_relay']}}, 'swap_scene_view': {'side': SIDE, 'initial_empty': MAT_ROLE, 'move_index': MOVE_INDEX, 'target': {'type': 'string', 'enum': ['mat_a', 'mat_b', 'mat_c', 'confirmation_cap']}}, 'swap_mat_move': {'side': SIDE, 'initial_empty': MAT_ROLE, 'move_index': MOVE_INDEX, 'source_mat': MAT_ROLE, 'destination_mat': MAT_ROLE}, 'swap_confirm_move': {'side': SIDE, 'initial_empty': MAT_ROLE, 'move_index': MOVE_INDEX, 'count': {'type': 'integer', 'minimum': 1, 'maximum': 1}, 'retract_home': {'type': 'boolean'}}, 'button_view': {'side': SIDE, 'button_role': {'type': 'string', 'enum': ['first_red', 'second_red', 'confirm_blue']}}, 'press_card_button': {'side': SIDE, 'button_role': {'type': 'string', 'enum': ['first_red', 'second_red', 'confirm_blue']}, 'count': {'type': 'integer', 'minimum': 1, 'maximum': 9}}, 'active_surface_view': {'side': SIDE, 'colour': COLOUR}, 'stack_colour_pair': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'colour': COLOUR, 'support_colour': COLOUR}, 'stage_stack_foundation': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'colour': COLOUR}, 'return_episode_home': {'side': SIDE}, 'swap_return_both_home': {'side': {'type': 'string', 'enum': ['left']}}, 'counted_press': {'side': SIDE, 'delta': DELTA, 'count': {'type': 'integer', 'minimum': 1, 'maximum': 12}, 'depth': {'type': 'number', 'minimum': 0.003, 'maximum': 0.025}, 'dwell': {'type': 'integer', 'minimum': 1, 'maximum': 8}}, 'grasp_test_lift': {'side': SIDE, 'delta': DELTA, 'clearance': CLEARANCE, 'colour': COLOUR, 'orientation': {'type': 'string', 'enum': ['keep', 'down']}}, 'grasp_visible_block': {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'colour': COLOUR}, 'stack_place': {'side': SIDE, 'delta': DELTA, 'clearance': CLEARANCE, 'colour': COLOUR, 'support_colour': COLOUR}, 'align_yaw': {'side': SIDE, 'angle': {'type': 'number', 'minimum': -math.pi / 12, 'maximum': math.pi / 12}, 'colour': COLOUR, 'reference_colour': COLOUR}, 'pour_segment': {'side': SIDE, 'axis': {'type': 'array', 'items': {'type': 'number', 'minimum': -1, 'maximum': 1}, 'minItems': 3, 'maxItems': 3}, 'angle': {'type': 'number', 'minimum': -math.pi / 3, 'maximum': math.pi / 3}, 'dwell': {'type': 'integer', 'minimum': 1, 'maximum': 8}, 'return_upright': {'type': 'boolean'}}}
ACTION_CONTRACTS = {'swap_return_both_home': {'precondition': 'The deterministic swap workflow has completed its three native transfer/confirmation boundaries.', 'result': 'Plan both own episode home joint paths and merge them on the original shared official clock, preserving opening commands.', 'uncertainty': 'Both arm endpoint errors remain public feedback; a home command is not an official success claim.'}, 'board_return_idle_wrist': {'precondition': 'An actual own placement and own home precede a wait, while the other own wrist still differs from its public episode home.', 'result': 'Restore only that idle own arm to episode home joints without advancing the board command cursor.', 'uncertainty': 'No opponent, scene, score or game-trigger state is read or changed; this is a public own-robot reset.'}, 'board_scene_view': {'precondition': 'A current visible public board pattern and an eligible own open wrist need another metric view.', 'result': 'Issue the bounded original wrist-view phases before current source and board qualification.', 'uncertainty': 'A view cannot claim contact, hidden board ownership or official success.'}, 'board_turn_stage': {'precondition': 'The next own turn is a current-public legal minimax placement or its own joint-home return.', 'result': 'Retain the mature token approach, dense grip, closed carry, release, retract and own-joint home semantics.', 'uncertainty': 'Own commanded cells and image occupancy are distinct; neither is contact or official completion.'}, 'board_wait_opponent': {'precondition': 'A positively issued own placement and own home precede the nonfinal public opponent wait.', 'result': 'Hold own commands and require the original occupied-count and consecutive stable public-frame condition.', 'uncertainty': 'Timeout is an explicit finite action-budget outcome; no hidden opponent state or score is read.'}, 'cover_scene_view': {'precondition': 'The public instruction requires cups covering blocks; current cap or block surfaces need another actual wrist view and both own gripper commands are open near home.', 'result': 'Acquire a bounded existing public wrist-camera view without fixed image crops or asset-height tables.', 'uncertainty': 'A view does not establish metric geometry, grasp, real finger contact or official success.'}, 'cover_cup_transfer': {'precondition': 'Current public cap rays and a strict action-bearing same-episode prefix qualify only the next cover, uncover, own-joint reset or settle stage.', 'result': 'Retain the original cup/block pairing, red-green-blue uncover order, original cup homes, dense gripper sampling, closed carry and both own-joint resets.', 'uncertainty': 'Initially observed homes are return goals, not current object poses. Command completion and release imagery never prove contact or official success.'}, 't_scene_view': {'precondition': 'Two declared T silhouettes are visible but metric correspondence is uncertain; both own public gripper commands are open and the selected wrist is near its observed home.', 'result': 'Acquire one bounded actual public wrist view and return current same-colour planar evidence.', 'uncertainty': 'An aimed wrist or silhouette does not establish contact, layout coordinates or a known asset height.'}, 't_transfer_stage': {'precondition': 'A qualified current T exchange and a strict prefix of this episode own issued-command receipts permit only the next serial stage.', 'result': 'Reuse mature left-pick, closed-gripper parking, right-pick, right-first placement and own-home semantics; reuse only own publicly derived command targets.', 'uncertainty': 'A commanded rigid grasp transform is not measured attachment. Partial commands cannot be replayed and no stage reports benchmark success.'}, 'equation_scene_view': {'precondition': 'A current equation completion identifies a visible answer-token or missing-pad region; both public gripper commands are open and the selected own arm is near observed home.', 'result': 'Reuse the original target-directed public wrist-view recipe, then reconstruct only independently qualified current source or pad surfaces.', 'uncertainty': 'An aimed camera is not an object pose; unavailable or conflicting same-instance RGB rays remain uncertain.'}, 'equation_token_transfer': {'precondition': 'The original public recognizer and arithmetic beam identify the missing token, current source/pad surfaces are independently qualified and the selected own arm has an open public command.', 'result': 'Execute the mature glyph/operator grasp, bounded retry, carry or table relay, opposite-arm regrasp, original contact fallback, release and retreat.', 'uncertainty': 'Gripper values are not true fingers or contact. A relay must be publicly reacquired; issued actions and robot arrival never prove official completion.'}, 'swap_scene_view': {'precondition': 'The actual instruction and current RGB establish the empty-support swap; the selected role is currently visible and both public grippers are open near episode home.', 'result': 'Reuse the existing bounded public wrist-view recipe for one current support or confirmation cap.', 'uncertainty': 'A completed view does not establish a metric surface or completed transfer.'}, 'swap_mat_move': {'precondition': 'Current RGB establishes distinct occupied/empty support roles and public metric geometry for the requested next transfer of the original swap procedure.', 'result': 'Execute one original approach/grasp/close/settle/lift/transfer/place/release/retreat sequence, then inspect visible occupancy.', 'uncertainty': 'Command closure and robot endpoint checks are not finger, contact, identity or official success measurements.'}, 'swap_confirm_move': {'precondition': 'Current visible occupancy qualifies the requested transferred stage and public multi-view cap rays qualify contact motion.', 'result': 'Execute one original confirmation stroke, preserving its span and hold duration; only non-final stages retract home.', 'uncertainty': 'One pose-qualified command cycle is not the true button counter or official success.'}, 'button_view': {'precondition': 'The actual instruction requires red-button count cards followed by blue confirmation; current RGB establishes the selected cap and public open grippers.', 'result': 'Reuse the existing bounded own-wrist view recipe, without contact or guessed object coordinates.', 'uncertainty': 'A completed camera motion is not a triangulated contact surface or a confirmed button press.'}, 'press_card_button': {'precondition': 'The requested count matches the current associated card and the cap has consistent unoccluded public multi-view rays; the original own-robot trajectory planner is available.', 'result': 'Repeat the original approach/down/hold/release stroke for the current card count and return to observed episode home.', 'uncertainty': 'Pose-qualified command cycles are not physical button counts, contact, reward or official success.'}, 'stage_stack_foundation': {'precondition': 'The instructed bottom block and all three block surfaces are currently RGB-grounded; no completed stacking relation is being dismantled.', 'result': 'Restore the mature foundation-relocation stage via original grasp/carry skills and a publicly inferred shared workspace; review the actual released foundation before stacking.', 'uncertainty': 'No fixed layout grid or metric offsets are supplied by the model. Planning and gripper commands do not confirm placement, attachment or official success.'}, 'active_surface_view': {'precondition': 'A selected square-block face is visible in head RGB but reliable multi-view geometry is missing; the selected arm is near its observed home and both public gripper commands are open.', 'result': 'Rotate only the own wrist to acquire a current second view, then accept only consistent RGB triangulation before returning control to the planner.', 'uncertainty': 'The provisional camera-aim reference is not an object position or grasp target; failed multi-view reconstruction remains uncertain and consumes the same robot action budget.'}, 'stack_colour_pair': {'precondition': 'One explicit instruction-supported colour pair and consistent current RGB surface reconstruction; the foundation is never selected as a source.', 'result': 'Select an open arm via original approach-and-grasp cuRobo checks (auto), or respect an explicit side. Compose grasp, visual test lift, closed carry, release, review and observed-home return.', 'uncertainty': 'No metric offsets are guessed by the planner; uncertain geometry, attachment or released support stops this one-pair operation without a success claim.'}, 'return_episode_home': {'precondition': 'The selected public arm must return to its own pose recorded at this episode reset, after visually safe release.', 'result': 'Restore the observed episode-home pose and open the selected gripper without changing the other arm.', 'uncertainty': 'Robot home arrival is not evidence of object placement or official success.'}, 'counted_press': {'precondition': 'A current RGB-grounded hover offset, correct visible button/card association and an unambiguous requested count; no speculative extra presses.', 'result': 'Down, dwell and complete release cycles; robot-pose-qualified cycles are not official button counts.', 'uncertainty': 'Blocked or poorly aligned motion aborts with bounded retreat; do not blindly retry an uncertain count.'}, 'grasp_visible_block': {'precondition': 'The actual instruction concerns a block or cube; its selected colour has current square-face RGB geometry, an open eligible arm and a declared original trajectory planner.', 'result': 'Original reachable approach/descent selection and open/close/settle/test-lift execution with public image-change review; not object attachment or whole-task success.', 'uncertainty': 'Missing or ambiguous RGB geometry forbids a guessed pose. Failed reachability or an ambiguous test lift permits bounded recovery, not a fabricated grasp claim.'}, 'grasp_test_lift': {'precondition': 'A visibly selected coloured target and grounded relative grasp correction; use downward orientation only for a clear vertical approach.', 'result': 'Open, approach, descend, close, settle and small orientation-preserving test lift, followed by mandatory RGB grasp review.', 'uncertainty': 'A public gripper value, closed command or completed lift is not contact or secure grasp.'}, 'stack_place': {'precondition': 'The public instruction explicitly supports the selected colour/support relation; the support is currently visible and the held object is visually checked.', 'result': 'Lift before lateral transfer, preserve orientation, lower, release only after public pose arrival, and retreat.', 'uncertainty': 'The measured robot pose is not object seating; inspect the new RGB support relation before the next layer.'}, 'align_yaw': {'precondition': 'One elongated region of each selected colour is visible; the planner supplies a small visually justified relative robot yaw, not image-to-world calibration.', 'result': 'A bounded yaw correction and before/after current-image axis errors.', 'uncertainty': 'Image error is not a world angle or task score; stop or reverse only after fresh visual evidence.'}, 'pour_segment': {'precondition': 'The instruction requires pouring; current RGB supports a held source above the visible receiving opening and a clear tilt route.', 'result': 'Small orientation increments with dwell observations and optional restoration of the starting orientation.', 'uncertainty': 'Wrist rotation does not prove contents entered the receiver; inspect source, receiver and spills before another segment.'}}
ACTION_PARAMETERS.update(align_scene_view={'side': SIDE}, dual_bar_stage={'side': {'type': 'string', 'enum': ['left']}, 'stage': {'type': 'string', 'enum': ['prepare', 'sweep', 'release_home']}})
ACTION_CONTRACTS.update(align_paired_view={'precondition': 'Both own open wrists are eligible for the original bounded camera-view poses in the current public alignment scene.', 'result': 'Aim both wrist views on a shared command clock; preserve an actual nine-action open dwell and bound further view attempts.', 'uncertainty': 'Public gripper values and view receipts are own command evidence, not contact, physical fingers, object poses or official success.'}, align_scene_view={'precondition': 'The alignment instruction and visible pusher scene require an eligible public wrist view.', 'result': 'Use the existing bounded open-wrist view to qualify current multi-view RGB geometry.', 'uncertainty': 'A view is not a grasp, contact, scene pose or official success.'}, dual_bar_stage={'precondition': 'The current pusher scene or the ordered previous own command recipe qualifies this paired stage.', 'result': 'Execute one original prepare, quasi-static common sweep, or public-joint release/home stage with both arms.', 'uncertainty': 'Both-arm command completion is not physical contact or official success; partly issued stages cannot replay.'})
TUBE_ACTIONS = ('tube_scene_view', 'grasp_tube', 'lift_tube', 'upright_tube', 'seat_tube', 'reseat_tube', 'finish_tube_sequence')
ACTION_LIMITS.update(tube_scene_view=32, grasp_tube=256, lift_tube=128, upright_tube=192, seat_tube=256, reseat_tube=192, finish_tube_sequence=128)
ACTION_MINIMUM_STEPS.update(tube_scene_view=9, grasp_tube=8, lift_tube=2, upright_tube=2, seat_tube=14, reseat_tube=13, finish_tube_sequence=2)
TUBE_TRANSFER_PARAMETERS = {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'slot': {'type': 'integer', 'minimum': 0, 'maximum': 3}, 'target_index': {'type': 'integer', 'minimum': 0, 'maximum': 2}}
ACTION_PARAMETERS.update({name: deepcopy(TUBE_TRANSFER_PARAMETERS) for name in ('grasp_tube', 'lift_tube', 'upright_tube', 'seat_tube')})
ACTION_PARAMETERS.update(tube_scene_view={'side': SIDE, 'cursor': {'type': 'integer', 'minimum': 0, 'maximum': 3}}, reseat_tube={'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'target_index': {'type': 'integer', 'minimum': 0, 'maximum': 2}}, finish_tube_sequence={'side': SIDE, 'stage': {'type': 'string', 'enum': ['home_left', 'home_right', 'settle']}})
ACTION_CONTRACTS.update(tube_scene_view={'precondition': 'A current visible cap or holder and an eligible empty own wrist need an actual public view.', 'result': 'Open both empty home grippers if necessary, then issue the bounded own-wrist view before public qualification.', 'uncertainty': 'No fixed crop, asset-height table, fabricated observation or scene-success claim.'}, grasp_tube={'precondition': 'The current source cap and horizontal body qualify the next source or single explicit fallen-tube recovery.', 'result': 'Select the first original reachable approach and preserve contact offset and dense close clock.', 'uncertainty': 'A completed grasp command is not measured finger contact or a confirmed object grasp.'}, lift_tube={'precondition': 'An actually completed own grasp handoff supplies the current source slot.', 'result': 'Try the nine original lift heights in order, retain slot-specific speed and simultaneous other-gripper close.', 'uncertainty': 'Do not infer a physical lift merely from own pose execution or require a newly visible occluded cap.'}, upright_tube={'precondition': 'An actual own lift precedes route selection from the latest observed own joints.', 'result': 'Rank the original yaw choices at each height, preserve continuous joint branches and execute the first feasible height.', 'uncertainty': 'All previews use public own state and current observed receiving goals, not hidden object poses.'}, seat_tube={'precondition': 'A completed upright handoff supplies its own-joint route and initial public receiving goal.', 'result': 'Preserve cuRobo-FK local descent, insertion hold, release, withdrawal and the original own-joint home.', 'uncertainty': 'A failed withdrawal may retain the original home fallback; command completion is not physical seating.'}, reseat_tube={'precondition': 'The three initial inserts are issued and a currently visible raised cap qualifies one reseat attempt.', 'result': 'Try the original ordered yaw previews, then grasp, close, push, release and home, with explicit abort release/home.', 'uncertainty': 'Never infer an empty hole from missing detections or repeat the declared single attempt.'}, finish_tube_sequence={'precondition': 'The own insertion cursor is complete and the next own reset stage is ordered.', 'result': 'Return both arms to episode home joints, then open both grippers with the mature settle clock.', 'uncertainty': 'Own reset must not depend on a clear scene image; only the official evaluator decides completion.'})
NUMBER_ACTIONS = ('number_scene_view', 'number_home', 'number_empty_approach', 'grasp_number', 'lift_number', 'carry_number', 'place_number', 'verify_number_placement', 'recover_number_grasp', 'relay_number')
PICKUP_ACTIONS = ('pickup_scene_view', 'pickup_pregrasp', 'pickup_contact', 'pickup_close', 'pickup_warmup_lift', 'pickup_full_lift', 'pickup_verify_lift', 'pickup_reset_grasp')
ACTION_LIMITS.update(pickup_scene_view=192, pickup_pregrasp=256, pickup_contact=128, pickup_close=16, pickup_warmup_lift=128, pickup_full_lift=192, pickup_verify_lift=16, pickup_reset_grasp=192)
ACTION_MINIMUM_STEPS.update({name: 16 for name in PICKUP_ACTIONS})
ACTION_PARAMETERS.update({name: {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}} for name in PICKUP_ACTIONS})
ACTION_CONTRACTS.update({name: {'precondition': 'The actual pickup instruction and current public target evidence qualify this bounded mature stage.', 'result': 'Execute original own robot interpolation, sixteen-command replay, public normalized holds and measured own endpoint corrections.', 'uncertainty': 'A command is not object contact or official success; missing current visual geometry or failed lift remains uncertain.'} for name in PICKUP_ACTIONS})
GARMENT_ACTIONS = ('garment_scene_view', 'garment_approach_contact', 'garment_lower_contact', 'garment_close_grasp', 'garment_lift_fold', 'garment_carry_fold', 'garment_settle_fold', 'garment_release_fold', 'garment_clear_wrist', 'garment_recover_contact', 'garment_return_home')
ACTION_LIMITS.update({name: 16 for name in GARMENT_ACTIONS})
ACTION_MINIMUM_STEPS.update({name: 16 for name in GARMENT_ACTIONS})
ACTION_PARAMETERS.update({name: {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}} for name in GARMENT_ACTIONS})
ACTION_CONTRACTS.update({name: {'precondition': 'The current fold instruction, independently measured garment outline and recorded own stage qualify this tool.', 'result': 'Execute one whole original sixteen-command stage with own joint winding, normalized jaws and measured own endpoints.', 'uncertainty': 'Image following is uncertain evidence, not contact; missing landmarks or endpoint mismatch cannot advance a fold.'} for name in GARMENT_ACTIONS})
ACTION_LIMITS.update(number_scene_view=32, number_home=192, number_empty_approach=256, grasp_number=256, lift_number=192, carry_number=400, place_number=400, verify_number_placement=192, recover_number_grasp=256, relay_number=400)
ACTION_MINIMUM_STEPS.update(number_scene_view=2, number_home=2, number_empty_approach=4, grasp_number=14, lift_number=2, carry_number=4, place_number=12, verify_number_placement=2, recover_number_grasp=11, relay_number=32)
NUMBER_PARAMETERS = {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}, 'digit': {'type': 'integer', 'minimum': 0, 'maximum': 9}, 'target_index': {'type': 'integer', 'minimum': 0, 'maximum': 7}}
ACTION_PARAMETERS.update({name: deepcopy(NUMBER_PARAMETERS) for name in NUMBER_ACTIONS if name not in ('number_scene_view', 'number_home')})
ACTION_PARAMETERS.update(number_scene_view={'side': SIDE, 'cursor': {'type': 'integer', 'minimum': 0, 'maximum': 8}, 'view_family': {'type': 'string', 'enum': ['targets', 'numerals']}}, number_home={'side': {'type': 'string', 'enum': ['left', 'right']}, 'purpose': {'type': 'string', 'enum': ['inspect', 'finish']}})
ACTION_PARAMETERS['number_empty_approach']['side'] = {'type': 'string', 'enum': ['left', 'right']}
ACTION_CONTRACTS.update({name: {'precondition': 'Current public numeral/target evidence and the next own-command stage independently qualify this tool.', 'result': 'Execute only the declared bounded mature phase and preserve original sampling, opening holds and own-joint home.', 'uncertainty': 'Missing visual evidence or uncertain execution permits bounded reobservation; progress is not native success.'} for name in NUMBER_ACTIONS})
ACTION_CONTRACTS['number_empty_approach'] = {'precondition': 'A fresh public numeral and open jaws qualify a homed acting arm with its empty observer away from home.', 'result': 'Run only original open2 and checked sampled_free approach; retain exact anchor/progress/route without contact or advancement.', 'uncertainty': 'A whole own receipt authorizes only observer return; new RGB and unchanged acting approach must qualify before original grasp.'}
ACTION_CONTRACTS['relay_number'] = {'precondition': 'A fresh numeral has no qualified single-arm route; current clear support RGB and both own-arm routes qualify one relay.', 'result': 'Replay the mature intermediary transfer, release and own home; only fresh RGB may update the source, and the real destination cursor stays unchanged.', 'uncertainty': 'A predicted receiver approach or support patch is not contact or true collision clearance; a failed relay is not repeated.'}
AUTOMATIC_ARM_SIDE = {'type': 'string', 'enum': ['auto', 'left', 'right']}
PLANAR_T_ACTIONS = ('planar_t_scene_view', 'approach_t_contact', 'descend_t_contact', 'close_t_contact', 'slide_t_contact', 'finish_t_contact', 'settle_t_contact', 'return_t_observer')
ACTION_LIMITS.update(planar_t_scene_view=32, approach_t_contact=256, descend_t_contact=64, close_t_contact=16, slide_t_contact=8, finish_t_contact=256, settle_t_contact=16, return_t_observer=128)
ACTION_MINIMUM_STEPS.update(planar_t_scene_view=2, approach_t_contact=5, descend_t_contact=3, close_t_contact=10, slide_t_contact=3, finish_t_contact=32, settle_t_contact=16, return_t_observer=7)
ACTION_PARAMETERS.update({name: {'side': deepcopy(SIDE)} for name in PLANAR_T_ACTIONS})
ACTION_PARAMETERS['approach_t_contact'] = {'side': deepcopy(AUTOMATIC_ARM_SIDE)}
ACTION_CONTRACTS.update({name: {'precondition': 'Current public T/printed goal stereo evidence and acknowledged own-command progress qualify this bounded contact stage.', 'result': 'Compose original relative approach, three-sample servo, dense gripper ramp, release, retreat and own home using only public APIs.', 'uncertainty': 'Own-URDF commands and attachment prediction are not measured contact, object GT or official success.'} for name in PLANAR_T_ACTIONS})
CUBE_ACTIONS = ('cube_scene_view', 'grasp_measured_cube', 'lift_measured_cube', 'carry_measured_cube', 'release_measured_cube', 'verify_measured_stack', 'reset_cube_grasp', 'finish_cube_stack')
ACTION_LIMITS.update(cube_scene_view=64, grasp_measured_cube=192, lift_measured_cube=192, carry_measured_cube=160, release_measured_cube=192, verify_measured_stack=128, reset_cube_grasp=128, finish_cube_stack=128)
ACTION_MINIMUM_STEPS.update({name: 2 for name in CUBE_ACTIONS})
ACTION_PARAMETERS.update({name: {'side': deepcopy(AUTOMATIC_ARM_SIDE), 'slot': {'type': 'integer', 'minimum': 0, 'maximum': 3}} for name in CUBE_ACTIONS})
for name in ('verify_measured_stack', 'reset_cube_grasp', 'finish_cube_stack'):
    ACTION_PARAMETERS[name]['side'] = {'type': 'string', 'enum': ['left']}
ACTION_CONTRACTS.update({name: {'precondition': 'Current independent free cube models and bounded acknowledged own-command stage qualify this real skill.', 'result': 'Compose original finite grasp alternatives, sampled clearance, closed carry, dense release and paired own home.', 'uncertainty': 'Public normalized gripper response is not finger width or contact; predictions and sequence progress are not native success.'} for name in CUBE_ACTIONS})
VESSEL_ACTIONS = ('vessel_scene_view', 'recover_pour_ball', 'grasp_pour_cup', 'lift_pour_cup', 'relay_pour_cup', 'carry_pour_cup', 'observe_pour_rim', 'tilt_pour_cup', 'upright_pour_cup', 'return_pour_cup', 'home_vessel_arms', 'reset_vessel_grasp', 'finish_pour_sequence')
ACTION_LIMITS.update(vessel_scene_view=96, recover_pour_ball=256, grasp_pour_cup=192, lift_pour_cup=128, relay_pour_cup=320, carry_pour_cup=192, observe_pour_rim=256, tilt_pour_cup=160, upright_pour_cup=128, return_pour_cup=256, home_vessel_arms=128, reset_vessel_grasp=128, finish_pour_sequence=128)
ACTION_MINIMUM_STEPS.update({name: 2 for name in VESSEL_ACTIONS})
ACTION_PARAMETERS.update({name: {'side': deepcopy(AUTOMATIC_ARM_SIDE), 'index': {'type': 'integer', 'minimum': 0, 'maximum': 8}} for name in VESSEL_ACTIONS})
for name in ('home_vessel_arms', 'reset_vessel_grasp', 'finish_pour_sequence'):
    ACTION_PARAMETERS[name]['side'] = {'type': 'string', 'enum': ['left']}
ACTION_CONTRACTS.update({name: {'precondition': 'The public instruction concerns pouring balls into a vase; bounded current image fits and acknowledged own command progress qualify this original stage.', 'result': 'Compose real vessel perception, original timed own Cartesian or cuRobo routes, normalized dense gripper replay, conditional relay, rim pivot, drain and reverse-path return.', 'uncertainty': 'Current visual attachment and support remain hypotheses; true finger width, GT object pose, saved layout and scorer state are never inputs or successful grasp claims.'} for name in VESSEL_ACTIONS})
LIQUID_ACTIONS = ('liquid_scene_view', 'liquid_support_cup', 'liquid_approach_bottle', 'liquid_close_bottle', 'liquid_reset_grasp', 'liquid_lift_bottle', 'liquid_observe_spout', 'liquid_carry_bottle', 'liquid_tilt_bottle', 'liquid_drain', 'liquid_upright', 'liquid_hold_upright')
ACTION_LIMITS.update(liquid_scene_view=48, liquid_support_cup=160, liquid_approach_bottle=192, liquid_close_bottle=80, liquid_reset_grasp=80, liquid_lift_bottle=64, liquid_observe_spout=48, liquid_carry_bottle=80, liquid_tilt_bottle=96, liquid_drain=112, liquid_upright=40, liquid_hold_upright=16)
ACTION_MINIMUM_STEPS.update({name: 1 for name in LIQUID_ACTIONS})
ACTION_MINIMUM_STEPS.update(liquid_scene_view=0, liquid_observe_spout=0)
ACTION_PARAMETERS.update({name: {'side': deepcopy(AUTOMATIC_ARM_SIDE)} for name in LIQUID_ACTIONS})
ACTION_CONTRACTS.update({name: {'precondition': 'A liquid-pouring instruction and this finite stage are qualified by current public RGB, independently measured geometry and acknowledged own command progress.', 'result': 'Run this real composed stage with the original fixed command count, continuous own IK, constant normalized gripper commands and exact pivot samples.', 'uncertainty': 'Current visual attachment remains a hypothesis; no object asset dimensions, private layout, true fingers, contact or scorer are accessible.'} for name in LIQUID_ACTIONS})
RIGID_TRANSFER_ACTIONS = ('transfer_pregrasp', 'transfer_contact', 'transfer_close', 'transfer_lift', 'transfer_carry', 'transfer_align', 'transfer_place', 'transfer_release', 'transfer_retract', 'transfer_verify', 'transfer_reset', 'transfer_home', 'transfer_finish')
RIGID_VIEW_ACTIONS = {'tool_scene_view': 'store_tools_in_toolbox', 'category_scene_view': 'classify_objects_by_language', 'doll_row_scene_view': 'sort_nesting_dolls_by_size', 'bowl_stack_scene_view': 'stack_bowls_random'}
ACTION_LIMITS.update({name: 240 for name in RIGID_TRANSFER_ACTIONS})
ACTION_LIMITS.update({name: 240 for name in RIGID_VIEW_ACTIONS})
ACTION_LIMITS.update(transfer_close=40, transfer_release=40, transfer_verify=2, transfer_reset=8, transfer_finish=8)
ACTION_MINIMUM_STEPS.update({name: 1 for name in (*RIGID_TRANSFER_ACTIONS, *RIGID_VIEW_ACTIONS)})
ACTION_PARAMETERS.update({name: {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}} for name in (*RIGID_TRANSFER_ACTIONS, *RIGID_VIEW_ACTIONS)})
ACTION_CONTRACTS.update({name: {'precondition': 'A semantic transfer goal, current independent RGB references and bounded own stage qualify this tool.', 'result': 'Execute one actual independent stage with the mature own-feedback increment or already-sampled normalized command clock.', 'uncertainty': 'Surface following and placement remain uncertain public evidence, never true contact, fingers, layout or official success.'} for name in (*RIGID_TRANSFER_ACTIONS, *RIGID_VIEW_ACTIONS)})
RIGID_RECIPE_ACTIONS = ('reference_approach', 'reference_contact', 'reference_close', 'reference_lift', 'reference_carry', 'reference_align', 'reference_insert', 'reference_rotate', 'reference_press', 'reference_handover', 'reference_release', 'reference_retract', 'reference_hold', 'reference_reset')
RIGID_RECIPE_VIEW_ACTIONS = {'category_partition_view': 'classify_objects', 'box_packing_view': 'pack_objects_into_box', 'ring_peg_view': 'play_stacking_toy', 'connector_socket_view': 'plug_in_charger', 'coin_bank_view': 'deposit_coin', 'key_handover_view': 'insert_key', 'table_organize_view': 'organize_table', 'pen_receiver_view': 'fill_pen_holder', 'egg_holder_view': 'fill_egg_holder', 'mug_handle_view': 'hang_mugs', 'thread_geometry_view': 'fasten_screws', 'conveyor_memory_view': 'match_and_pick_from_conveyor', 'image_board_view': 'pick_from_conveyor_by_image', 'sorting_demo_view': 'imitate_sorting_sequence'}
RIGID_RECIPE_SHARED_VIEWS = ('toaster_scene_view',)
RECIPE_TOOLS = (*RIGID_RECIPE_ACTIONS, *RIGID_RECIPE_VIEW_ACTIONS, *RIGID_RECIPE_SHARED_VIEWS)
ACTION_LIMITS.update({name: 240 for name in RECIPE_TOOLS})
ACTION_LIMITS.update(reference_close=40, reference_release=40, reference_hold=40, reference_reset=8)
ACTION_MINIMUM_STEPS.update({name: 1 for name in RECIPE_TOOLS})
ACTION_PARAMETERS.update({name: {'side': {'type': 'string', 'enum': ['auto', 'left', 'right']}} for name in RECIPE_TOOLS})
ACTION_CONTRACTS.update({name: {'precondition': 'The current public semantic goal and exact next finite recipe step qualify this independent tool.', 'result': 'Execute and acknowledge only this bounded own-reference movement or already-sampled command stage.', 'uncertainty': 'Observed geometry, issued jaw commands and action completion do not establish true contact, attachment or official success.'} for name in RECIPE_TOOLS})
ACTION_BINDINGS['instrument_rule_boundary'] = 'tasks/play_Xylophone/skills/instrument_rule_boundary/implementation.py'
ACTION_LIMITS['instrument_rule_boundary'] = 640
ACTION_MINIMUM_STEPS['instrument_rule_boundary'] = 1
ACTION_PARAMETERS['instrument_rule_boundary'] = {'side': {'type': 'string', 'enum': ['auto']}, 'stage': {'type': 'string', 'enum': ['pick_and_lift', 'transport', *['strike_' + str(index) for index in range(8)]]}}
ACTION_CONTRACTS['instrument_rule_boundary'] = {'precondition': 'The public instrument instruction and explicit same-episode original rule boundary are qualified.', 'result': 'Execute the complete original pick-and-lift, transport or next strike with one dense-control sampler and public own feedback.', 'uncertainty': 'Ray geometry, shaft assumptions and normalized jaw feedback remain uncertain; neither command completion nor a closed jaw is official success or contact.'}

def action_contract(name):
    return dict(ACTION_CONTRACTS[name], parameters=deepcopy(ACTION_PARAMETERS[name]), budget={'maximum_robot_actions': ACTION_LIMITS[name], 'shared_episode_budget': True, 'minimum_robot_actions': ACTION_MINIMUM_STEPS.get(name, 1)}, statuses=['completed', 'uncertain', 'precondition_unmet', 'budget'], run_interface='run(restricted_public_robot_api, validated_request)')

def validate_action_parameters(request):
    name = request['skill']
    parameters = ACTION_PARAMETERS[name]
    if name == 'number_empty_approach' and (type(request.get('steps')) is not int or not 1 <= request['steps'] <= ACTION_LIMITS[name]):
        raise ExecutionError('The empty approach must remain within its declared shared action allowance')
    if set(request) != set(parameters) | {'skill', 'steps', 'reason'}:
        raise ExecutionError('A task-tool request must contain only its declared parameters')
    for field, schema in parameters.items():
        value = request[field]
        category = schema['type']
        if category == 'string':
            valid = isinstance(value, str) and value in schema['enum']
        elif category == 'boolean':
            valid = type(value) is bool
        elif category in ('integer', 'number'):
            valid = type(value) is int if category == 'integer' else type(value) in (int, float)
            valid = valid and math.isfinite(value)
            if valid:
                valid = value >= schema.get('minimum', -math.inf) and value <= schema.get('maximum', math.inf)
                valid = valid and ('exclusiveMinimum' not in schema or value > schema['exclusiveMinimum'])
        else:
            valid = isinstance(value, list) and len(value) == 3 and all((type(component) in (int, float) and math.isfinite(component) and (schema['items']['minimum'] <= component <= schema['items']['maximum']) for component in value))
        if not valid:
            raise ExecutionError('Invalid declared task-tool parameter: ' + field)
    if 'delta' in request and math.hypot(*request['delta']) > 0.15:
        raise ExecutionError('A task-tool correction exceeds the public relative motion radius')
    if 'axis' in request and (not 0.99 <= math.hypot(*request['axis']) <= 1.01):
        raise ExecutionError('A pouring axis must be a unit public robot rotation axis')
    if 'reference_colour' in request and request['reference_colour'] == request['colour']:
        raise ExecutionError('An alignment needs distinct visible source/reference colours')
    if request['skill'] == 't_transfer_stage':
        side = 'right' if 'right' in request['stage'] else 'left'
        if request['side'] != side:
            raise ExecutionError('A serial T stage cannot select the other physical arm')
    return deepcopy(request)


def action_program(observation, request, home_targets=None, geometry_memory=None, trajectory=None, home_joints=None, robot_bases=None, command_memory=None, issued_action_count=0, workflow_factory=None):
    validate_action_parameters(request)
    module = importlib.import_module(ACTION_BINDINGS[request['skill']][:-3].replace('/', '.'))
    api = PublicRobotAPI(observation, home_targets, geometry_memory, trajectory=trajectory, home_joints=home_joints, robot_bases=robot_bases, command_memory=command_memory, issued_action_count=issued_action_count)
    if workflow_factory is not None:
        if request['skill'] != 'instrument_rule_boundary' or not callable(workflow_factory):
            raise ExecutionError('Only the declared original instrument workflow may retain its explicit own session')
        program = workflow_factory(api, deepcopy(request))
    else:
        program = module.run(api, deepcopy(request))
    if not isinstance(program, PhasedProgram):
        raise ExecutionError('A registered task tool must return a bounded public phase program')
    return program

def validate_action_feedback(request, feedback):
    request = validate_action_parameters(request)
    declared_sides = ACTION_PARAMETERS[request['skill']]['side']['enum']
    if request['skill'] == 'number_empty_approach':
        from skills.shared.planning.observed_number_contract_observed_start.implementation import validate_measurements
        from skills.shared.orchestration.resolve_number_sequence_observed_start.contract import validate_empty_approach_feedback
        basic = {'source_visible_before', 'source_visible_after', 'support_visible_before', 'support_visible_after', 'measurement_ambiguous', 'axis_error_before_radians', 'axis_error_after_radians'}
        extra = {'public_number_anchor', 'public_number_progress', 'public_number_route', 'public_number_inventory', 'command_phases_completed', 'current_source_identity_before_grasp', 'number_phase_notes', 'gripper_signal_semantics'}
        measurements = feedback.get('image_measurements')
        if not isinstance(measurements, dict) or not basic <= set(measurements) or set(measurements) - basic - extra:
            raise ExecutionError('Empty approach feedback contains only its declared public number and empty command evidence')
        plain = deepcopy(feedback)
        plain['image_measurements'] = {key: measurements[key] for key in basic}
        checked = validate_phase_feedback(request, plain)
        try:
            validate_measurements(measurements, checked['status'], checked['actions_executed'])
            validate_empty_approach_feedback(request, feedback)
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError(str(error)) from error
        checked['image_measurements'] = deepcopy(measurements)
    else:
        checked = validate_phase_feedback(request, feedback, automatic_side='auto' in declared_sides)
    minimum = ACTION_MINIMUM_STEPS
    if request['skill'] == 'counted_press':
        expected = 2 + request['count'] * (4 + request['dwell'])
    elif request['skill'] == 'press_card_button':
        expected = 2 + request['count'] * 7
    elif request['skill'] == 'swap_confirm_move':
        expected = 8 + (2 if request['retract_home'] else 0)
    elif request['skill'] == 'swap_mat_move':
        if checked['required_minimum_actions'] not in (23, 27):
            raise ExecutionError('An original swap transfer must retain its short or cross-transfer command schedule')
        return checked
    elif request['skill'] == 'dual_bar_stage':
        reused = checked['image_measurements'].get('open_dwell_reused', False)
        expected = {'prepare': 4 if reused else 13, 'sweep': 2, 'release_home': 14}[request['stage']]
        if reused and any((row['phase'] == 'open_both_pushers' for row in checked['phase_reports'])):
            raise ExecutionError('An already fulfilled opening dwell cannot also be counted as new actions')
    elif request['skill'] == 't_transfer_stage':
        from skills.shared.motion.serial_transfer_contract.implementation import MINIMUM
        expected = MINIMUM[request['stage']]
    elif request['skill'] == 'cover_cup_transfer':
        from skills.shared.orchestration.container_return_contract.implementation import MINIMUM
        expected = MINIMUM[request['stage']]
    elif request['skill'] in ('board_turn_stage', 'board_wait_opponent'):
        from skills.shared.orchestration.turn_based_board_contract.implementation import minimum
        expected = minimum(request.get('stage', 'wait'), request['turn'])
    elif request['skill'] in TUBE_ACTIONS:
        schedules = {'tube_scene_view': (2, 9, 11), 'grasp_tube': (2, 8, 10), 'lift_tube': (2,), 'upright_tube': (2,), 'seat_tube': (2, 14), 'reseat_tube': (2, 13), 'finish_tube_sequence': (2, 3)}
        if checked['required_minimum_actions'] not in schedules[request['skill']]:
            raise ExecutionError('Tube feedback must retain the declared original bounded phase schedule')
        return checked
    elif request['skill'] in NUMBER_ACTIONS:
        schedules = {'number_scene_view': (2, 9, 10), 'number_home': (2,), 'number_empty_approach': (2, 4), 'grasp_number': (2, 14), 'lift_number': (2,), 'carry_number': (2, 4), 'place_number': (2, 12), 'verify_number_placement': (2, 4), 'recover_number_grasp': (2, 11), 'relay_number': (2, 32)}
        if checked['required_minimum_actions'] not in schedules[request['skill']]:
            raise ExecutionError('Numeral feedback must retain its reviewed mature opening, release and own-home schedule')
        return checked
    elif request['skill'] in PICKUP_ACTIONS:
        required = checked['required_minimum_actions']
        if required != 2 and (required % 16 or not 16 <= required <= ACTION_LIMITS[request['skill']]):
            raise ExecutionError('Pickup feedback must retain whole original sixteen-command segments')
        return checked
    elif request['skill'] in GARMENT_ACTIONS:
        if checked['required_minimum_actions'] not in (2, 16):
            raise ExecutionError('A garment tool must retain one whole original sixteen-command segment or an explicit unmet precondition')
        return checked
    elif request['skill'] in RECIPE_TOOLS:
        state = checked['image_measurements']['public_rigid_transfer_state']
        expected = {1}
        if request['skill'] in (*RIGID_RECIPE_VIEW_ACTIONS, *RIGID_RECIPE_SHARED_VIEWS):
            expected = {1, 2, 8} if request['skill'] in ('conveyor_memory_view', 'sorting_demo_view') else {1, 2}
        elif request['skill'] == 'reference_reset':
            expected = {8}
        elif request['skill'] in ('reference_close', 'reference_release', 'reference_hold'):
            clock = checked['image_measurements'].get('normalized_command_clock')
            if clock is None:
                if checked['status'] != 'precondition_unmet' or checked['actions_executed'] != 0:
                    raise ExecutionError('A recipe clock needs its actual prepared command receipt')
            else:
                prepared = clock['prepared_native_commands']
                if set(clock) != {'prepared_native_commands', 'issued_native_commands', 'segment_end_indices', 'command_source', 'sampling_reapplied', 'arm_trajectory_replanned', 'held_joints_source', 'true_fingers_or_contact_read', 'official_success_claimed'} or type(prepared) is not int or (not 1 <= prepared <= 40) or (clock['issued_native_commands'] != checked['actions_executed']) or (clock['segment_end_indices'] != [prepared]) or (clock['command_source'] != 'explicit_already_sampled_own_normalized_commands') or (clock['held_joints_source'] != 'last_own_command_or_initial_public_joints') or any((clock[name] is not False for name in ('sampling_reapplied', 'arm_trajectory_replanned', 'true_fingers_or_contact_read', 'official_success_claimed'))):
                    raise ExecutionError('A recipe clock preserves one bounded already-sampled command schedule')
                expected = {prepared}
        if checked['status'] == 'precondition_unmet' and checked['actions_executed'] == 0:
            expected.add(1)
        if checked['required_minimum_actions'] not in expected:
            raise ExecutionError('Recipe feedback must preserve the actual acknowledged independent stage schedule')
        return checked
    elif request['skill'] in (*RIGID_TRANSFER_ACTIONS, *RIGID_VIEW_ACTIONS):
        state = checked['image_measurements']['public_rigid_transfer_state']
        expected = {1}
        if request['skill'] in RIGID_VIEW_ACTIONS:
            expected = {1, 2}
        elif request['skill'] == 'transfer_home':
            expected = {2}
        elif request['skill'] in ('transfer_verify', 'transfer_reset', 'transfer_finish'):
            expected = {{'transfer_verify': 2, 'transfer_reset': 8, 'transfer_finish': 8}[request['skill']]}
        elif request['skill'] == 'transfer_close' and state['goal'] is not None:
            expected = {state['goal']['close_commands'] + state['goal'].get('close_settle_commands', 0)}
        elif request['skill'] == 'transfer_release' and state['goal'] is not None:
            expected = {(26 if state['goal']['release_strategy'] == 'gradual' else state['goal']['release_commands']) + state['goal'].get('pre_release_hold_commands', 0) + state['goal'].get('release_settle_commands', 0)}
        if checked['status'] == 'precondition_unmet' and checked['actions_executed'] == 0:
            expected.add(1)
        if checked['required_minimum_actions'] not in expected:
            raise ExecutionError('Transfer feedback must preserve its actual reviewed own motion or command schedule')
        return checked
    elif request['skill'] in LIQUID_ACTIONS:
        if not ACTION_MINIMUM_STEPS[request['skill']] <= checked['required_minimum_actions'] <= 320:
            raise ExecutionError('Liquid feedback must retain a bounded original fixed-step schedule or explicit zero-action observation')
        return checked
    elif request['skill'] in VESSEL_ACTIONS:
        if not 2 <= checked['required_minimum_actions'] <= 64:
            raise ExecutionError('Vessel tools require finite original motion and dense-dwell schedules')
        return checked
    elif request['skill'] in CUBE_ACTIONS:
        schedules = {'cube_scene_view': (2, 8, 13), 'grasp_measured_cube': (2, 9), 'lift_measured_cube': (2, 4, 6), 'carry_measured_cube': (2,), 'release_measured_cube': (2, 7), 'verify_measured_stack': (6,), 'reset_cube_grasp': (7,), 'finish_cube_stack': (17,)}
        if checked['required_minimum_actions'] not in schedules[request['skill']]:
            raise ExecutionError('Cube feedback preserves the exact bounded dense dwell and sampled phase schedule')
        return checked
    elif request['skill'] in PLANAR_T_ACTIONS:
        schedules = {'planar_t_scene_view': (2, 4, 9), 'approach_t_contact': (2, 5), 'descend_t_contact': (2, *range(3, 31, 3)), 'close_t_contact': (2, 10), 'slide_t_contact': (2, 3, 4), 'finish_t_contact': (2, 32), 'settle_t_contact': (2, 16), 'return_t_observer': (2, 7)}
        if checked['required_minimum_actions'] not in schedules[request['skill']]:
            raise ExecutionError('Planar contact must preserve its actual relative and sampled phase schedule')
        return checked
    elif request['skill'] == 'pour_segment':
        expected = max(1, math.ceil(abs(request['angle']) / (math.pi / 12))) * (2 + request['dwell'])
        expected += 2 if request['return_upright'] else 0
    else:
        expected = minimum[request['skill']]
    if checked['required_minimum_actions'] != expected:
        raise ExecutionError('Task-tool feedback must match its executable phase schedule')
    return checked
