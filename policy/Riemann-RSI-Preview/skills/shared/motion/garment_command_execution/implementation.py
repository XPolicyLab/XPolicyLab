from copy import deepcopy

import numpy as np

from harness.execution import ExecutionError
from skills.shared.motion.garment_joint_sampling.implementation import GarmentOwnModel
from skills.shared.motion.own_sampled_phase_execution.implementation import OwnSampledPhaseProgram
from skills.shared.motion.public_phase_execution.implementation import pose_phase, source_measurements
from skills.shared.motion.urdf_pose_kinematics.implementation import matrix_pose
from skills.shared.verification.observed_garment_mount_state.implementation import validate


class GarmentProgram(OwnSampledPhaseProgram):
    def __init__(self, api, request, current, controls=None, following=None):
        commands = list(controls or [])
        side = current['route']['arms'][0] if current['route'] else 'left'
        selected = dict(request, side=side)
        model = GarmentOwnModel()
        roots = model.roots(api)
        self.model, self.roots = model, roots
        phases = []
        if len(commands) % 16 or len(commands) > request['steps']:
            raise ValueError('A garment stage must reserve every complete original sixteen-command chunk')
        chunks = [commands[index:index + 16] for index in range(0, len(commands), 16)]
        for index, chunk in enumerate(chunks):
            for command in chunk:
                expected = {arm + suffix for arm in ('left', 'right') for suffix in ('_arm_joint_state', '_ee_joint_state')}
                if not isinstance(command, dict) or set(command) != expected:
                    raise ValueError('A garment sample contains only its own two arms and normalized jaws')
                for arm in ('left', 'right'):
                    joints = np.asarray(command[arm + '_arm_joint_state'], dtype=float)
                    opening = np.asarray(command[arm + '_ee_joint_state'], dtype=float)
                    if (joints.shape != (6,) or not np.isfinite(joints).all()
                            or np.any(joints < model.model.lower) or np.any(joints > model.model.upper)
                            or opening.shape != (1,) or not 0 <= opening[0] <= 1):
                        raise ValueError('A garment sample exceeds its declared own-robot limits')
            endpoint = roots[side] @ model.eef(chunk[-1][side + '_arm_joint_state'])
            phases.append(pose_phase(request['skill'] + '_segment_' + str(index + 1), matrix_pose(endpoint),
                float(chunk[-1][side + '_ee_joint_state'][0]), 16, False))
        if not phases:
            phases = [pose_phase('garment_public_precondition_unmet', api.targets()[side + '_ee_pose'],
                                float(api.targets()[side + '_ee_joint_state'][0]), 2, False)]
        super().__init__(api, selected, phases, precondition=bool(commands), measurements=source_measurements(api))
        self.current, self.controls, self.following = validate(current), chunks, following
        self.reviewed = False
        self.visual = {'status': 'uncertain', 'head_colour_fraction': 0., 'wrist_colour_fraction': 0.,
                       'gripper_measurement_is_contact': False}
        for index, chunk in enumerate(chunks):
            self.own_joint_samples[index] = np.asarray([command[side + '_arm_joint_state'] for command in chunk])

    def snapshot(self):
        return self.controls[self.phase_index][min(self.phase_actions, 15)]

    def sampled_target(self, observation):
        target = self.own_joint_samples.get(self.phase_index)
        if target is None:
            return None
        joints = np.asarray(observation['state'][self.side + '_arm_joint_state'], dtype=float)
        candidate = target[min(self.phase_actions, 15)]
        if self.phase_actions == 0 and np.max(abs(candidate - joints)) > .30:
            self.stop_after_held_action = True
            return joints.copy()
        return candidate.copy()

    def additional_joint_commands(self, unused_observation):
        other = 'right' if self.side == 'left' else 'left'
        return {other + '_arm_joint_state': np.asarray(self.snapshot()[other + '_arm_joint_state'])}

    def additional_gripper_commands(self, unused_observation):
        return {side + '_ee_joint_state': np.asarray(self.snapshot()[side + '_ee_joint_state']) for side in ('left', 'right')}

    def observe(self, observation):
        super().observe(observation)
        if self.finished and self.status == 'completed' and not self.reviewed:
            self.reviewed = True
            if self.endpoint_reached():
                self.review_completion()
            else:
                if self.continue_after_shortfall():
                    # The part's own attempt budget is not spent on a recovery
                    # that the official action budget cannot fund, so the
                    # shortfall becomes this part's outcome and the fold keeps
                    # its remaining stages.
                    self.current.update(phase=self.shortfall_phase(), attempts=0)
                    return
                self.current['attempts'] = min(3, self.current['attempts'] + 1)
                if self.current['route'] is not None:
                    # A carry that stopped short is a bounded tracking shortfall,
                    # not a lost grasp.  The packaged rule re-solves the rest of
                    # the transfer from the current acknowledged pose; aborting
                    # to a view instead spent a whole part retry on a grasp that
                    # was still holding the cloth (r70: 18 cm short, wrist 0.94).
                    self.current['phase'] = ('carry' if self.request['skill'] == 'garment_carry_fold'
                                             else 'recover')
                self.status = 'uncertain'

    def endpoint_reached(self):
        # The packaged rule corrected exactly one pose: the contact pose, to
        # 2.5 mm, right after the cuff lower.  It never gated an approach, a
        # lift, a transfer, a settle or a withdrawal on an endpoint, because
        # those stages legitimately end short when the target sits at the edge
        # of the reachable envelope.  Gating them here turned a sound fold into
        # a recovery loop: r72 reached the second cuff, carried it, then failed
        # settle on an unreachable place pose that inherited the unreachable
        # carry target.
        endpoint = self.controls[-1][-1]
        actual = self.api.own_joint_states()
        if self.request['skill'] == 'garment_lower_contact':
            # Cloth-induced paired-hem contact can settle several centimetres
            # from the geometric endpoint even after the bounded own route
            # executes. Keep cuff lower_contact at the original 1.2 cm bound;
            # only the two-arm body stage uses the approach-stage bound.
            tolerance = .06 if self.current.get('part') == 2 else .012
        elif self.request['skill'] == 'garment_return_home':
            tolerance = .025
        else:
            # A transfer or a settle legitimately ends short when its target is
            # the far silhouette vertex at the edge of the reachable envelope:
            # r72 carried the second cuff 18 cm short with the cloth still
            # pinched.  An absolute bound cannot separate that from a stage the
            # robot never executed, because both move the wrist about 18 cm.  Ask
            # the question the contract actually means instead: a stage that
            # was not asked to move must not move, and a stage that was asked
            # must cover at least half of the commanded travel.
            origin = self.controls[0][0]
            for side in ('left', 'right'):
                wanted = self.roots[side] @ self.model.eef(endpoint[side + '_arm_joint_state'])
                start = self.roots[side] @ self.model.eef(origin[side + '_arm_joint_state'])
                measured = self.roots[side] @ self.model.eef(actual[side])
                commanded = float(np.linalg.norm(wanted[:3, 3] - start[:3, 3]))
                travelled = float(np.linalg.norm(measured[:3, 3] - start[:3, 3]))
                if commanded > .05:
                    # A cross-body cuff transfer covers about half of its
                    # nominal travel before the reachable envelope stops it, so
                    # the bound has to sit below that.  A stage that never moved
                    # still reads zero and is rejected here.
                    if travelled < .35 * commanded:
                        return False
                elif travelled > .05:
                    return False
            return True
        for side in ('left', 'right'):
            target = self.roots[side] @ self.model.eef(endpoint[side + '_arm_joint_state'])
            measured = self.roots[side] @ self.model.eef(actual[side])
            if np.linalg.norm(target[:3, 3] - measured[:3, 3]) > tolerance:
                return False
            if self.request['skill'] == 'garment_return_home' and np.max(abs(np.asarray(endpoint[side + '_arm_joint_state']) - actual[side])) > .12:
                return False
        return True

    # FoldClothesCommon declares step_lim = 500 for this task and every garment
    # stage reserves one whole sixteen-command chunk, so the nominal three-part
    # program already occupies 496 of the 500 official actions.  A regrasp adds
    # six stages and the fold it interrupted still owes five, so a retry can
    # never be followed by the fold it was meant to rescue: every observed
    # retry ended the episode mid-arc with the jaw still closed, and a closed
    # jaw fails the official is_all_gripper_open gate outright.
    OFFICIAL_ACTION_BUDGET = 500
    STAGE_ACTIONS = 16
    REGRASP_STAGES = 6
    FOLD_TAIL_STAGES = 5
    PART_STAGES = 10
    HOME_STAGES = 1

    def recovery_affordable(self):
        """Whether a regrasp can still be followed by the rest of the fold."""
        part = max(0, min(2, int(self.current.get('part') or 0)))
        needed = (self.REGRASP_STAGES + self.FOLD_TAIL_STAGES
                  + (2 - part) * self.PART_STAGES + self.HOME_STAGES)
        return (self.api.issued_action_count() + needed * self.STAGE_ACTIONS
                <= self.OFFICIAL_ACTION_BUDGET)

    def continue_after_shortfall(self):
        """Accept a short stage when no recovery could be funded.

        A contact stage reaches this through the same arithmetic as a fold
        review.  The observed part-one loop spent lower_contact, recover_contact,
        scene_view and approach five times over without ever clearing the 1.2 cm
        contact bound, because RecoverGarment.review_completion restarts the
        part with a fresh attempt budget; twenty stages is four fifths of what
        the rest of the program still needed, so the loop could only end with
        the official action budget exhausted mid-fold.
        """
        return (self.request['skill'] in ('garment_lift_fold', 'garment_carry_fold',
                                          'garment_lower_contact')
                and not self.recovery_affordable())

    def shortfall_phase(self):
        """The phase a short stage continues into instead of recovering.

        A fold that ended short has already made its transfer, so it carries on
        to its settle.  A contact that ended short is still holding nothing, so
        it carries on to its own close and grasps from wherever the pad landed.
        """
        return 'close' if self.request['skill'] == 'garment_lower_contact' else 'carry'

    def review_completion(self):
        if self.following is not None:
            self.current['phase'] = self.following

    def feedback(self, observation, budget=False):
        value = super().feedback(observation, budget)
        if self.status == 'uncertain' and not self.reviewed and self.current['route'] is not None:
            self.current['phase'] = 'recover'
        value['image_measurements'].update(public_garment_state=validate(self.current),
            garment_visual_review=deepcopy(self.visual),
            garment_clock={'segment_width': 16, 'trajectory_resampled': False, 'true_finger_measurements_used': False},
            command_phases_completed=bool(self.actions >= self.minimum_actions and self.actions >= 16
                                          and self.status in ('completed', 'uncertain')))
        return value


def allowed(current, phase):
    return current['phase'] == phase and current['attempts'] < 3 and (current['route'] is not None or phase == 'view')


def unavailable(api, request, current):
    return GarmentProgram(api, request, current)
