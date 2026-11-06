import math
import numpy as np

from skills.shared.motion.garment_joint_sampling.implementation import (GarmentOwnModel, sampled_wrist_path,
    sampled_body_arc, cuff_carry_chunks, body_fold_chunks)
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, allowed, unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import pose_matrix
from skills.shared.perception.current_garment_landmarks.implementation import lifted_surface_feedback
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class CarryGarment(GarmentProgram):
    def review_completion(self):
        self.current['chunk'] += 1
        limit = self.current['route']['body_chunks' if self.current['part'] == 2 else 'carry_chunks']
        if self.current['chunk'] < limit:
            # The packaged rule commands the whole transfer as one cartesian
            # call and verifies the carried cloth once, after the last chunk.
            # Reviewing every intermediate chunk rejected a valid grasp while
            # the wrist camera was still sweeping past its own pad, and each
            # rejection spent a full part retry on an otherwise sound fold.
            return
        if self.current['part'] == 2:
            # The packaged fold_body() verified the paired hem grasp once, right
            # after hem_lift (body_contact_evidence, then recover_lifted_hem on
            # a failed arm), and never re-verified after body_fold: it went
            # straight on to body_settle.  LiftGarment already runs that review
            # here.  Asking lifted_surface_feedback() again at the end of the
            # torso carry asks it about a pad that has already been placed on
            # the cloth, where its own "pad is lifted above the support" guard
            # (pad[2] - height_m < .025) returns no evidence at all -- r91's
            # layout 2 failed there with 0.000 on both cameras, and the zero
            # evidence spent a whole part retry on a sound fold.
            self.current['phase'], self.current['chunk'] = 'lower', 0
            return
        self.visual = lifted_surface_feedback(self.api, self.current['anchor'], self.current['route']['arms'])
        if self.visual['status'] != 'current_surface_follows_own_pad':
            if self.continue_after_shortfall():
                # The transfer this review follows has already been made, so the
                # sleeve is placed even when the wrist camera cannot see the pad
                # any more.  A regrasp the budget cannot fund would leave it in
                # mid-air, so settle and release it instead.
                self.current['phase'], self.current['chunk'] = 'lower', 0
                return
            self.current['phase'], self.status = 'recover', 'uncertain'
            return
        self.current['phase'], self.current['chunk'] = 'lower', 0


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if not allowed(current, 'carry') or current['start_wrists'] is None:
        return unavailable(api, request, current)
    model = GarmentOwnModel()
    # Same planned contact frame the packaged body_arc() anchored on: the torso
    # arc is a fixed 0..pi sweep about that frame, so anchoring it on the settled
    # pose both shifts the apex and lands the fold short of the shoulder margin.
    anchor = current['route']['contact_poses'] if current['part'] == 2 else current['start_wrists']
    starts = {side: pose_matrix(value) for side, value in anchor.items()}
    try:
        if current['part'] == 2:
            displacements = {side: np.asarray(current['route']['destinations'][side])[:2]
                             - np.asarray(current['route']['contacts'][side])[:2] for side in starts}
            if current['chunk'] == 0:
                current['route'] = dict(current['route'],
                    body_chunks=body_fold_chunks(api, model, starts, displacements))
            controls = sampled_body_arc(api, model, starts, displacements, math.pi / 2., math.pi,
                                        current['chunk'], current['route']['body_chunks'])
        else:
            goals = {}
            for side, start in starts.items():
                goal = start.copy()
                goal[:2, 3] += np.asarray(current['route']['destinations'][side])[:2] - np.asarray(current['route']['contacts'][side])[:2]
                goal[2, 3] += .075
                goals[side] = goal
            if current['chunk'] == 0:
                current['route'] = dict(current['route'],
                    carry_chunks=cuff_carry_chunks(api, model, goals))
            for side, goal in goals.items():
                actual = pose_matrix(api.targets()[side + '_ee_pose'])
                goal[:3, 3] = actual[:3, 3] + (goal[:3, 3] - actual[:3, 3]) / (current['route']['carry_chunks'] - current['chunk'])
            controls = sampled_wrist_path(api, model, goals)
    except ValueError:
        current['phase'] = 'recover'
        return unavailable(api, request, current)
    return CarryGarment(api, request, current, controls)
