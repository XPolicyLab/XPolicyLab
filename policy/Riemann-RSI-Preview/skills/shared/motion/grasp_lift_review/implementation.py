import numpy as np

from skills.shared.motion.public_phase_execution.implementation import PhasedProgram


class GraspTestLift(PhasedProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.before_lift = None

    def observe(self, observation):
        self.api.observe(observation)
        if (self.before_lift is None and self.phase_index < len(self.phases)
                and self.phases[self.phase_index]['name'] == 'test_lift_keep_orientation'):
            self.before_lift = (self.api.regions(self.request['colour']), self.api.image().shape,
                                self.api.targets()[self.side + '_ee_pose'][:3].copy())
        super().observe(observation)
        if not self.finished or self.status != 'completed' or self.before_lift is None:
            return
        regions, shape, position = self.before_lift
        current = self.api.regions(self.request['colour'])
        movement = float(np.linalg.norm(self.api.targets()[self.side + '_ee_pose'][:3] - position))
        self.visual_lift_feedback['robot_translation_m'] = movement
        if len(regions) != 1 or len(current) != 1 or movement < 0.015 or shape != self.api.image().shape:
            self.visual_lift_feedback['status'] = 'ambiguous'
            self.status = 'uncertain'
            return
        displacement = float(np.linalg.norm((current[0]['centroid'] - regions[0]['centroid']) / [shape[1], shape[0]]))
        ratio = float(current[0]['area'] / regions[0]['area'])
        follows = displacement >= 0.004 or abs(ratio - 1) >= 0.10
        self.visual_lift_feedback.update(status='visible_change' if follows else 'no_visible_follow',
                                        head_displacement_fraction=displacement, head_area_ratio=ratio)
        if not follows:
            self.status = 'uncertain'
