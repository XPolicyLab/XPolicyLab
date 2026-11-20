from copy import deepcopy
import re

import numpy as np

from harness.runtime.manipulation import COLOURS
from skills.shared.motion.colour_surface_manipulation.implementation import grasp_phases
from skills.shared.motion.grasp_lift_review.implementation import GraspTestLift
from skills.shared.motion.public_phase_execution.implementation import pose_phase, source_measurements
from skills.shared.motion.vertical_grasp_selection.implementation import PublicVerticalGraspSelector
from skills.shared.perception.rgb_surface_geometry.implementation import scene_geometry
from tasks.stack_blocks_by_language.skills.read_colour_order.implementation import instruction_order, stack_roles


class VisibleBlockGrasp(GraspTestLift):
    def __init__(self, api, request):
        instruction = api.instruction()
        side = 'left' if request['side'] == 'auto' else request['side']
        initial = api.targets()[side + '_ee_pose']
        semantic = bool(re.search(r'\b(?:blocks?|cubes?)\b', instruction, re.IGNORECASE))
        try:
            roles = stack_roles(instruction, instruction_order(instruction))
        except RuntimeError:
            roles = {}
        semantic = semantic and roles.get('bottom') != request['colour']
        colours = tuple(dict.fromkeys((request['colour'], *(colour for colour in COLOURS
            if re.search(r'\b' + colour + r'\b', instruction, re.IGNORECASE)))))
        geometry = scene_geometry(api, colours, api.geometry_memory()) if semantic else {'targets': {}}
        surface = geometry['targets'].get(request['colour'])
        available = surface is not None and api.trajectory_available()
        selected, selection_failed = None, False
        self.grasp_selection = None
        if available:
            selector = PublicVerticalGraspSelector(api, request, surface, geometry['support_plane_m'])
            try:
                selected = selector.select()
            except RuntimeError as error:
                if 'No reachable vertical grasp' not in str(error):
                    raise
                available, selection_failed = False, True
                self.grasp_selection = {'status': 'no_reachable_candidate', 'requested_side': request['side'],
                                        'candidate_limit': 8 * len(selector.robots)}
            else:
                side = selected['side']
                initial = api.targets()[side + '_ee_pose']
                self.grasp_selection = dict(selected['event'], status='reachable', requested_side=request['side'])
        if available:
            phases, _ = grasp_phases(initial, surface, selected)
            available = all(np.linalg.norm(phase['pose'][:3] - initial[:3]) <= 0.65 for phase in phases)
        else:
            phase = ('semantic_precondition_unmet' if not semantic else 'grasp_reachability_not_found'
                     if selection_failed else 'public_geometry_or_trajectory_unavailable')
            phases = [pose_phase(phase, initial, 1)]
        super().__init__(api, dict(request, side=side), phases, precondition=available,
                         measurements=source_measurements(api, request['colour']))
        self.geometry = deepcopy(geometry)
        self.public_geometry_update = None
        self.pose_tolerance_m = 0.05
        self.minimum_actions = 22
        self.measurements['measurement_ambiguous'] = semantic and not available
        if available:
            self.preplanned[1] = selected['approach_plan']
        if selection_failed:
            self.status, self.finished, self.failed = 'uncertain', True, True
        if request['steps'] < self.minimum_actions:
            self.status, self.finished = 'precondition_unmet', True

    def observe(self, observation):
        super().observe(observation)
        if self.finished and self.status == 'completed':
            update = self.api.geometry_memory()
            update['support_plane_m'] = self.geometry['support_plane_m']
            self.public_geometry_update = update


def run(api, request):
    return VisibleBlockGrasp(api, request)
