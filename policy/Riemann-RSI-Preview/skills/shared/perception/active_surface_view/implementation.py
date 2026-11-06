"""A bounded own-wrist camera view change, never a guessed grasp target."""
from copy import deepcopy
from skills.shared.motion.public_phase_execution.implementation import PhasedProgram, source_measurements
from skills.shared.perception.aim_public_wrist_view.implementation import phases as view_phases
from skills.shared.perception.rgb_surface_geometry.implementation import bright_face, scene_geometry


class ActiveSurfaceView(PhasedProgram):
    def __init__(self, api, request):
        face = bright_face(api.image(), request['colour'])
        phases, allowed = view_phases(api, request['side'], face['centroid'] if face is not None else None)
        super().__init__(api, request, phases, precondition=allowed,
                         measurements=source_measurements(api, request['colour']))
        self.public_geometry_update = None
        self.reviewed = False

    def observe(self, observation):
        super().observe(observation)
        if self.reviewed or not self.finished or self.status != 'completed':
            return
        self.reviewed = True
        remembered = self.api.geometry_memory()
        geometry = scene_geometry(self.api, (self.request['colour'],), remembered)
        target = geometry['targets'].get(self.request['colour'])
        if target is None or target['stereo'] is None:
            self.status = 'uncertain'
            self.measurements['measurement_ambiguous'] = True
            return
        self.public_geometry_update = deepcopy(remembered)
        self.public_geometry_update['support_plane_m'] = geometry['support_plane_m']


def run(api, request):
    return ActiveSurfaceView(api, request)
