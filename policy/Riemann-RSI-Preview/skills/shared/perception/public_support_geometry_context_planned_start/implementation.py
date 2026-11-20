import numpy as np
from harness.execution import ExecutionError
from skills.shared.perception import retained_public_support
from skills.shared.perception.retained_public_support_observed_start.implementation import CURRENT_SOURCE, SOURCE, validate_geometry

from skills.shared.perception.public_support_geometry_context.implementation import possible_own_closure

def validate(value):
    return validate_geometry(value)


def run(api, previous=None, passive_view=False):
    from skills.shared.motion.planned_start_phase_execution.implementation import PublicRobotAPI
    if type(api) is not PublicRobotAPI or type(passive_view) is not bool:
        raise ExecutionError('A support context uses only the restricted public robot API and an explicit view condition')
    prior = validate(previous) if previous is not None else None
    before_contact = passive_view and (not possible_own_closure(api))
    retained = prior if prior is not None and prior['source'] == SOURCE and before_contact else None
    result = retained_public_support.implementation.run(api, retained, before_contact)
    if result['memory'] is not None:
        return result
    observed = result['support']
    if observed['status'] != 'observed' or not observed.get('camera_alignment') or (not api.own_robot_bases_available()):
        return result
    plane = observed['plane']
    record = validate({'height_estimate_m': plane['distance_m'], 'height_uncertainty_m': plane['height_uncertainty_m'], 'camera_alignments': observed['camera_alignment'], 'own_base_poses': {side: pose.tolist() for side, pose in api.own_robot_bases().items()}, 'source': CURRENT_SOURCE})
    return dict(result, memory=record, reason='fresh_public_support_geometry_not_before_contact_permission')
