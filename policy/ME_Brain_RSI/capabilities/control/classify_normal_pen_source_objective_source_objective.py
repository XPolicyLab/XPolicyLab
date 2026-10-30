"""Soft-rank source routes using the existing full P1 objective and sag."""
import numpy as np

from coap.motion import TCP_OFFSET, normalize_quat, tcp_to_link
from coap.tasks.classify_r07_periodic_ik import PeriodicPostureKinematics


def rank_source_objective(ctx, proposals):
    """Retain every route and add its actual source objective plus downward sag."""
    if not hasattr(ctx, '_classify_source_periodic_ik'):
        ctx._classify_source_periodic_ik = PeriodicPostureKinematics(ctx.kinematics)
    solver = ctx._classify_source_periodic_ik
    cache = {}
    scale = float(np.linalg.norm(TCP_OFFSET))
    for proposal in proposals:
        arm = proposal['arm']
        point = np.asarray(proposal['point'], float) + [0., 0., proposal['lift_height']]
        quaternion = normalize_quat(np.asarray(proposal['quaternion'], float))
        key = (arm, point.tobytes(), quaternion.tobytes())
        if key not in cache:
            reference = np.asarray(ctx.state[arm + '_arm_joint_state'], float)
            result = solver.solve(arm, tcp_to_link(point, quaternion), current_joints=reference,
                                  position_tol=.002, rotation_tol=.020,
                                  max_starts=8, max_nfev=140)
            endpoint = np.asarray(result['joint_value'], float)
            endpoint_points = np.asarray([solver.raw.forward(arm, value, tcp=True)[:3]
                                          for value in (reference, endpoint)])
            minimum_z = float(endpoint_points[:, 2].min())
            points = np.asarray([solver.raw.forward(arm, reference + fraction * (endpoint - reference),
                                                    tcp=True)[:3]
                                 for fraction in np.linspace(1. / 16., 1., 16)])
            normalized_sag = np.maximum(minimum_z - points[:, 2], 0.) / scale
            costs = normalized_sag ** 2
            cost = float(3. * costs.mean() + 2. * costs.max())
            cache[key] = dict(cost=cost, p1_objective=float(result['objective']),
                              p1_joint_distance=float(result['joint_distance']),
                              tool_scale_m=scale, chord_samples=16,
                              maximum_relative_tcp_sag_m=float(normalized_sag.max() * scale),
                              minimum_chord_tcp_z_m=float(points[:, 2].min()),
                              parent_posture_cost=float(result['posture_cost']),
                              endpoint_position_error_m=float(result['position_error']),
                              endpoint_rotation_error_rad=float(result['rotation_error']))
        evidence = cache[key]
        proposal['source_objective_parent_cost'] = float(proposal['cost'])
        proposal['source_objective_sag_cost'] = evidence['cost']
        proposal['source_objective_p1_cost'] = evidence['p1_objective']
        proposal['source_objective_geometry'] = dict(evidence)
        proposal['cost'] += evidence['p1_objective'] + evidence['cost']
    return min(proposals, key=lambda item: item['cost'])
