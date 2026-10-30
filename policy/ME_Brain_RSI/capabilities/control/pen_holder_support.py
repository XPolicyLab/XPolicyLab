"""Stationary cup support with cached initial geometry and audited release."""
from copy import copy, deepcopy
import hashlib
import json
import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from b88r42wq.world import record_measurement
from b88r42wq.manipulation import visual_evidence, verification
from .pen_holder_visibility import (cup_fragments, physical_reverse_validity,
                                    projected_robot_occlusion)
from b88r42wq.robot_motion_geometry import _assets, _solids, _separation


HEXAGON = np.array([[-.0401, -.0011], [-.0201, -.0357], [.0199, -.0357],
                    [.0399, -.0011], [.0199, .0336], [-.0201, .0336]])
NATIVE = np.vstack([np.c_[HEXAGON, np.full(6, -.060)],
                    np.c_[HEXAGON, np.full(6, .060)]])
ASSET = {'id': 'RoboDojo/Rigid/pen_holder/00000',
         'sha256': hashlib.sha256(NATIVE.tobytes()).hexdigest(),
         'source': 'public_hexagonal_outer_geometry', 'solid_is_conservative_outer_hull': True}
OBJECT_ID = 'supported_pen_holder'
LIFT = .020
POSE_ERROR = .005
SUPPORT_ROTATION_ERROR = math.radians(5)
FOLLOWING_ERROR = .008
COARSE_POSITION_ERROR = .012


def _project(points, camera_geometry, shape):
    origin, rotation, fx, fy = camera_geometry
    local = np.einsum('ji,nj->ni', rotation, points-origin)
    if np.any(local[:, 2] >= -1e-6):
        raise ValueError('Public cup model is behind camera')
    return np.c_[shape[1]/2-fx*local[:, 0]/local[:, 2],
                 shape[0]/2+fy*local[:, 1]/local[:, 2]]


def _points(parameters):
    rotation = Rotation.from_rotvec(parameters[3:]).as_matrix()
    return np.einsum('ij,nj->ni', rotation, NATIVE)+parameters[:3]


def _edge_samples(contour, count=96):
    """Resample a closed contour at a fixed total arc-length sample count."""
    vertices = np.asarray(contour, float).reshape(-1, 2)
    edges = np.roll(vertices, -1, axis=0)-vertices
    lengths = np.linalg.norm(edges, axis=1)
    if not len(vertices) or not np.isfinite(lengths).all() or lengths.sum() <= 1e-8:
        raise ValueError('Degenerate cup contour')
    cumulative = np.r_[0., np.cumsum(lengths)]
    distances = np.linspace(0, cumulative[-1], count, endpoint=False)
    indices = np.searchsorted(cumulative, distances, side='right')-1
    fractions = (distances-cumulative[indices])/np.maximum(lengths[indices], 1e-12)
    return vertices[indices]+fractions[:, None]*edges[indices]


def _cup_contour(image, predicted, *, detailed=False, diagnostics=None, robot_occlusion=None):
    """Use visible cup boundaries with explicit image clipping and occlusion."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] > 10) & (hsv[..., 0] < 28) &
            (hsv[..., 1] > 130) & (hsv[..., 2] > 135)).astype(np.uint8)*255
    # Occluder and crop boundaries carry no evidence of the physical cup edge.
    dark = ((hsv[..., 2] < 110) | ((hsv[..., 1] < 70) & (hsv[..., 2] > 110))).astype(np.uint8)
    occlusion = cv2.dilate(dark, np.ones((7, 7), np.uint8))
    observed = cup_fragments(image, mask, predicted, occlusion,
                               sample_contour=_edge_samples, diagnostics=diagnostics,
                               robot_occlusion=robot_occlusion)
    return observed if detailed or observed is None else observed['hull']


def _silhouette_residual(parameters, views, *, return_validity=False, fixed_validity=None):
    """Use visible orange boundaries without completing an occluded polygon."""
    values, validity = [], []
    for view in views:
        geometry, shape, observation = view[1:]
        predicted = cv2.convexHull(np.asarray(_project(_points(parameters), geometry, shape), np.float32))
        observed = np.asarray(observation['contour'], np.float32).reshape(-1, 1, 2)
        for point, clear in zip(observation['samples'], observation['clear_samples']):
            values.append(cv2.pointPolygonTest(predicted, tuple(map(float, point)), True))
            validity.append(bool(clear))
        reverse_points = _edge_samples(predicted)
        reverse_clear = physical_reverse_validity(reverse_points, observation)
        for point, physical in zip(reverse_points, reverse_clear):
            pixel = np.round(point).astype(int)
            in_image = 0 <= pixel[0] < shape[1] and 0 <= pixel[1] < shape[0]
            clear = in_image and bool(physical)
            values.append(cv2.pointPolygonTest(observed, tuple(map(float, point)), True))
            validity.append(bool(clear))
    validity = np.asarray(validity if fixed_validity is None else fixed_validity, bool)
    result = np.asarray(values)*validity
    return (result, validity) if return_validity else result


def _geometric_jacobian(parameters, views, fixed_validity=None):
    """Freeze visibility so pixel-mask switches contribute no pose information."""
    _, valid = _silhouette_residual(parameters, views, return_validity=True)
    if fixed_validity is not None:
        valid = np.asarray(fixed_validity, bool)
    def residual(candidate):
        return _silhouette_residual(candidate, views, fixed_validity=valid)
    return np.column_stack([(residual(parameters+np.eye(6)[index]*1e-5)-
                             residual(parameters-np.eye(6)[index]*1e-5))/2e-5
                            for index in range(6)])


def _pose_covariance(jacobian, sigma):
    """Keep every singular direction; never turn an unknown variance into zero."""
    jacobian = np.asarray(jacobian, float)
    if jacobian.ndim != 2 or jacobian.shape[1] != 6 or not np.isfinite(jacobian).all():
        return None
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    if len(singular) != 6 or singular.min() <= 1e-5:
        return None
    covariance = np.einsum('ki,k,kj->ij',right,(sigma/singular)**2,right)
    if not np.isfinite(covariance).all():
        return None
    return covariance


def _pose_uncertainty(jacobian, sigma):
    covariance = _pose_covariance(jacobian, sigma)
    if covariance is None:
        return None
    return (float(2*np.sqrt(np.trace(covariance[:3,:3]))),
            float(2*np.sqrt(np.trace(covariance[3:,3:]))))


def _upright_uncertainty(parameters, covariance):
    """Propagate complete rotvec covariance to the cup axis, excluding pure yaw."""
    parameters = np.asarray(parameters, float)
    def axis(vector):
        return Rotation.from_rotvec(vector).apply([0., 0., 1.])
    jacobian = np.column_stack([(axis(parameters+np.eye(3)[index]*1e-5)-
                                axis(parameters-np.eye(3)[index]*1e-5))/2e-5
                               for index in range(3)])
    variance = jacobian@np.asarray(covariance)[3:, 3:]@jacobian.T
    return float(2*np.sqrt(max(0., np.trace(variance))))


def _native_sites(pose):
    """Derive rim and base from the complete measured rigid rotation."""
    rotation = Rotation.from_quat(pose['quaternion'],scalar_first=True).as_matrix()
    offset = np.einsum('ij,j->i',rotation,[0.,0.,.060])
    body = np.asarray(pose['position'])
    vertices = np.einsum('ij,nj->ni',rotation,NATIVE)+body
    return {'center':(body+offset).tolist(),'rim_center':(body+offset).tolist(),
            'base_center':(body-offset).tolist(),'base_z':float((body-offset)[2]),
            'lowest_outer_vertex_z':float(vertices[:,2].min()),
            'center_semantics':'native_rim_root','body_yaw':pose['yaw']}


def _fit_pose(anchor, yaw, views, allow_single):
    """Estimate pose using fixed visibility for each local geometric Jacobian."""
    # Freeze current physical-edge labels before optimization, independently of fit score.
    _, fit_validity = _silhouette_residual(np.r_[anchor, 0., 0., yaw], views, return_validity=True)
    def residual(parameters, fixed=None):
        return _silhouette_residual(parameters, views, fixed_validity=fit_validity if fixed is None else fixed)
    def absolute_jacobian(parameters):
        return _geometric_jacobian(parameters, views, fixed_validity=fit_validity)
    window = .060 if allow_single else .025
    lower = np.r_[anchor-window, -.30, -.30, yaw-math.pi/3]
    upper = np.r_[anchor+window, .30, .30, yaw+math.pi/3]
    fits = []
    for angle in (yaw, yaw-math.pi/6, yaw+math.pi/6):
        fit = least_squares(residual, np.r_[anchor, 0., 0., angle], jac=absolute_jacobian,
                            bounds=(lower, upper), loss='soft_l1', f_scale=1., max_nfev=90)
        distances, valid = _silhouette_residual(fit.x, views, return_validity=True)
        rms = float(np.sqrt(np.mean(distances[valid]**2))) if np.any(valid) else float('inf')
        fits.append((rms, fit))
    fits.sort(key=lambda item: item[0]); rms, fit = fits[0]
    best_residual, best_valid = _silhouette_residual(fit.x, views, return_validity=True)
    best_sse = float(np.dot(best_residual, best_residual))
    profile_window = float(chi2.ppf(.99, 6)*max(rms, .35)**2)
    # Compare every discrete fit on the same visible boundary, so dropping
    # difficult pixels cannot manufacture a low-noise competing solution.
    competitors = [item for _, item in fits[1:] if item.success and
                   float(np.sum(_silhouette_residual(item.x, views, fixed_validity=best_valid)**2))-best_sse <= profile_window]
    separated = [item for item in competitors if np.linalg.norm(item.x[:3]-fit.x[:3]) > POSE_ERROR]
    jacobian = absolute_jacobian(fit.x)
    covariance = _pose_covariance(jacobian, max(rms, .35))
    uncertainty = _pose_uncertainty(jacobian, max(rms, .35))
    position_error, rotation_error = uncertainty if uncertainty else (None, None)
    rotation = Rotation.from_rotvec(fit.x[3:])
    axis = rotation.apply([0., 0., 1.])
    tilt = float(np.arccos(np.clip(axis[2], -1, 1)))
    upright_error = _upright_uncertainty(fit.x[3:], covariance) if covariance is not None else None
    ambiguity_error = max((float(np.linalg.norm(item.x[:3]-fit.x[:3])) for item in competitors), default=0.)
    axis_ambiguity = max((float(np.arccos(np.clip(np.dot(axis, Rotation.from_rotvec(item.x[3:]).apply([0., 0., 1.])), -1, 1)))
                          for item in competitors), default=0.)
    orientation_ambiguity = max((min(float((rotation.inv()*Rotation.from_rotvec(item.x[3:])*
                                           Rotation.from_euler('z', index*math.pi/3)).magnitude())
                                     for index in range(6))
                                 for item in competitors), default=0.)
    # The public hexagon has six equivalent yaw classes. Discrete translation
    # and axis ambiguity still enlarge the coarse observation error bounds.
    if position_error is not None:
        position_error += ambiguity_error
        upright_error += axis_ambiguity
        rotation_error += orientation_ambiguity
    rank = int(np.linalg.matrix_rank(jacobian, tol=1e-5))
    distances, valid = _silhouette_residual(fit.x, views, return_validity=True)
    active_counts = [int(np.sum(valid[192*i:192*i+192])) for i in range(len(views))]
    failures = []
    if rank != 6 or covariance is None:
        failures.append('rank_deficient_pose')
    if not fit.success or not np.isfinite(fit.x).all():
        failures.append('optimizer_not_converged')
    if not np.isfinite(rms) or rms >= 3.:
        failures.append('visible_silhouette_residual')
    if any(count < 48 for count in active_counts):
        failures.append('insufficient_visible_edges')
    position_limit = COARSE_POSITION_ERROR if allow_single else POSE_ERROR
    rotation_limit = np.deg2rad(8 if allow_single else 5)
    upright_limit = np.deg2rad(12 if allow_single else 8)
    if position_error is None or position_error > position_limit:
        failures.append('position_uncertainty')
    if rotation_error is None or rotation_error > rotation_limit:
        failures.append('rotation_uncertainty')
    if upright_error is None or tilt+upright_error > upright_limit:
        failures.append('upright_uncertainty')
    if separated and not allow_single:
        failures.append('separated_translation_solutions')
    accepted = not failures
    return dict(accepted=bool(accepted), position=fit.x[:3].tolist(),
                quaternion=rotation.as_quat(scalar_first=True).tolist(),
                yaw=float(rotation.as_euler('xyz')[2]), silhouette_rms_pixels=rms,
                position_error_m=position_error, upright_tilt_rad=tilt,
                rotation_error_rad=rotation_error, upright_error_rad=upright_error,
                competing_translation_solutions=len(separated),
                free_height=True, table_support_prior_used=False, jacobian_rank=rank,
                cameras=[view[0] for view in views],
                measurement_role='coarse_observation' if allow_single else 'strict_grasp',
                grasp_authorized=bool(accepted and not allow_single),
                position_limit_m=position_limit, rotation_limit_rad=rotation_limit,
                upright_limit_rad=upright_limit, translation_ambiguity_error_m=ambiguity_error,
                upright_ambiguity_error_rad=axis_ambiguity,
                rotation_ambiguity_error_rad=orientation_ambiguity,
                profile_sse_window_pixels_squared=profile_window,
                profile_visibility_frozen=True,
                rejected_gates=failures,
                visibility_frozen_for_covariance=True, noise_from_active_edges_only=True,
                views={view[0]: dict(partial_contour=view[3]['partial_contour'],
                                     excluded_sample_count=view[3]['excluded_sample_count'],
                                     predicted_image_clipping=view[3].get('predicted_image_clipping', False),
                                     visible_projected_area_pixels=view[3].get('visible_projected_area_pixels'),
                                     visible_area_ratio=view[3].get('visible_area_ratio'),
                                     active_residual_count=active_counts[index],
                                     associated_fragment_count=view[3].get('associated_fragment_count', 1),
                                     unique_physical_boundary_pixels=view[3].get('unique_physical_boundary_pixels'),
                                     association_robot_gap_pixels=view[3].get('association_robot_gap_pixels', 0),
                                     synthetic_bridge_edge_used=False)
                       for index, view in enumerate(views)},
                reason='' if accepted else 'Current cup observation failed: '+', '.join(failures))


def measure(ctx, anchor, arm, yaw=0., *, allow_single=False):
    """Fit current pixels; a coarse observation never authorizes cup contact."""
    anchor = np.asarray(anchor, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError('Invalid cup association anchor')
    initial = np.r_[anchor, 0., 0., yaw]
    # Fit one immutable public RGB/proprioception snapshot, never mixed frames.
    step = int(ctx.steps)
    state = {name: np.asarray(value).copy() for name, value in ctx.state.items()}
    images = {camera: np.asarray(view['color'], np.uint8)[..., :3].copy()
              for camera, view in ctx.vision.items()}
    stamp = {'measurement_step': step,
             'state_sha256': hashlib.sha256(json.dumps(
                 {name: value.tolist() for name, value in sorted(state.items())},
                 sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
             'images': {name: hashlib.sha256(value.tobytes()).hexdigest()
                        for name, value in sorted(images.items())},
             'source': 'one_current_public_observation'}
    views, skipped, visibility = [], {}, {}
    reason = ''
    for camera in ('cam_head', 'cam_'+arm+'_wrist', 'cam_'+('left' if arm=='right' else 'right')+'_wrist'):
        if camera not in images:
            skipped[camera] = 'Current camera image is unavailable'; continue
        image = images[camera]
        geometry = ctx.perception.camera_geometry(camera, state, image.shape)
        try:
            predicted = _project(_points(initial), geometry, image.shape)
        except ValueError as exc:
            skipped[camera] = str(exc); continue
        visibility[camera] = {}
        robot_mask = projected_robot_occlusion(ctx,camera,state,image)
        observation = _cup_contour(image, predicted, detailed=True, diagnostics=visibility[camera],
                                   robot_occlusion=robot_mask)
        if observation is None:
            skipped[camera] = 'Associated visible cup boundary is absent or insufficient'; continue
        views.append((camera, geometry, image.shape, observation))
    omitted_clipped = []
    complete = [view for view in views if view[3].get('predicted_image_clipping') is False]
    # View selection uses projection visibility before fitting, never fit scores.
    if not allow_single and len(complete) >= 2:
        complete_cameras = {view[0] for view in complete}
        omitted_clipped = [view[0] for view in views if view[0] not in complete_cameras]
        views = complete
    result = None
    if (len(views) >= 2 or (allow_single and len(views) == 1)) and not reason:
        try:
            result = _fit_pose(anchor, yaw, views, allow_single)
            # An inconsistent occluded wrist cannot prevent a bounded high
            # observation from the head. It never authorizes grasping, and the
            # disagreement remains explicit for a later strict stereo retry.
            if allow_single and not result['accepted'] and len(views) > 1 and any(view[0] == 'cam_head' for view in views):
                initial_failure = result
                head_result = _fit_pose(anchor, yaw, [view for view in views if view[0] == 'cam_head'], True)
                if head_result['accepted']:
                    result = head_result
                    result['omitted_inconsistent_views'] = dict(
                        cameras=initial_failure['cameras'][1:],
                        combined_silhouette_rms_pixels=initial_failure['silhouette_rms_pixels'],
                        combined_rejected_gates=initial_failure['rejected_gates'],
                        purpose='High observation only; strict multiview retry required before contact')
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            reason = 'Current cup shape fit failed: '+str(exc)
    if result is None:
        result = dict(accepted=False, reason=reason or 'Current cup shape is unobservable',
                      free_height=True, cameras=[view[0] for view in views],
                      rejected_gates=['insufficient_current_multiview_geometry'],
                      measurement_role='coarse_observation' if allow_single else 'strict_grasp',
                      grasp_authorized=False)
    if int(ctx.steps) != step:
        raise RuntimeError('Cup observation changed while fitting its public snapshot')
    result['measurement_step'] = step
    result['observation_stamp'] = stamp
    result['omitted_clipped_cameras'] = omitted_clipped
    result['view_selection'] = 'Prefer two current complete projections before fitting'
    result['skipped_cameras'] = skipped
    result['visibility_diagnostics'] = visibility
    values = {key: result[key] for key in ('position', 'quaternion') if key in result}
    record_measurement(ctx, OBJECT_ID, 'pen_holder', values,
                       error={key: value for key, value in result.items() if key not in ('position', 'quaternion')},
                       source='rgb_public_cup_visible_free_height_silhouette',
                       cameras=tuple(result['cameras']), accepted=result['accepted'],
                       observable_fields=tuple(values), reason=result['reason'], asset=ASSET)
    return result


def following(before, after, first_tcp, last_tcp):
    """Require a resolved cup translation, rejecting a stationary or slipped cup."""
    if any(pose.get('measurement_role') == 'coarse_observation' for pose in (before, after)):
        return dict(accepted=False, reason='A coarse cup observation cannot verify a support grasp')
    if any(pose.get('grasp_authorized') is False for pose in (before, after)):
        return dict(accepted=False, reason='Current strict cup pose failed: '+
                    '; '.join(pose.get('reason', '') for pose in (before, after)
                              if pose.get('grasp_authorized') is False))
    if not (before.get('accepted') and after.get('accepted') and before.get('free_height') and
            after.get('free_height') and len(before.get('cameras',[]))>=2 and
            len(after.get('cameras',[]))>=2 and after['measurement_step'] > before['measurement_step']):
        return dict(accepted=False, reason='Ordered independent free-height cup observations are required')
    observed = np.asarray(after['position'])-before['position']
    expected = np.asarray(last_tcp)-first_tcp
    tolerance = max(before['position_error_m'], after['position_error_m'])
    accepted = (expected[2] >= .015 and tolerance <= POSE_ERROR and
                observed[2] >= .012 and abs(observed[2]-expected[2]) <= FOLLOWING_ERROR and
                np.linalg.norm(observed-expected) <= FOLLOWING_ERROR and
                after['upright_tilt_rad'] <= np.deg2rad(8))
    return dict(accepted=bool(accepted), observed_motion_m=observed.tolist(),
                before_measurement_step=int(before['measurement_step']),
                after_measurement_step=int(after['measurement_step']),
                tool_motion_m=expected.tolist(), maximum_translation_error_m=FOLLOWING_ERROR,
                stationary_cup_rejected=observed[2] < .012,
                reason='' if accepted else 'RGB cup motion does not resolve upright following of the support hand')


def _simulate(preview, action):
    state = dict(preview.state)
    for side in ('left', 'right'):
        state[side+'_arm_joint_state'] = np.asarray(action[side+'_arm_joint_state'])
        state[side+'_ee_joint_state'] = np.asarray(action[side+'_ee_joint_state'])
        state[side+'_ee_pose'] = preview.kinematics.forward(side, state[side+'_arm_joint_state'])
    preview.update_obs({'state': state, 'vision': preview.vision})


def _cup_envelope(cup, motion_height=0., motion_vector=None, *, include_stationary=True):
    """Enclose registered pose error and the movement of this exact phase."""
    quaternion = np.asarray(cup['quaternion'],float)
    rotation = Rotation.from_quat(quaternion,scalar_first=True).as_matrix()
    vertices = np.einsum('ij,nj->ni',rotation,NATIVE)+cup['position']
    angle_error = float(cup.get('rotation_error_rad',float('nan')))
    if not np.isfinite(angle_error) or angle_error < 0:
        raise ValueError('Unknown cup rotation error cannot authorize obstacle geometry')
    radius = float(np.linalg.norm(NATIVE,axis=1).max())
    position_error = float(cup.get('position_error_m', float('nan')))
    if not np.isfinite(position_error) or position_error < 0:
        raise ValueError('Unknown cup position error cannot authorize obstacle geometry')
    expansion = max(POSE_ERROR, position_error)+2*radius*math.sin(min(angle_error,math.pi)/2)
    offsets = np.array([[x,y,z] for x in (-expansion,expansion)
                        for y in (-expansion,expansion) for z in (-expansion,expansion)])
    shift = np.array([0., 0., float(motion_height)]) if motion_vector is None else np.asarray(motion_vector)
    swept = np.r_[vertices, vertices+shift] if include_stationary else vertices+shift
    expanded = (swept[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    return (expanded.min(0),expanded.max(0),ConvexHull(expanded).equations,expanded),expansion


def _joint_chord(first, last, maximum_step=.025):
    """Sample emitted command chords; this is a bounded sampling audit."""
    first,last = np.asarray(first,float),np.asarray(last,float)
    if first.shape!=(6,) or last.shape!=(6,) or not np.isfinite(np.r_[first,last]).all():
        raise ValueError('Nonfinite cup audit command chord')
    count=max(1,int(np.ceil(np.max(np.abs(last-first))/maximum_step)))
    return [first+(last-first)*fraction for fraction in np.linspace(0.,1.,count+1)]


def _audit(ctx, factory, arm, cup, action_limit=240, *, allow_finger_contact=True,
           cup_motion_height=0., carried_evidence=None, stationary_table_contact=False):
    """Audit real yields and command chords; never synthesize future RGB."""
    from b88r42wq.debug_thresholds import disabled, audit_bypass
    if disabled(ctx):
        result = audit_bypass(ctx, 'pen_cup_obstacle_clearance')
        result.update(actions=0, registered_cup_geometry_verified=False)
        return result
    stationary = (stationary_table_contact and
                  cup.get('geometry_model') == 'initial_table_supported_rgb' and
                  cup.get('table_support_prior_used') is True)
    if allow_finger_contact and not stationary and (cup.get("grasp_authorized") is False or
                                                    cup.get("measurement_role") == "coarse_observation"):
        raise ValueError("Coarse cup observation cannot authorize a finger-contact exception")
    carried = carried_evidence is not None
    if carried:
        after_step = carried_evidence.get('after_measurement_step')
        before_step = carried_evidence.get('before_measurement_step')
        if (carried_evidence.get('accepted') is not True or cup.get('accepted') is not True
                or after_step != int(ctx.steps) or cup.get('measurement_step') != int(ctx.steps)
                or type(before_step) is not int or type(after_step) is not int
                or before_step >= after_step or len(cup.get('cameras', ())) < 2
                or cup_motion_height >= 0):
            raise ValueError('Carried cup geometry requires the current verified independent lift witness')
    preview = copy(ctx)
    preview.obs = {'state': deepcopy(ctx.state), 'vision': ctx.vision}
    preview.memory = deepcopy(ctx.memory)
    preview.joint_targets = deepcopy(ctx.joint_targets)
    preview.gripper_targets = dict(ctx.gripper_targets)
    assets, _, _ = _assets()
    cup_solid,expansion = _cup_envelope(cup)
    from b88r42wq.motion import link_to_tcp
    initial_tool = link_to_tcp(preview.kinematics.forward(arm, preview.joint_targets[arm]))
    previous = {side:np.asarray(ctx.joint_targets[side],float).copy() for side in ('left','right')}
    count,samples = 0,0
    minimum = float('inf')
    iterator=iter(factory(preview))
    while True:
        try:
            action=next(iterator)
        except StopIteration:
            break
        except (RuntimeError,ValueError) as exc:
            exc.motion_diagnostics=deepcopy(getattr(preview,'motion_plan',{}))
            exc.support_segment=deepcopy(preview.memory.get('_pen_support_segment',{}))
            raise
        count += 1
        if count > action_limit:
            raise RuntimeError('Cup support phase exceeds its audited action budget')
        for side in ('left', 'right'):
            target=np.asarray(action[side+'_arm_joint_state'],float)
            for joints in _joint_chord(previous[side],target):
                samples += 1
                if cup_motion_height:
                    active_joints = joints if side == arm else action[arm+'_arm_joint_state']
                    current_tool = link_to_tcp(preview.kinematics.forward(arm, active_joints))
                    delta = current_tool-initial_tool
                    # Bound only movement reached at this robot sample, never future cup poses.
                    delta[2] = np.clip(delta[2], min(0., cup_motion_height), max(0., cup_motion_height))
                    cup_solid,expansion = _cup_envelope(cup, motion_vector=delta,
                                                       include_stationary=not carried)
                solids = _solids(preview.kinematics,side,joints,assets)
                for name,solid in solids.items():
                    # Only active fingers may contact this registered target.
                    if allow_finger_contact and side == arm and name in ('link7','link8'):
                        continue
                    evidence = _separation(solid,cup_solid,.001)
                    bound=evidence['distance_lower_bound_m']
                    if bound is None or bound < .001:
                        error = RuntimeError('Registered cup clearance is not certified: '+side+'/'+name)
                        error.support_segment=deepcopy(preview.memory.get('_pen_support_segment',{}))
                        error.cup_collision_diagnostics={'arm':side,'mesh':name,**evidence}
                        raise error
                    minimum=min(minimum,float(bound))
            previous[side]=target.copy()
        _simulate(preview,action)
    return {'accepted':True,'actions':count,'maximum_actions':action_limit,
            'robot_core_gates_unchanged':True,'target_finger_contact_exception_only':([arm+'/link7',arm+'/link8'] if allow_finger_contact else []),
            'target_outer_hull_is_conservative':True,'unknown_opening_interval':[0,1],
            'cup_position_error_box_m':max(POSE_ERROR,float(cup['position_error_m'])),'cup_rotation_error_rad':cup['rotation_error_rad'],
            'cup_total_error_box_m':expansion,'target_minimum_distance_lower_bound_m':minimum,
            'cup_phase_motion_height_m': float(cup_motion_height),
            'cup_motion_bound_is_time_aligned': True,
            'carried_cup_geometry_after_verified_lift': carried,
            'stationary_cup_hypothesis_included': not carried,
            'command_chord_samples':samples,'maximum_joint_sample_gap_rad':.025,
            'limits':'Registered cup only; bounded chord samples are not a continuous collision proof or all-scene obstacle audit'}


def _failure(ctx, arm, report, reason):
    report.update(verified=False, reason=reason, support_arm=arm, measurement_step=ctx.steps)
    ctx.memory['pen_holder_support'] = deepcopy(report)
    evidence = visual_evidence(ctx, OBJECT_ID, 'retention', accepted=False,
                               source='rgb_cup_support_motion', cameras=('cam_head', 'cam_'+arm+'_wrist'),
                               metrics=report, reason=reason)
    verification(ctx, ctx.stage, OBJECT_ID, evidence, 'retention')
    raise RuntimeError('Pen holder support unverified: '+reason)


def _probe(ctx, arm, report):
    anchor = np.asarray(report['body_center'])
    before = measure(ctx, anchor, arm, report['packing_yaw'])
    if not before['accepted']:
        _failure(ctx, arm, report, before['reason'])
    first = ctx.tcp(arm)[:3].copy()
    target = first+[0, 0, LIFT]
    quaternion = np.asarray(report['grasp_quaternion'])
    def program(plan):
        yield from plan.move(arm, target, quaternion, 0., 8)
        yield from plan.hold(3)
    ctx.stage = 'pen_holder_support_lift'
    report['lift_preflight'] = _audit(ctx, program, arm, before, cup_motion_height=LIFT)
    yield from program(ctx)
    after = measure(ctx, anchor+[0, 0, LIFT], arm, report['packing_yaw'])
    metrics = following(before, after, first, ctx.tcp(arm)[:3])
    evidence = visual_evidence(ctx, OBJECT_ID, 'retention', accepted=metrics['accepted'],
                               source='rgb_stereo_cup_resolved_vertical_motion',
                               cameras=('cam_head', 'cam_'+arm+'_wrist'), metrics=metrics, reason=metrics['reason'])
    outcome = verification(ctx, ctx.stage, OBJECT_ID, evidence, 'retention')
    report['lift_evidence'] = metrics
    if not outcome.verified:
        _failure(ctx, arm, report, metrics['reason'])
    def lower(plan):
        yield from plan.move(arm, first, quaternion, 0., 8)
        yield from plan.hold(3)
    ctx.stage = 'pen_holder_support_lower_closed'
    report['lower_preflight'] = _audit(ctx, lower, arm, after, cup_motion_height=-LIFT,
                                      carried_evidence=metrics)
    yield from lower(ctx)
    final = measure(ctx, before['position'], arm, report['packing_yaw'])
    if not final['accepted'] or np.linalg.norm(np.asarray(final['position'])-before['position']) > POSE_ERROR:
        _failure(ctx, arm, report, 'Cup did not return upright to the measured support site')
    report.update(_native_sites(final))
    report.update(verified=True, body_center=final['position'], packing_yaw=final['yaw'],
                  measurement_step=ctx.steps, current_pose=final, body_center_reference=final['position'], support_tcp_reference=ctx.tcp(arm).tolist(),
                  support_quaternion_reference=np.asarray(ctx.state[arm+'_ee_pose'])[3:].tolist(),
                  limits='A short-lift motion witness; not a measured force or physical opening')
    ctx.memory['pen_holder_support'] = deepcopy(report)
    return report


def _strict_support_record(ctx, pose, reference=None):
    """Adapt exact current RGB results without granting coarse contact authority."""
    current = deepcopy(pose)
    gates = []
    if not current.get('grasp_authorized', False):
        gates.append('strict_grasp_not_authorized')
    if current.get('measurement_step') != int(ctx.steps):
        gates.append('observation_not_current')
    for key, limit in (('position_error_m', POSE_ERROR), ('rotation_error_rad', SUPPORT_ROTATION_ERROR)):
        value = current.get(key)
        if value is None or not np.isfinite(value) or value > limit or value < 0:
            gates.append(key)
    if reference is not None and 'position' in current:
        displacement = float(np.linalg.norm(np.asarray(current['position'])-reference))
        current['precontact_displacement_m'] = displacement
        current['maximum_precontact_displacement_m'] = POSE_ERROR
        if displacement > POSE_ERROR:
            gates.append('precontact_displacement')
    accepted = bool(current.get('accepted') and not gates)
    reason = current.get('reason', '')
    if gates:
        reason = (reason+'; ' if reason else '')+'Strict support gates rejected: '+', '.join(gates)
    current['shared_grasp_rejected_gates'] = gates
    values = {key: current[key] for key in ('position', 'quaternion') if key in current}
    return record_measurement(ctx, OBJECT_ID, 'pen_holder', values,
        error={key: value for key, value in current.items() if key not in ('position', 'quaternion')},
        source='rgb_public_cup_visible_free_height_silhouette', cameras=tuple(current.get('cameras', ())),
        accepted=accepted, observable_fields=tuple(values), reason=reason, asset=ASSET,
        role='strict_grasp', identity={'method': 'current_public_cup_geometry_association',
                                    'stable_object_id': OBJECT_ID},
        relations={'grasp_authorized': accepted, 'retention_verified': False})


def _support_record_pose(record):
    """Recover task geometry while keeping the shared record's acceptance."""
    pose = dict(deepcopy(record.error), **deepcopy(record.values))
    pose.update(accepted=record.accepted, reason=record.reason,
                measurement_step=record.measurement_step, measurement_role=record.role)
    return pose


def _support_lift_actions(plan, arm, first_tcp, quaternion):
    yield from plan.move(arm, np.asarray(first_tcp)+[0., 0., LIFT], quaternion, 0., 8)
    yield from plan.hold(3)


def _support_lower_verified(ctx, arm, report, before, after, first_tcp, action_limit):
    """Keep the old lift witness timestamp and confirm the fresh returned pose."""
    from b88r42wq.manipulation import bounded_actions
    quaternion = np.asarray(report['grasp_quaternion'])
    def lower(plan):
        yield from plan.move(arm, first_tcp, quaternion, 0., 8)
        yield from plan.hold(3)
    ctx.stage = 'pen_holder_support_lower_closed'
    report['lower_preflight'] = _audit(ctx, lower, arm, after, cup_motion_height=-LIFT,
                                      carried_evidence=report['lift_evidence'])
    remaining = max(0, action_limit-int(ctx.steps))
    if report['lower_preflight']['actions'] > remaining:
        _failure(ctx, arm, report, 'Audited lower path exceeds the acquisition action budget')
    lifted_tcp = ctx.tcp(arm)[:3].copy()
    yield from bounded_actions(ctx, lower(ctx), action_limit)
    final = measure(ctx, before['position'], arm, report['packing_yaw'])
    if not final['accepted']:
        # Use a carried-pose bound only to obtain a new view, never as fresh RGB.
        predicted = deepcopy(after)
        predicted['position'] = (np.asarray(after['position'])+ctx.tcp(arm)[:3]-lifted_tcp).tolist()
        predicted['measurement_role'] = 'carried_pose_prediction_for_viewpoint'
        report['postlower_observation_retry'] = {
            'first_registration': deepcopy(final), 'start_step': int(ctx.steps),
            'support_hand_moves': False, 'predicted_pose_is_not_measurement': True}
        yield from _prepare_observer(ctx, arm, predicted, report, action_limit,
            require_strict_view=True, allow_held_contact=True)
        final = measure(ctx, before['position'], arm, report['packing_yaw'])
        report['postlower_observation_retry'].update(
            final_registration=deepcopy(final), end_step=int(ctx.steps))
    report['postlower_registration'] = deepcopy(final)
    record = _strict_support_record(ctx, final, np.asarray(before['position']))
    if not record.accepted:
        _failure(ctx, arm, report, 'Cup did not return upright to the measured support site: '+record.reason)
    evidence = visual_evidence(ctx, OBJECT_ID, 'geometry', accepted=True,
        source='rgb_current_cup_returned_after_verified_lift', cameras=tuple(final['cameras']),
        metrics={'current_pose': final, 'prior_retention_measurement_step': report['lift_evidence_measurement_step'],
                 'new_attachment_proof': False, 'maximum_return_displacement_m': POSE_ERROR})
    outcome = verification(ctx, ctx.stage, OBJECT_ID, evidence, 'geometry')
    report['postlower_geometry_outcome'] = outcome.as_dict()
    report.update(_native_sites(final))
    report.update(verified=True, body_center=final['position'], packing_yaw=final['yaw'],
        measurement_step=ctx.steps, current_pose=final, body_center_reference=final['position'],
        support_tcp_reference=ctx.tcp(arm).tolist(),
        support_quaternion_reference=np.asarray(ctx.state[arm+'_ee_pose'])[3:].tolist(),
        limits='A prior resolved RGB lift and a current returned pose; not force or physical opening')
    ctx.memory['pen_holder_support'] = deepcopy(report)
    return report


def _grasp_candidates(arm, cup):
    """Align the closing axis with opposite native hex faces, not arbitrary yaw."""
    body = np.asarray(cup['position'], float)
    rotation = Rotation.from_quat(cup['quaternion'], scalar_first=True).as_matrix()
    desired = math.pi/3 if arm == 'right' else 2*math.pi/3
    normals = ConvexHull(HEXAGON).equations[:, :2]
    faces = []
    for normal in normals:
        local_yaw = math.atan2(normal[1], normal[0])-math.pi/2
        flat = Rotation.from_euler('z', local_yaw).as_matrix()
        tool_rotation = rotation@flat
        world_yaw = math.atan2(tool_rotation[1,0], tool_rotation[0,0])
        rank = abs(math.atan2(math.sin(world_yaw-desired), math.cos(world_yaw-desired)))
        faces.append((rank, local_yaw, normal, tool_rotation))
    faces.sort(key=lambda item:item[0])
    center_xy = HEXAGON.mean(0)
    candidates = []
    # Pinching a rim wall avoids requiring the open jaw to surround the full cup width.
    for _, local_yaw, normal, upright in faces:
        face_distance = float(np.max((HEXAGON-center_xy)@normal))
        local = np.r_[center_xy+normal*face_distance, .06657]
        approach = -rotation[:, 2]
        closing = rotation@np.r_[normal, 0.]
        tool_rotation = np.column_stack((approach, closing, np.cross(approach, closing)))
        grasp = body+rotation@local
        quaternion = Rotation.from_matrix(tool_rotation).as_quat(scalar_first=True)
        candidates.append(dict(grasp=grasp.tolist(), outside=(grasp+[0.,0.,.045]).tolist(),
            high=(grasp+[0.,0.,.090]).tolist(), quaternion=quaternion.tolist(),
            native_face_normal=normal.tolist(), local_tool_yaw_rad=local_yaw,
            native_grasp_height_m=.06657, native_tool_pitch_rad=math.pi/2,
            closing_axis_face_alignment=True, entry_kind='top_rim_wall_pinch',
            world_tool_yaw_rad=local_yaw))
    for height, pitch in ((.045, math.pi/12), (.045, math.pi/6),
                          (.045, 0.), (.025, math.pi/12)):
        for _, local_yaw, normal, upright in faces:
            tool_rotation = upright@Rotation.from_euler('y',pitch).as_matrix()
            grasp = body+rotation@np.r_[center_xy,height]
            inward = tool_rotation[:,0]
            outside = grasp-inward*.100
            high = outside+[0.,0.,.045]
            quaternion = Rotation.from_matrix(tool_rotation).as_quat(scalar_first=True)
            candidates.append(dict(grasp=grasp.tolist(), outside=outside.tolist(), high=high.tolist(),
                                   quaternion=quaternion.tolist(), native_face_normal=normal.tolist(),
                                   local_tool_yaw_rad=local_yaw, native_grasp_height_m=height,
                                   native_tool_pitch_rad=pitch, closing_axis_face_alignment=True,
                                   world_tool_yaw_rad=math.atan2(inward[1],inward[0])))
    return candidates

def _grasp_view_coverage(ctx, arm, cup, candidate):
    """Rank nominal contact optics without claiming a future RGB observation."""
    from b88r42wq.motion import tcp_to_link
    camera = 'cam_'+arm+'_wrist'
    if camera not in ctx.vision:
        return dict(visible_fraction=0., predicted_image_clipping=True)
    shape = np.asarray(ctx.vision[camera]['color']).shape
    state = dict(ctx.state)
    state[arm+'_ee_pose'] = tcp_to_link(candidate['grasp'], candidate['quaternion'])
    geometry = ctx.perception.camera_geometry(camera, state, shape)
    parameters = np.r_[cup['position'], Rotation.from_quat(
        cup['quaternion'], scalar_first=True).as_rotvec()]
    try:
        pixels = _project(_points(parameters), geometry, shape)
    except ValueError:
        return dict(visible_fraction=0., predicted_image_clipping=True)
    projected = cv2.convexHull(np.asarray(pixels, np.float32))
    screen = np.array([[2., 2.], [shape[1]-2., 2.],
                       [shape[1]-2., shape[0]-2.], [2., shape[0]-2.]], np.float32)
    visible_area, _ = cv2.intersectConvexConvex(projected, screen)
    area = float(cv2.contourArea(projected))
    fraction = float(visible_area/max(area, 1e-9))
    return dict(visible_fraction=fraction, predicted_image_clipping=fraction < .999,
                future_rgb_observation=False)


def _observer_candidates(ctx, support_arm, cup):
    """Aim the free wrist at the cup from its own side with an unclipped model."""
    observer = 'left' if support_arm == 'right' else 'right'
    camera = 'cam_'+observer+'_wrist'
    if camera not in ctx.vision:
        return []
    shape = np.asarray(ctx.vision[camera]['color']).shape
    body = np.asarray(cup['position'], float)
    base = ctx.kinematics.base_transforms[observer][:3, 3]
    direction = base[:2]-body[:2]
    direction /= max(np.linalg.norm(direction), 1e-9)
    local_state = dict(ctx.state)
    local_state[observer+'_ee_pose'] = np.array([0., 0., 0., 1., 0., 0., 0.])
    mount, mount_rotation, _, _ = ctx.perception.camera_geometry(camera, local_state, shape)
    solid, _ = _cup_envelope(cup)
    candidates = []
    for distance, height in ((.24, .20), (.30, .20), (.24, .26), (.30, .26)):
        origin = body+np.r_[direction*distance, height]
        backward = origin-body
        backward /= np.linalg.norm(backward)
        horizontal = np.cross([0., 0., 1.], backward)
        horizontal /= np.linalg.norm(horizontal)
        upward = np.cross(backward, horizontal)
        camera_rotation = np.column_stack((horizontal, upward, backward))
        tool_rotation = camera_rotation@mount_rotation.T
        link_position = origin-tool_rotation@mount
        quaternion = Rotation.from_matrix(tool_rotation).as_quat(scalar_first=True)
        state = dict(ctx.state)
        state[observer+'_ee_pose'] = np.r_[link_position, quaternion]
        geometry = ctx.perception.camera_geometry(camera, state, shape)
        pixels = _project(solid[3], geometry, shape)
        margin = float(np.min(np.r_[pixels[:, 0]-8., shape[1]-8.-pixels[:, 0],
                                    pixels[:, 1]-8., shape[0]-8.-pixels[:, 1]]))
        if margin < 0:
            continue
        candidates.append(dict(arm=observer, camera=camera, link_position=link_position.tolist(),
            quaternion=quaternion.tolist(), camera_origin=origin.tolist(),
            projected_image_margin_pixels=margin, complete_uncertainty_envelope_in_frame=True,
            nominal_view_distance_m=float(np.linalg.norm(origin-body))))
    return candidates


def _prepare_observer(ctx, support_arm, cup, report, action_limit, *, require_strict_view=False,
                      allow_held_contact=False):
    """Move only the free hand along a cup-audited path, then receive real pixels."""
    from b88r42wq.manipulation import bounded_actions
    rejected = []
    for candidate in _observer_candidates(ctx, support_arm, cup):
        def observe(plan):
            arm = candidate['arm']
            plan.memory['_pen_support_segment'] = dict(name='independent_wrist_observation',
                                                       candidate=deepcopy(candidate))
            yield from plan.move(arm, candidate['link_position'], candidate['quaternion'],
                                 plan.gripper_targets[arm], 22, tcp=False)
            yield from plan.hold(3)
        try:
            audit = _audit(ctx, observe, support_arm, cup, action_limit=200,
                           allow_finger_contact=allow_held_contact)
            if audit['actions'] > action_limit-int(ctx.steps):
                raise RuntimeError('Independent observation exceeds the acquisition budget')
        except (RuntimeError, ValueError) as exc:
            rejected.append(_path_rejection(candidate, exc))
            continue
        report['independent_observation'] = dict(candidate=deepcopy(candidate),
            preflight=audit, rejected_candidates=rejected, uses_predicted_images=False)
        ctx.memory['pen_holder_support'] = deepcopy(report)
        ctx.stage = 'pen_holder_support_independent_observation'
        yield from bounded_actions(ctx, observe(ctx), action_limit)
        if require_strict_view:
            current = measure(ctx, cup['position'], support_arm, cup['yaw'])
            report['independent_observation']['registration'] = deepcopy(current)
            ctx.memory['pen_holder_support'] = deepcopy(report)
            if (not current['accepted'] or not current.get('grasp_authorized') or
                    len(current.get('cameras', ())) < 2):
                rejected.append(dict(candidate=deepcopy(candidate), phase='current_rgb_observation',
                    registration=deepcopy(current),
                    reason=current['reason'] or 'Independent view is not a strict current stereo pose'))
                continue
        return True
    report['independent_observation'] = dict(rejected_candidates=rejected,
        candidate=None, uses_predicted_images=False, reason='No audited independent wrist view')
    ctx.memory['pen_holder_support'] = deepcopy(report)
    return False


def _support_move(plan, arm, candidate, name, position, duration):
    """Identify the failed segment while retaining the original motion gates."""
    plan.memory['_pen_support_segment'] = dict(name=name, target_tcp=list(position),
                                             candidate=deepcopy(candidate))
    yield from plan.move(arm, position, candidate['quaternion'], 1., duration)

def _path_rejection(candidate, exc):
    return dict(candidate=deepcopy(candidate), reason=str(exc),
                segment=getattr(exc,'support_segment',{}),
                motion_diagnostics=getattr(exc,'motion_diagnostics',{}),
                cup_collision_diagnostics=getattr(exc,'cup_collision_diagnostics',{}))


def acquire(ctx, holder, center):
    """Clamp the original table-supported cup without retention probes or relocation."""
    from .long_horizon import _pen_rim_position
    if not ctx.numeric_ik:
        raise RuntimeError('Static cup support requires audited numerical motion')
    ctx.stage = 'pen_holder_initial_geometry'
    position = _pen_rim_position(ctx, holder)
    position[2] = ctx.table_z+.060
    yaw = float(ctx.memory.get('pen_rim_geometry', {}).get('yaw', holder.get('yaw', 0.)))
    if not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError('Initial cup geometry is not finite')
    arm = ctx.arm_for(center)
    pose = dict(position=position.tolist(), quaternion=Rotation.from_euler('z', yaw).as_quat(
        scalar_first=True).tolist(), yaw=yaw, accepted=True, measurement_step=int(ctx.steps),
        cameras=[holder.get('camera', 'cam_head')], position_error_m=POSE_ERROR,
        rotation_error_rad=math.radians(3), geometry_model='initial_table_supported_rgb',
        table_support_prior_used=True, measurement_role='ordinary_table_supported_contact',
        geometry_bound_source='public_table_pose_and_ordinary_clearance_reserve',
        physical_retention_verified=False)
    rejected = []
    selected = None
    desired_yaw = math.pi if arm == 'right' else 0.
    candidates = sorted(_grasp_candidates(arm, pose), key=lambda value: (
        value.get('entry_kind') == 'top_rim_wall_pinch', value['native_grasp_height_m'],
        abs(math.atan2(math.sin(value['world_tool_yaw_rad']-desired_yaw),
                       math.cos(value['world_tool_yaw_rad']-desired_yaw)))))
    for candidate in candidates:
        def approach(plan, candidate=candidate):
            yield from _support_move(plan, arm, candidate, 'static_hover', candidate['high'], 18)
            yield from _support_move(plan, arm, candidate, 'static_descent', candidate['outside'], 12)
            yield from _support_move(plan, arm, candidate, 'side_entry', candidate['grasp'], 12)
            yield from plan.grip(arm, 0., 10)
            yield from plan.hold(4)
        ctx.stage = 'pen_holder_static_clamp'
        try:
            audit = _audit(ctx, approach, arm, pose, stationary_table_contact=True)
        except (RuntimeError, ValueError) as error:
            rejected.append(_path_rejection(candidate, error))
            continue
        selected = (deepcopy(candidate), approach, audit)
        break
    if selected is None:
        ctx.memory['pen_holder_support'] = dict(support_arm=arm, geometry_verified=True,
            retention_verified=False, support_mode='stationary_clamp', rejected_candidates=rejected)
        raise RuntimeError('No safe original-site cup clamping path: '+str(rejected[-1] if rejected else 'no candidates'))
    candidate, program, audit = selected
    yield from program(ctx)
    ctx.memory.pop('_pen_support_segment', None)
    report = dict(verified=True, geometry_verified=True, support_mode='stationary_clamp',
        retention_verified=False, physical_retention_verified=False,
        acquisition_command_sequence_complete=True, support_arm=arm,
        current_pose=deepcopy(pose), origin_pose=deepcopy(pose),
        body_center=pose['position'], packing_yaw=pose['yaw'],
        measurement_step=pose['measurement_step'], command_completion_step=int(ctx.steps),
        body_center_reference=pose['position'], support_tcp_reference=ctx.tcp(arm).tolist(),
        support_quaternion_reference=np.asarray(ctx.state[arm+'_ee_pose'])[3:].tolist(),
        grasp_position=candidate['grasp'], outside=candidate['outside'],
        grasp_quaternion=candidate['quaternion'], approach_candidate=candidate,
        approach_preflight=audit, rejected_candidates=rejected,
        short_lift_performed=False, cup_relocated=False,
        limits='Initial table-supported RGB geometry and audited clamping commands; no following or retention measurement',
        **_native_sites(pose))
    ctx.memory['pen_holder_support'] = deepcopy(report)
    return arm, report


def verify(ctx, support_arm, report):
    """Check present stability without moving the support hand or claiming a new grip."""
    if not report.get('verified') or support_arm != report.get('support_arm'):
        _failure(ctx, support_arm, dict(report), 'A previously verified support acquisition is required')
    ctx.stage = 'pen_holder_support_stability'
    yield from ctx.hold(3)
    current = measure(ctx, report['body_center'], support_arm, report['packing_yaw'], allow_single=True)
    actual = np.asarray(ctx.tcp(support_arm), float)
    expected = np.asarray(report['support_tcp_reference'], float)
    tool_error = float(np.linalg.norm(actual[:3]-expected[:3]))
    tool_angle = float((Rotation.from_quat(np.asarray(ctx.state[support_arm+'_ee_pose'])[3:],scalar_first=True).inv()*
                        Rotation.from_quat(report['support_quaternion_reference'],scalar_first=True)).magnitude())
    body_error = float('inf') if not current['accepted'] else float(
        np.linalg.norm(np.asarray(current['position'])-report['body_center_reference']))
    accepted = current['accepted'] and body_error <= .008 and tool_error <= .005 and tool_angle <= SUPPORT_ROTATION_ERROR
    metrics = {'current_pose': current, 'cup_displacement_m': None if not np.isfinite(body_error) else body_error,
               'tool_displacement_m': tool_error, 'tool_rotation_displacement_rad': tool_angle, 'initial_short_lift_verified': True,
               'new_attachment_proof': False, 'support_hand_not_moved': True,
               'existing_verified_grasp_only': True, 'maximum_cup_displacement_m': .008,
               'limits': 'Present pose consistency following a prior resolved lift; no new force or attachment measurement'}
    evidence = visual_evidence(ctx, OBJECT_ID, 'geometry', accepted=bool(accepted),
                               source='rgb_current_supported_cup_relative_pose_stability',
                               cameras=('cam_head', 'cam_'+support_arm+'_wrist'), metrics=metrics,
                               reason='' if accepted else 'Current supported cup or support hand is unstable or unobservable')
    outcome = verification(ctx, ctx.stage, OBJECT_ID, evidence, 'geometry')
    if not outcome.verified:
        _failure(ctx, support_arm, dict(report), evidence.reason)
    fresh = deepcopy(report)
    # A coarse maintenance view cannot replace the earlier registered insertion geometry.
    refresh_geometry = (current.get('position_error_m', float('inf')) <= POSE_ERROR and
                        current.get('grasp_authorized', False))
    if refresh_geometry:
        fresh.update(_native_sites(current))
        fresh.update(body_center=current['position'], packing_yaw=current['yaw'])
    fresh.update(verified=True, measurement_step=ctx.steps, stability_evidence=metrics,
                 insertion_geometry_refreshed=refresh_geometry)
    ctx.memory['pen_holder_support'] = deepcopy(fresh)
    return fresh


def freeze(ctx, support_arm, report):
    """Cache acquired cup geometry and keep its support command until release."""
    if not report.get('verified') or support_arm != report.get('support_arm'):
        raise ValueError('Fixed cup support requires its completed initial acquisition')
    fresh = deepcopy(report)
    fresh['cached_support_pose'] = deepcopy(report['current_pose'])
    fresh['insertion_geometry_measurement_step'] = int(report['measurement_step'])
    fresh['insertion_geometry_source'] = ('initial_stationary_cup_pose'
        if report.get('support_mode') == 'stationary_clamp' else 'initial_returned_cup_pose')
    fresh['insertion_geometry_refreshed'] = False
    ctx.lock_pen_holder_support(support_arm)
    fresh['support_command_locked'] = True
    ctx.memory['pen_holder_support'] = deepcopy(fresh)
    return fresh


def put_back(ctx, support_arm, report):
    """Release at the cached table site after auditing the opening and retreat."""
    if not report.get('verified') or support_arm != report.get('support_arm'):
        _failure(ctx, support_arm, dict(report), 'Cannot release an unverified cup support')
    current = deepcopy(report['cached_support_pose'])
    def release(plan):
        plan.unlock_pen_holder_support()
        yield from plan.grip(support_arm, 1., 8)
        yield from plan.hold(3)
        yield from plan.move(support_arm, report['outside'], report['grasp_quaternion'], 1., 12)
        yield from plan.hold(3)
    ctx.stage = 'pen_holder_support_release'
    options = {'stationary_table_contact': True} if report.get('support_mode') == 'stationary_clamp' else {}
    audit = _audit(ctx, release, support_arm, current, **options)
    yield from release(ctx)
    return dict(completed=True, support_arm=support_arm, released=True,
                **_native_sites(current), completion_step=int(ctx.steps),
                geometry_measurement_step=report['insertion_geometry_measurement_step'],
                geometry_source=report['insertion_geometry_source'],
                new_visual_verification=False, preflight=audit)
