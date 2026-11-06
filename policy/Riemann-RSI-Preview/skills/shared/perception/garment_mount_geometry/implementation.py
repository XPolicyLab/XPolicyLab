"""Canonical stateless helpers; caller-owned state and clocks remain local."""
from copy import deepcopy
import hashlib
import cv2
import numpy as np
from skills.shared.motion.declared_pickup_kinematics.implementation import run as fixed_own_model
from skills.shared.motion.garment_joint_sampling.implementation import current_snapshot
from skills.shared.perception.garment_scene_view.implementation import GarmentSceneView
from skills.shared.motion.garment_command_execution.implementation import unavailable
from skills.shared.motion.urdf_pose_kinematics.implementation import matrix_pose
from skills.shared.perception.current_garment_landmarks.implementation import (
    LANDMARKS, prompted, outline_supported, points_on_plane, template_landmarks,
    current_mask, segment, seeded_masks, _component_outline, torso_observed,
    plane_points, numeric_outline)
from skills.shared.planning.current_garment_contacts.implementation import run as contact_route
from skills.shared.verification.garment_mounting_datum.implementation import (
    SOURCE as MOUNTING_DATUM_SOURCE, URDF_SHA256, validate as validate_mounting_datum)
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


def current_mounting_datum(api, epoch):
    """Original v45 roots from current public EEF + six joints + static URDF."""
    joints, targets = api.own_joint_states(), api.targets()
    state = {side + suffix: np.asarray(joints[side] if suffix == '_arm_joint_state'
                else targets[side + suffix]).tolist()
             for side in ('left', 'right') for suffix in ('_arm_joint_state', '_ee_pose')}
    model = fixed_own_model()
    roots = {side: model.root(state[side + '_arm_joint_state'], state[side + '_ee_pose'])
             for side in ('left', 'right')}
    heights = [transform[2, 3] for transform in roots.values()]
    datum = {'source': MOUNTING_DATUM_SOURCE,
        'reference_semantics': 'v45_mean_current_public_own_root_Z_static_task_datum',
        'kinematic_model_sha256': URDF_SHA256, 'current_public_action_epoch': epoch,
        'public_own_state': state, 'own_base_poses': {s: matrix_pose(v).tolist() for s, v in roots.items()},
        'mount_z_m': float(np.mean(heights)),
        'root_height_disagreement_m': float(abs(heights[0] - heights[1])),
        'height_is_observed_surface': False, 'surface_height_error_bound_m': None}
    return validate_mounting_datum(datum)

def provenance(api, epoch):
    return {'issued_action_epoch': epoch, 'command_time_seconds': epoch / 25.,
            'RGB_sha256': {name: hashlib.sha256(api.image(name).tobytes()).hexdigest()
                           for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}}

def _retained_torso_with_datum(previous, datum, empty, reason):
    """Keep the last qualified estimate through an incoherent reading."""
    anchor = deepcopy(previous)
    anchor.update(support_memory=datum, height_m=float(datum['mount_z_m']),
                  support_uncertainty_m=None)
    return dict(empty, anchor=anchor, alternatives=[anchor], status='observed',
                torso_read_fallback=reason)

def torso_read_with_datum(api, previous, datum, image, empty, retry=True):
    """Replace only the two hem landmarks from the current silhouette.

    The packaged locate(phase="sleeves_folded") kept the whole flat estimate and
    re-read nothing but the two hem corners, because its template fit had
    already followed the cloth: the shoulders, chests and cuffs stayed where the
    flat reading put them, and only the hem edge moved when the sleeves were
    folded inboard.  Re-deriving a grasp end and a landing end from the current
    silhouette instead put the grasp on the collar and the landing on the
    sleeve tip (r81, r82), so the torso fold never folded the garment in half.
    """
    if previous is None:
        return empty
    retained = _retained_torso_with_datum(previous, datum, empty, 'no_current_torso_reading')
    lateral = np.asarray(previous.get('lateral_axis'), dtype=float)
    longitudinal = np.asarray(previous.get('longitudinal_axis'), dtype=float)
    origin = np.asarray(previous.get('center'), dtype=float)
    if (lateral.shape != (2,) or longitudinal.shape != (2,) or origin.shape != (2,)
            or not (np.isfinite(lateral).all() and np.isfinite(longitudinal).all()
                    and np.isfinite(origin).all())):
        return retained
    lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-9)
    longitudinal = longitudinal / max(float(np.linalg.norm(longitudinal)), 1e-9)
    # The retained axes are the fitted template rotation, so they carry the
    # garment's own frame across the sleeve stage without a second fit.
    rotation = np.column_stack([lateral, longitudinal])
    height = float(datum['mount_z_m'])
    observed = torso_observed(image, previous)
    if observed is None:
        return retained
    mask = observed
    rows, columns = np.where(mask > 0)
    if len(rows) < 300:
        return retained
    xy = plane_points(api, 'cam_head', np.column_stack([columns, rows]), height + .005)[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    if np.count_nonzero(valid) < 300:
        return retained
    xy = xy[valid]
    local = (xy - origin) @ rotation
    # A sleeve that stayed outside the torso must not turn into an imaginary
    # wide hem: restrict the edge search to the retained public hem
    # neighbourhood, with the packaged 35 mm deformation margin.
    public_hem = np.asarray([np.asarray(previous[side + '_hem'], dtype=float)
                             for side in ('left', 'right')])
    if public_hem.shape != (2, 2) or not np.isfinite(public_hem).all():
        return retained
    hem_local = (public_hem - origin) @ rotation
    lower, upper = float(hem_local[:, 0].min()) - .035, float(hem_local[:, 0].max()) + .035
    hem_seed = float(np.mean(hem_local[:, 1]))
    hem_region = local[(local[:, 0] > lower) & (local[:, 0] < upper)
                       & (local[:, 1] < hem_seed + .065) & (local[:, 1] > hem_seed - .065)]
    if len(hem_region) < 100:
        if retry:
            # The packaged rule re-segmented once from the numeric outline seed
            # before it gave up on the current frame.
            seed = numeric_outline(previous)
            if seed is not None:
                try:
                    resegmented = segment(image, seed)
                except ValueError:
                    resegmented = None
                if resegmented is not None and int(np.count_nonzero(resegmented)) >= 300:
                    rows, columns = np.where(resegmented > 0)
                    xy = plane_points(api, 'cam_head', np.column_stack([columns, rows]),
                                      height + .005)[:, :2]
                    valid = np.isfinite(xy).all(axis=1)
                    if np.count_nonzero(valid) >= 300:
                        local = (xy[valid] - origin) @ rotation
                        hem_region = local[(local[:, 0] > lower) & (local[:, 0] < upper)
                                           & (local[:, 1] < hem_seed + .065)
                                           & (local[:, 1] > hem_seed - .065)]
                        if len(hem_region) >= 100:
                            mask = resegmented
        if len(hem_region) < 100:
            return retained
    left_x, right_x = (float(value) for value in np.quantile(hem_region[:, 0], [.04, .96]))
    points = {}
    for side, x in (('left', left_x + .018), ('right', right_x - .018)):
        corner = hem_region[np.abs(hem_region[:, 0] - x) < .020]
        if len(corner) < 20:
            return retained
        corner_y = float(np.quantile(corner[:, 1], .03))
        points[side + '_hem'] = (rotation @ np.array([x, corner_y + .028]) + origin).tolist()
    # supported_contact() reads the current mask, so the retained outline must
    # be this frame's silhouette rather than the flat one.
    outline = _component_outline(mask)
    if outline is None:
        return retained
    samples = image[mask > 0]
    if len(samples) < 200:
        return retained
    anchor = dict(previous, **points)
    anchor.update(source='current_public_rgb_garment_outline', support_memory=datum,
                  height_m=height, support_uncertainty_m=None,
                  head_outline=outline.tolist(),
                  appearance_hsv=np.asarray(previous['appearance_hsv'], dtype=float).tolist())
    return dict(empty, anchor=anchor, alternatives=[anchor], status='observed',
                current_torso_bounds_local_m=[left_x, right_x, hem_seed])

def read_with_datum(api, datum, previous=None, reuse=False, phase='flat'):
    empty = {'anchor': None, 'support_memory': previous['support_memory'] if previous else None,
             'status': 'not_observed', 'rgb_geometry_is_hidden_surface': False}
    datum = validate_mounting_datum(datum)
    empty['support_memory'] = datum
    image = api.image('cam_head')
    if phase != 'flat':
        return torso_read_with_datum(api, previous, datum, image, empty)
    height = float(datum['mount_z_m'])
    regions = prompted(api, ['shirt', 'clothing', 'garment'])
    heads = [row for row in regions['cam_head'] if not row['clipped']]
    outline = None
    if len(heads) == 1:
        candidate = np.asarray(heads[0]['outline'], dtype=float)
        if outline_supported(api, previous, candidate, image):
            outline = candidate
    # The packaged rule reuses the first flat estimate for the second sleeve:
    # fold_sleeve() refines only the contact point against the current head
    # image and never re-fits a flat-shirt outline once a sleeve is folded.
    # A fresh fit on a half-folded garment is what previously produced an empty
    # anchor and a zero-action abort, so reuse keeps the retained axes and
    # snaps every landmark onto the current public outline instead.
    if reuse and previous is not None:
        if outline is None:
            outline = np.asarray(previous['head_outline'], dtype=float)
        landmarks = deepcopy({name: previous[name] for name in
                              ('center', 'longitudinal_axis', 'lateral_axis', 'confidence', *LANDMARKS)})
        projected = points_on_plane(api, 'cam_head', outline, height)[:, :2]
        for name in ('left_cuff', 'right_cuff', 'left_hem', 'right_hem',
                     'left_chest', 'right_chest'):
            point = np.asarray(landmarks[name])
            distances = np.linalg.norm(projected - point, axis=1)
            if distances.min() < .055:
                landmarks[name] = projected[np.argmin(distances)].tolist()
        # supported_contact() reads head_outline, so a reused anchor must carry
        # the current frame's public outline.  The shoulders are not snapped:
        # they are the body fold's landing point and come from the fitted
        # public template, not from the current silhouette edge.
        landmarks['head_outline'] = outline.tolist()
        landmark_candidates = [landmarks]
    else:
        # The packaged flat reading fitted a public garment outline and
        # transformed that template's own public landmarks.
        fit = None
        if outline is not None:
            try:
                fit = template_landmarks(api, current_mask(image, outline), height)
            except ValueError:
                # A retained outline that cannot be rasterized into this frame
                # is a missing reading, exactly like the unusable segmentation
                # below it, not a reason to abandon the episode with no further
                # robot action.
                fit = None
        if fit is None:
            # The packaged retry_flat_appearance(): six bounded current-RGB
            # seed windows were segmented independently and the fit that
            # explained the most of its own mask was kept.
            best = None
            candidates = []
            try:
                candidates.append(segment(image))
            except ValueError:
                pass
            candidates.extend(seeded_masks(image))
            for mask in candidates:
                candidate = template_landmarks(api, mask, height)
                if candidate is None:
                    continue
                if best is None or candidate['iou'] > best[0]['iou']:
                    best = (candidate, mask)
            if best is not None:
                fit = best[0]
                retry_outline = _component_outline(best[1])
                if retry_outline is not None:
                    outline = retry_outline
        if fit is None:
            if previous is not None:
                landmarks = deepcopy(previous)
                if outline is not None:
                    landmarks['head_outline'] = outline.tolist()
                landmark_candidates = [landmarks]
            else:
                return empty
        else:
            if outline is None:
                outline = np.asarray(previous['head_outline'], dtype=float) if previous else None
                if outline is None:
                    return empty
            marks = fit['landmarks']
            rotation = np.asarray(fit['rotation'], dtype=float)
            landmark_candidates = [{
                'center': np.asarray(fit['parameters'][:2], dtype=float).tolist(),
                'longitudinal_axis': (rotation @ np.array([0., 1.])).tolist(),
                'lateral_axis': (rotation @ np.array([1., 0.])).tolist(),
                'confidence': float(min(1., max(0., fit['iou']))),
                'left_cuff': marks['left_sleeve'], 'right_cuff': marks['right_sleeve'],
                'left_hem': marks['left_hem'], 'right_hem': marks['right_hem'],
                'left_chest': marks['left_chest'], 'right_chest': marks['right_chest'],
                'left_shoulder': marks['left_shoulder'], 'right_shoulder': marks['right_shoulder'],
                'head_outline': outline.tolist()}]
    if not landmark_candidates:
        return empty
    mask = current_mask(image, outline)
    samples = image[mask > 0]
    if len(samples) < 200:
        return empty
    appearance = np.median(cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3), axis=0)
    if previous is not None:
        # The cloth keeps one colour for the whole episode.  Re-measuring it
        # from a silhouette that already includes table let r81 drift the
        # retained appearance from S=33 to S=76, the table's own saturation,
        # and every later colour test then described the table instead of the
        # garment.
        appearance = np.asarray(previous['appearance_hsv'], dtype=float)
    anchors = [dict(landmarks, source='current_public_rgb_garment_outline',
                    support_memory=datum, height_m=float(datum['mount_z_m']),
                    support_uncertainty_m=None,
                    head_outline=outline.tolist(), appearance_hsv=appearance.tolist())
               for landmarks in landmark_candidates]
    return dict(empty, anchor=anchors[0], alternatives=anchors, status='observed')
