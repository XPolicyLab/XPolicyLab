"""Use calibrated retained source material only as an eligible identity soft cost."""
import importlib
from types import FunctionType

import cv2
import numpy as np

PREFIX = 'classify_r32_language_source_material'
COST_SCALE = .010
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
FRAME_KEY = '_hexa_source_material_frame'
BEFORE_KEY = '_hexa_source_material_before'


def snapshot(ctx):
    """Copy RGB and measured camera geometry from one observation without yielding."""
    observation = ctx.obs
    state = {key: np.asarray(value).copy() for key, value in observation['state'].items()}
    cameras = {}
    for camera in CAMERAS:
        if camera not in observation['vision']:
            continue
        rgb = np.asarray(observation['vision'][camera]['color'], np.uint8)[..., :3].copy()
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, state, rgb.shape)
        height, width = rgb.shape[:2]
        matrix = np.array([[fx, 0., width / 2.], [0., fy, height / 2.], [0., 0., 1.]])
        transform = np.eye(4)
        # Coap rays use (x,-y,-1); Hexa optical rays use (x,y,+1).
        transform[:3, :3] = np.asarray(rotation) @ np.diag([1., -1., -1.])
        transform[:3, 3] = origin
        cameras[camera] = dict(image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               intrinsics=matrix, extrinsics_world=transform)
    return dict(cameras=cameras, state=state, step=int(ctx.steps))


def entity(frame, obj, table_z):
    """Bind an actual entity mask and seed to matching pixels in that same frame."""
    camera = obj.get('camera', 'cam_head')
    if camera != 'cam_head' or camera not in frame['cameras']:
        return None
    crop = np.asarray(obj.get('crop', ()), np.uint8)
    mask = obj.get('entity_mask')
    if mask is None or crop.ndim != 3:
        return None
    mask = np.asarray(mask, np.uint8) > 0
    x, y, width, height = map(int, obj['bbox'])
    image = frame['cameras'][camera]['image']
    if (mask.shape != (height, width) or crop.shape[:2] != mask.shape
            or x < 0 or y < 0 or y + height > image.shape[0] or x + width > image.shape[1]):
        return None
    current_crop = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2RGB)
    if not mask.any() or not np.array_equal(current_crop[mask], crop[..., :3][mask]):
        return None
    pixel = np.asarray(obj.get('grasp_pixel', obj.get('pixel', ())), float)
    if pixel.shape != (2,):
        return None
    local = np.rint(pixel - [x, y]).astype(int)
    if not (0 <= local[0] < width and 0 <= local[1] < height and mask[local[1], local[0]]):
        return None
    full_mask = np.zeros(image.shape[:2], np.uint8)
    full_mask[y:y + height, x:x + width] = mask.astype(np.uint8) * 255
    calibration = frame['cameras'][camera]
    ray = calibration['extrinsics_world'][:3, :3] @ np.linalg.solve(calibration['intrinsics'], np.r_[pixel, 1.])
    origin = calibration['extrinsics_world'][:3, 3]
    surface = float(obj.get('surface_z', table_z))
    if abs(ray[2]) < 1e-10:
        return None
    point = origin + ray * ((surface - origin[2]) / ray[2])
    return dict(mask=full_mask, point=point, frame=frame, bbox=list(obj['bbox']))


def retention(before, after, observed, table_z):
    """Run the original Hexa matching/triangulation with private entity-center masks."""
    unknown = dict(status='unknown', retention_verified=False, matched_points=0)
    if before is None or observed is None:
        return unknown
    hexa = importlib.import_module('coap.tasks.' + PREFIX + '_hexa')
    active_mask = [None]

    def camera_data(frame, name):
        camera = hexa.SOURCES[name]
        data = frame['cameras'][camera]
        active_mask[0] = (before['mask'] if frame is before['frame'] else observed['mask']) if name == 'head' else None
        return data['image'], data['intrinsics'], data['extrinsics_world']

    class MaterialSift:
        """Keep head descriptor support inside the actual measured entity mask."""
        def __init__(self):
            self.detector = cv2.SIFT_create()

        def detectAndCompute(self, gray, mask):
            material = active_mask[0]
            keys, descriptors = self.detector.detectAndCompute(gray, material)
            if material is None or descriptors is None:
                return keys, descriptors
            distance = cv2.distanceTransform(material, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            retained = []
            for index, key in enumerate(keys):
                # SIFT uses four descriptor cells, width 3*sigma and sigma=size/2.
                radius = np.ceil(3. * (key.size / 2.) * np.sqrt(2.) * (4. + 1.) / 2.)
                x, y = np.rint(key.pt).astype(int)
                # Cover subpixel center rounding and the raster mask boundary.
                if distance[y, x] > radius + np.sqrt(2.):
                    retained.append(index)
            if not retained:
                return [], None
            return [keys[index] for index in retained], descriptors[retained]

    class VisionProxy:
        """Keep OpenCV bindings private to this one evidence call."""
        SIFT_create = MaterialSift

        def __getattr__(self, name):
            return getattr(cv2, name)

    namespace = dict(hexa.__dict__, camera_data=camera_data, cv2=VisionProxy())
    for name in ('rays', 'triangulate', 'sift_source_evidence'):
        original = getattr(hexa, name)
        namespace[name] = FunctionType(original.__code__, namespace, name, original.__defaults__, original.__closure__)
    result = namespace['sift_source_evidence'](before['frame'], after, before['point'], table_z)
    return dict(result, matched_points=int(result.get('matched_points', 0)))


def pair_bonus(ctx, track, obj):
    """Reward positive material retention only; unknown contributes exactly zero."""
    frame = getattr(ctx, FRAME_KEY)
    cache = frame.setdefault('pairs', {})
    key = (track['id'], id(obj))
    if key not in cache:
        measured = entity(frame, obj, ctx.table_z)
        result = retention(track.get(BEFORE_KEY), frame, measured, ctx.table_z)
        bonus = COST_SCALE if result['status'] == 'source_material_remains' else 0.
        cache[key] = bonus
        ctx.memory.setdefault('classify_b', {}).setdefault('source_material_evidence', []).append(dict(
            source_id=track['id'], after_step=frame['step'],
            before_step=track[BEFORE_KEY]['frame']['step'] if track.get(BEFORE_KEY) else None,
            fresh_bbox=list(obj['bbox']), bonus=bonus, **result))
    return cache[key]


def associate_language_sources(ctx, tracks, objects, refresh_index, baskets=()):
    """Preserve parent semantics and attempts while installing frame-local soft evidence."""
    association = importlib.import_module('coap.tasks.classify_r36_language_global_assignment_association')
    frame = snapshot(ctx)
    previous = getattr(ctx, FRAME_KEY, None)
    setattr(ctx, FRAME_KEY, frame)
    try:
        result = association.associate_language_sources(ctx, tracks, objects, refresh_index, baskets)
        for track in tracks:
            if track.get('visible'):
                track[BEFORE_KEY] = entity(frame, track['object'], ctx.table_z)
        return result
    finally:
        if previous is None:
            delattr(ctx, FRAME_KEY)
        else:
            setattr(ctx, FRAME_KEY, previous)
