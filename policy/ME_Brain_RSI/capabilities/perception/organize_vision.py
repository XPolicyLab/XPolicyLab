"""RGB measurements for desk organization, including moved-object recovery.

Public meshes are used as shape templates. Search areas are workspace bounds,
and never substitute for an observed object or destination.
"""
import math
from functools import lru_cache
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .containers_vision import (color_masks, contours, detect_organize, fit_family,
                                image, region_mask, world_points, models, project_points,
                                boundary_samples, line_distance, silhouette, fit_model)
from ..motion import quat_matrix


def detect_keyboard(ctx):
    rgb = image(ctx)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Include both the initial panel and the placement frame. Bright keycaps
    # belong to the panel; excluding them breaks the gray exterior contour.
    workspace = region_mask(ctx, [-.30, -.31, .36, .12], z=ctx.table_z+.018) > 0
    selected = (hsv[..., 1] < 85) & workspace
    candidates = []
    for contour in contours(selected, area=1700, close=3):
        x, y, width, height = cv2.boundingRect(np.asarray(contour, np.float32))
        if width < 100 or height < 20 or height > 140:
            continue
        item = fit_family(ctx, contour, 'keyboard', yaw_hint=0.0)
        if item is not None and item['fit_error'] < 12:
            candidates.append(item)
    return min(candidates, key=lambda item: item['fit_error']) if candidates else None


def detect_mouse(ctx):
    """Match each observed shell color without joining it to the wood grain."""
    rgb = image(ctx)
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    workspace = region_mask(ctx, [0., -.36, .55, .12], z=ctx.table_z+.03) > 0
    materials = (
        ('bluegray', ((h > 90) & (h < 166) & (s > 30)) | ((s < 100) & (v < 110))),
        ('red', (((h < 6) & (v > 180)) | (h > 170)) & (s > 95) & (v > 70)),
        ('white', (s < 65) & (v > 180)),
    )
    candidates = []
    for appearance, material in materials:
        for contour in contours(material & workspace, area=70, close=2):
            x, y, width, height = cv2.boundingRect(contour.astype(np.float32))
            if not (18 <= width <= 85 and 18 <= height <= 100):
                continue
            variants = [contour]
            if appearance == 'white':
                # The white shell surrounds black front buttons. Omitting
                # them truncates the nose and can flip a silhouette fit 180°.
                local = np.zeros(rgb.shape[:2], bool)
                local[max(0, y-12):y+height+12, max(0, x-12):x+width+12] = True
                buttons = (s < 75) & (v < 120)
                variants = contours((material | buttons) & workspace & local, area=70, close=3)
            for variant in variants:
                item = fit_family(ctx, variant, 'mouse')
                if item is not None and item['fit_error'] < 4:
                    item['appearance'] = appearance
                    candidates.append(item)
    return min(candidates, key=lambda item: item['fit_error']) if candidates else None


def detect_frame(ctx):
    """Recover the thin gray placement outline by its complete edge shape."""
    gray = cv2.cvtColor(image(ctx), cv2.COLOR_RGB2GRAY)
    workspace = region_mask(ctx, [-.07, -.095, .29, .095], z=ctx.table_z+.003) > 0
    candidates = []
    for low, high in ((30, 90), (50, 130)):
        mask = ((cv2.Canny(gray, low, high) > 0) & workspace).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in found:
            if cv2.contourArea(contour) < 80:
                continue
            points = world_points(ctx, contour, ctx.table_z+.003)[:, :2]
            position, size, _ = cv2.minAreaRect(points.astype(np.float32))
            size = np.sort(size)
            if not (.070 < size[0] < .135 and .205 < size[1] < .280):
                continue
            polygon = cv2.approxPolyDP(contour, .03*cv2.arcLength(contour, True), True)
            if 4 <= len(polygon) <= 8:
                candidates.append((float(np.linalg.norm(size-[.10, .24])),
                                   np.r_[position, ctx.table_z+.003]))
    return min(candidates, key=lambda entry: entry[0])[1] if candidates else None


def detect_cushion(ctx):
    rgb = image(ctx)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    workspace = region_mask(ctx, [-.30, -.015, -.06, .235], z=ctx.table_z+.0001) > 0
    edges = cv2.Canny(gray, 14, 45)
    mask = ((edges > 0) & workspace).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in found:
        if cv2.contourArea(contour) < 25:
            continue
        points = world_points(ctx, contour, ctx.table_z+.0001)[:, :2]
        rectangle = cv2.minAreaRect(points.astype(np.float32))
        center, size, angle = rectangle
        size = np.asarray(size)
        if not np.all((size > .035) & (size < .066)):
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .025*perimeter, True)
        if len(polygon) < 4 or len(polygon) > 8:
            continue
        score = float(np.linalg.norm(size-.05))
        candidates.append((score, np.r_[center, ctx.table_z+.0001]))
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


@lru_cache(maxsize=8)
def _drawer_clock_support_levels(model_key):
    """Public tray surface and outer rim heights; no episode geometry."""
    info, data = models()
    height = float(info[model_key]['height'])
    points = np.asarray(data[model_key+'_points'], float)
    # The raised perimeter is separate from the broad upper tray. The public
    # inner mesh lies between these two height cuts, below the perimeter.
    inner = points[(points[:,2] > height-.055) &
                   (points[:,2] < height-.015), 2]
    if not len(inner):
        return (height,)
    level, count = np.unique(np.round(inner,6),return_counts=True)
    broad = level[count >= 8]
    return (float(broad.max()),height) if len(broad) else (height,)


def _detect_elevated_alarm(ctx, expected_xy, expected_yaw):
    drawer = ctx.memory.get('desk_obstacles',{}).get('drawer')
    if drawer is None:
        return None
    info, _ = models()
    rgb = image(ctx)
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
    red, green, blue = rgb.astype(float).transpose(2,0,1)
    material = ((red > green*1.012) & (green > blue*1.025) &
                (saturation >= 8) & (saturation < 80) & (value > 100))
    position = np.asarray(expected_xy,float)
    levels = _drawer_clock_support_levels(drawer['model_key'])
    workspace = np.zeros(rgb.shape[:2],bool)
    for level in levels:
        for height in (0.,.04,.08):
            workspace |= region_mask(ctx,[position[0]-.120,position[1]-.120,
                position[0]+.120,position[1]+.120],z=ctx.table_z+level+height)>0
    candidates=[]
    for seed in contours(material & workspace,area=100,close=3):
        x,y,width,height=cv2.boundingRect(seed.astype(np.float32))
        if width>160 or height>150:
            continue
        labels=np.zeros(rgb.shape[:2],np.uint8)
        # Only a narrow observed boundary band is uncertain. A larger box
        # makes the white cabinet a competing foreground component.
        labels[max(0,y-3):y+height+3,max(0,x-3):x+width+3]=cv2.GC_PR_FGD
        seed_mask=np.zeros(rgb.shape[:2],np.uint8)
        cv2.fillPoly(seed_mask,[seed.astype(np.int32)],255)
        labels[(seed_mask>0)&material&workspace]=cv2.GC_FGD
        cv2.setRNGSeed(0)
        try:
            cv2.grabCut(rgb,labels,None,np.zeros((1,65)),np.zeros((1,65)),
                        4,cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        foreground=(labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)
        for contour in contours(foreground,area=100,close=2):
            for level in levels:
                item=fit_family(ctx,contour,'alarm',yaw_hint=expected_yaw,elevation=level)
                if (item is None or item['fit_error']>=3.0 or
                    np.linalg.norm(item['position'][:2]-position)>=.085):
                    continue
                rotation=Rotation.from_euler('z',item['yaw']).as_matrix()
                # A higher-plane fit can explain the same image with a clock
                # hanging beyond the tray. Require its whole public support.
                relative=Rotation.from_euler('z',-drawer['yaw']).as_matrix()
                outline=(_PUBLIC_ALARM_OUTER_HULL@rotation.T+
                    np.r_[item['position'][:2],ctx.table_z+level]-
                    np.r_[drawer['position'][:2],ctx.table_z])@relative.T
                bounds=np.asarray(info[drawer['model_key']]['bounds'])
                margin=float(min((outline[:,:2]-bounds[0,:2]).min(),
                                 (bounds[1,:2]-outline[:,:2]).min()))
                if margin<.004:
                    continue
                item['object_root']=np.r_[item['position'][:2],ctx.table_z+level]
                item['world_rotation']=rotation
                item['support_level']=level
                item['support_margin']=margin
                item['appearance']='cabinet_clock_warm_neutral'
                candidates.append(item)
    if not candidates:
        return None
    return min(candidates,key=lambda item:item['fit_error'])


def detect_alarm_near(ctx, expected_xy, expected_yaw=0.0, elevation=0.0):
    if elevation>0.:
        return _detect_elevated_alarm(ctx,expected_xy,expected_yaw)
    rgb = image(ctx)
    position = np.asarray(expected_xy, float)
    material = color_masks(rgb)['white']
    candidates = []
    for radius in (.075, .120):
        # A released casing can settle away from its commanded buffer point.
        # Recover its entire observed boundary before fitting the grasp root;
        # a cropped boundary must not masquerade as a smaller upright clock.
        workspace = region_mask(ctx, [position[0]-radius, position[1]-radius,
                                      position[0]+radius, position[1]+radius],
                                z=ctx.table_z+elevation+.04) > 0
        for contour in contours(material & workspace, area=80, close=3):
            item = fit_family(ctx, contour, 'alarm', yaw_hint=expected_yaw, elevation=elevation)
            if (item is not None and item['fit_error'] < 3.0 and
                    np.linalg.norm(item['position'][:2]-position) < .085):
                candidates.append(item)
        if candidates:
            return min(candidates, key=lambda item: item['fit_error'])
    return None


def _alarm_background_mask(ctx, alarm):
    """Censor the observed clock, retaining neighboring colored figure pixels."""
    rgb = image(ctx)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    contour = alarm.get('contour')
    if contour is not None:
        cv2.fillPoly(mask, [np.asarray(contour, np.int32).reshape(-1, 2)], 1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    else:
        x, y, width, height = alarm['bbox']
        mask[max(0, y):y+height, max(0, x):x+width] = 1
    masks = color_masks(rgb)
    # A clock rectangle can overlap a green neck or yellow pedestal only in
    # projection. Such material is positive figure evidence, not background.
    return (mask > 0) & ~(masks['green'] | masks['yellow'])


def _detect_garage_colored(ctx, separate_initial_clock=False):
    """Fit public figures from an observed material seed and pedestal."""
    rgb = image(ctx)
    masks = color_masks(rgb)
    workspace = np.zeros(rgb.shape[:2], bool)
    for height in (.015, .095, .175):
        workspace |= region_mask(ctx, [-.55, -.32, .035, .065], z=ctx.table_z+height) > 0
    selected = (masks['green'] | masks['yellow'] | masks['white']) & workspace
    clock_pixels = np.zeros(rgb.shape[:2], bool)
    if separate_initial_clock and ctx.memory.get('alarm_buffer') is None:
        # A white clock can touch a colored head in projection. Use colored
        # seeds first and exclude only the white clock pixels, so its coarse
        # bounding rectangle cannot erase the neighboring green head.
        selected = (masks['green'] | masks['yellow']) & workspace
        initial_alarm = detect_organize(ctx).get('alarm')
        if initial_alarm is not None:
            x, y, width, height = initial_alarm['bbox']
            clock_pixels[max(0, y-2):y+height+2, max(0, x-2):x+width+2] = True
            clock_pixels &= masks['white']
    # The cabinet and monitor lie above the manipulable object silhouettes.
    selected[:180] = False
    expected = ctx.memory.get('alarm_buffer')
    if expected is not None:
        alarm = detect_alarm_near(ctx, np.asarray(expected)[:2],
                                   ctx.memory.get('alarm_expected_yaw', 0.0),
                                   elevation=ctx.memory.get('alarm_elevation',0.0))
        if alarm is not None:
            clock_pixels |= _alarm_background_mask(ctx, alarm)
            selected[clock_pixels] = False
    choices = []
    seeds = sorted(contours(selected, area=120, close=3),
                   key=lambda contour: cv2.contourArea(contour.astype(np.float32)), reverse=True)[:6]
    for seed in seeds:
        x, y, width, height = cv2.boundingRect(seed.astype(np.float32))
        if width > 190 or height > 190:
            continue
        x0, y0 = max(0, x-18), max(180, y-15)
        x1, y1 = min(rgb.shape[1], x+width+35), min(rgb.shape[0], y+height+32)
        labels = np.zeros(rgb.shape[:2], np.uint8)
        labels[y0:y1, x0:x1] = cv2.GC_PR_FGD
        seed_mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillPoly(seed_mask, [seed.astype(np.int32)], 255)
        labels[(seed_mask > 0) & selected] = cv2.GC_FGD
        labels[clock_pixels] = cv2.GC_BGD
        try:
            cv2.setRNGSeed(0)
            cv2.grabCut(rgb, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                        4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
        for contour in contours(foreground, area=180, close=2):
            fitted = fit_family(ctx, contour, 'garage')
            if fitted is None or fitted['fit_error'] > 6:
                continue
            root = fitted['position'][:2]
            if not (-.55 < root[0] < .05 and -.34 < root[1] < .09):
                continue
            choices.append(fitted)
    return min(choices, key=lambda item: item['fit_error']) if choices else None


def _detect_garage_expanded(ctx, remove_white_background=False):
    """Fit public figures from an observed material seed and pedestal."""
    rgb = image(ctx)
    masks = color_masks(rgb)
    workspace = np.zeros(rgb.shape[:2], bool)
    for height in (.015, .095, .175):
        workspace |= region_mask(ctx, [-.55, -.34, .09, .235], z=ctx.table_z+height) > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    metal = (s < 80) & (v > 50) & (v < 195)
    red = ((h < 6) | (h > 170)) & (s > 90) & (v > 45)
    selected = (masks['green'] | masks['yellow'] | masks['white'] | metal | red) & workspace
    count, components, stats, _ = cv2.connectedComponentsWithStats(((s < 35) & (v > 225)).astype(np.uint8), 8)
    neutral_background = np.isin(components, np.flatnonzero((stats[:, cv2.CC_STAT_AREA] > 600) & (np.arange(count) > 0)))
    neutral_background = cv2.dilate(neutral_background.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if remove_white_background:
        selected &= ~neutral_background
    known = detect_organize(ctx)
    keyboard = detect_keyboard(ctx)
    if keyboard is not None:
        known['keyboard'] = keyboard
    expected = ctx.memory.get('alarm_buffer')
    if expected is not None:
        observed_alarm = detect_alarm_near(ctx, np.asarray(expected)[:2],
            ctx.memory.get('alarm_expected_yaw', 0.0),
                                   elevation=ctx.memory.get('alarm_elevation',0.0))
        if observed_alarm is not None:
            known['alarm'] = observed_alarm
        else:
            known.pop('alarm', None)
    excluded = np.zeros(rgb.shape[:2], bool)
    strict_excluded = np.zeros(rgb.shape[:2], bool)
    for kind in ('alarm', 'drawer', 'keyboard'):
        item = known.get(kind)
        if item is not None and 'bbox' in item:
            if kind == 'alarm':
                part = _alarm_background_mask(ctx, item)
            else:
                x, y, width, height = item['bbox']
                part = np.zeros(rgb.shape[:2], bool)
                part[max(0, y-3):y+height+3,max(0, x-3):x+width+3] = True
            excluded |= part
            if kind == 'alarm':
                strict_excluded |= part
    selected &= ~excluded
    # Furniture excludes seeds; soft foreground can recover an overlapping
    # head. The separately observed clock remains definite background.

    expected = ctx.memory.get('alarm_buffer')
    if expected is not None:
        alarm = detect_alarm_near(ctx, np.asarray(expected)[:2],
                                   ctx.memory.get('alarm_expected_yaw', 0.0),
                                   elevation=ctx.memory.get('alarm_elevation',0.0))
        if alarm is not None:
            part = _alarm_background_mask(ctx, alarm)
            excluded |= part
            strict_excluded |= part
            selected &= ~excluded
    choices = []
    seeds = sorted(contours(selected, area=120, close=3),
                   key=lambda contour: cv2.contourArea(contour.astype(np.float32)), reverse=True)[:6]
    for seed in seeds:
        x, y, width, height = cv2.boundingRect(seed.astype(np.float32))
        if width > 190 or height > 190:
            continue
        x0, y0 = max(0, x-45), max(0, y-35)
        x1, y1 = min(rgb.shape[1], x+width+50), min(rgb.shape[0], y+height+70)
        labels = np.zeros(rgb.shape[:2], np.uint8)
        labels[y0:y1, x0:x1] = cv2.GC_PR_FGD
        seed_mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillPoly(seed_mask, [seed.astype(np.int32)], 255)
        labels[(seed_mask > 0) & selected] = cv2.GC_FGD
        labels[strict_excluded] = cv2.GC_BGD
        if remove_white_background:
            labels[neutral_background] = cv2.GC_BGD
        try:
            cv2.setRNGSeed(0)
            cv2.grabCut(rgb, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                        4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
        for contour in contours(foreground, area=180, close=2):
            fitted = fit_family(ctx, contour, 'garage')
            if fitted is None or fitted['fit_error'] > 6:
                continue
            root = fitted['position'][:2]
            if not (-.55 < root[0] < .10 and -.36 < root[1] < .235):
                continue
            choices.append(fitted)
    return min(choices, key=lambda item: item['fit_error']) if choices else None


def detect_garage(ctx):
    """Keep accurate color fits and recover metallic or occluded silhouettes."""
    choices = []
    colored = _detect_garage_colored(ctx)
    if colored is not None:
        choices.append(colored)
        if colored['fit_error'] < 1.5:
            return colored
    if ctx.memory.get('alarm_buffer') is None:
        separated_clock = _detect_garage_colored(ctx, separate_initial_clock=True)
        if separated_clock is not None:
            choices.append(separated_clock)
            if separated_clock['fit_error'] < 1.5:
                return separated_clock
    expanded = _detect_garage_expanded(ctx)
    if expanded is not None:
        choices.append(expanded)
    if choices and min(item['fit_error'] for item in choices) < 2.5:
        return min(choices, key=lambda item: item['fit_error'])
    # A figure can overlap the bright monitor stand. The alternate observed
    # background mask separates that plane while retaining colored body seeds.
    separated = _detect_garage_expanded(ctx, remove_white_background=True)
    if separated is not None:
        choices.append(separated)
    return min(choices, key=lambda item: item['fit_error']) if choices else None


def scene(ctx):
    # Single-scene memo only. All public fit code and candidate order stay intact.
    from copy import deepcopy as _copy_fit
    from types import FunctionType as _FunctionType
    from . import containers_vision as _public_cv

    def _bits(value):
        if isinstance(value, np.ndarray):
            return ('array', value.dtype.str, value.shape, value.tobytes(order='C'))
        if isinstance(value, np.generic):
            return ('numpy_scalar', value.dtype.str, value.tobytes())
        if isinstance(value, float):
            return ('float64', np.asarray(value, np.float64).tobytes())
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(_bits(item) for item in value))
        if isinstance(value, dict):
            return ('dict', tuple((name, _bits(value[name])) for name in sorted(value)))
        return (type(value).__name__, value)

    def _frame(current):
        cameras = []
        for name in sorted(current.vision):
            color = np.asarray(current.vision[name]['color'])
            geometry = current.perception.camera_geometry(name, current.state, color.shape)
            cameras.append((name, _bits(color), _bits(geometry)))
        return (tuple(cameras),
                tuple((name, _bits(current.state[name])) for name in sorted(current.state)),
                _bits(current.table_z), _bits(current.obs.get('instruction')))

    _anchor = _frame(ctx)
    _fit_memo = {}
    _original_fit = _public_cv.fit_model

    def _same_frame_fit(current, contour, model_key, camera='cam_head', yaw_hint=None,
                        elevation=0.0, partial=False, top_only=False, optimize_z=False):
        if current is not ctx or _frame(current) != _anchor:
            return _original_fit(current, contour, model_key, camera, yaw_hint, elevation,
                                 partial, top_only, optimize_z)
        key = (_anchor, model_key, _bits(np.asarray(contour)), camera, _bits(yaw_hint),
               _bits(elevation), _bits(partial), _bits(top_only), _bits(optimize_z))
        if key in _fit_memo:
            return _copy_fit(_fit_memo[key])
        fitted = _original_fit(current, contour, model_key, camera, yaw_hint, elevation,
                               partial, top_only, optimize_z)
        _fit_memo[key] = _copy_fit(fitted)
        return fitted

    # Private function globals route only this invocation through the memo.
    # The shared module and the original ORG globals are never rebound.
    _cv_names = dict(_public_cv.__dict__)
    _cv_names['fit_model'] = _same_frame_fit
    for name in ('fit_family', 'detect_organize'):
        original = getattr(_public_cv, name)
        copied = _FunctionType(original.__code__, _cv_names, original.__name__,
                               original.__defaults__, original.__closure__)
        copied.__kwdefaults__ = original.__kwdefaults__
        _cv_names[name] = copied
    _base_names = globals()
    _org_names = dict(_base_names)
    for name, original in _base_names.items():
        if (isinstance(original, _FunctionType) and original.__globals__ is _base_names
                and name != 'scene'):
            copied = _FunctionType(original.__code__, _org_names, original.__name__,
                                   original.__defaults__, original.__closure__)
            copied.__kwdefaults__ = original.__kwdefaults__
            _org_names[name] = copied
    _org_names.update(fit_model=_same_frame_fit,
                      fit_family=_cv_names['fit_family'],
                      detect_organize=_cv_names['detect_organize'])

    def _run_scene(ctx):
        result = detect_organize(ctx)
        expected = ctx.memory.get('alarm_buffer')
        if expected is not None:
            alarm = detect_alarm_near(ctx, np.asarray(expected)[:2], ctx.memory.get('alarm_expected_yaw', 0.0),
                                   elevation=ctx.memory.get('alarm_elevation',0.0))
            if alarm is not None:
                result['alarm'] = alarm
            else:
                result.pop('alarm', None)
        keyboard = detect_keyboard(ctx)
        if keyboard is not None:
            result['keyboard'] = keyboard
        mouse = detect_mouse(ctx)
        if mouse is not None:
            result['mouse'] = mouse
        if result.get('frame_position') is None:
            frame = detect_frame(ctx)
            if frame is not None:
                result['frame_position'] = frame
        cushion = detect_cushion(ctx)
        if cushion is not None:
            result['cushion_position'] = cushion
        garage = detect_garage(ctx)
        if garage is not None:
            result['garage'] = garage
        else:
            result.pop('garage', None)
        if ctx.memory.get('alarm_buffer') is None and garage is not None:
            alarm = result.get('alarm')
            if alarm is not None and alarm['fit_error'] > 1.5:
                separated = _separate_initial_alarm(ctx, alarm, garage)
                if separated is not None and separated['fit_error'] < alarm['fit_error']:
                    result['alarm'] = separated
        return result

    _run_private = _FunctionType(_run_scene.__code__, _org_names, 'scene',
                                _run_scene.__defaults__, _run_scene.__closure__)
    return _run_private(ctx)



def _separate_initial_alarm(ctx, alarm, garage):
    """Remove a fitted neighboring figure from a touching white clock mask."""
    rgb = image(ctx)
    x, y, width, height = alarm['bbox']
    workspace = np.zeros(rgb.shape[:2], bool)
    workspace[max(0, y-3):y+height+3, max(0, x-3):x+width+3] = True
    polygon = silhouette(ctx, garage['model_key'], garage['position'][:2], garage['yaw'])
    excluded = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(excluded, np.rint(polygon).astype(np.int32), 255)
    excluded = cv2.dilate(excluded, np.ones((5, 5), np.uint8))
    if not np.any((excluded > 0) & workspace):
        return None
    mask = color_masks(rgb)['white'] & workspace & (excluded == 0)
    choices = []
    for contour in contours(mask, area=160, close=2):
        fitted = fit_family(ctx, contour, 'alarm', yaw_hint=0.)
        if (fitted is not None and fitted['fit_error'] < 2.5 and
                np.linalg.norm(fitted['position'][:2]-alarm['position'][:2]) < .045):
            choices.append(fitted)
    return min(choices, key=lambda item: item['fit_error']) if choices else None


# Public USD visual triangles clipped into 20 mm height slabs.
# Each box contains every triangle portion in its slab; no episode positions.
# Public alarm outer hull: 262 support planes, contains all USD mesh vertices.
_PUBLIC_ALARM_OUTER_HULL = np.array([[0.02777180066953824, -0.024933855499744417, 0.027163242329164845], [0.03996213477444649, 0.02344142924638117, 0.033396638677052524], [0.03918184918956642, 0.024500110260476504, 0.01470756036309007], [-0.038830049225630964, 0.024352937536589053, 0.007888212229002808], [0.03201607401746359, 0.02421540148655074, 0.0014988453267214923], [0.039244060707572706, 0.018370602733591087, 0.0016648398062217434], [0.03851573754470842, 0.025070143382668497, 0.01923029312685801], [0.03915155613192005, 0.024530871949253418, 0.014616395437106557], [0.03996213477444649, 0.020115107635148302, 0.004973748653249663], [0.03595872544074706, 0.0183664811222077, -2.0000000000054696e-06], [0.038616049821466755, 0.014542860495332603, 0.00033371223252222706], [-0.04004186343741417, -0.01962652643119452, 0.00931484976303433], [-0.03537084780499646, -0.023259322605959747, 0.029826097861572014], [0.03906933965038231, -0.019997133275091596, 0.0018227133860338508], [0.039252176170951864, -0.019942665822448167, 0.0019731640711522468], [-0.04004186343741417, -0.010435811069569713, 0.0011296996770161286], [-0.04004186343741418, 0.02381219254627486, 0.03057791169137596], [-0.03640477523662086, 0.025070143382668497, 0.005415704088741365], [-0.03757570178000764, 0.02490630474377559, 0.006360039517640616], [-0.03878193835597726, 0.024803442521368128, 0.012895586068648858], [-0.038526170885238506, 0.025070143382668497, 0.03217606002980347], [-0.039148151041518066, 0.02486698954033362, 0.02287790318705077], [-0.03801925808275659, 0.021969829068880682, 0.0019394498136047836], [-0.038686969176810186, 0.021442927907615348, 0.002223635723648809], [-0.03787762012155796, 0.024677576382250224, 0.005600456461945305], [-0.03845341919719046, 0.024405968475308414, 0.0060943866143147694], [-0.03604666320978622, 0.024886670112932774, 0.0038704813145166023], [-0.037744771947563184, 0.024623596001588765, 0.005102975672106727], [-0.037937604276287534, 0.024294538899083857, 0.004485547331753461], [-0.03572472284831521, 0.025070143382668494, 0.004355424913811759], [-0.03593407330740242, 0.024925115174823112, 0.003901733166259219], [-0.03314870786551929, 0.024591647554625933, 0.002115344608568305], [-0.02268061121492413, 0.025070143382668494, 0.0013878078051004598], [-0.022760898589413218, 0.025059480846699317, 0.0013692164033753292], [-0.03706688322170789, 0.01366527911497559, -2.000000000002e-06], [-0.039706868019983126, -0.0024631337842601444, 0.0005359984274392604], [-0.034440296455957156, 0.020734909437567044, 0.00017259922100027608], [-0.03249524473518055, 0.02006971038800051, -1.9999999999985307e-06], [-0.021904180753118747, 0.024838301700217624, 0.0010496326993566993], [0.022387776410262603, 0.023674981713195698, 0.00041093988969213846], [-0.036736379465537795, 0.019480953057737813, 0.0002736823549939274], [-0.03825620198199514, 0.016591509192574242, 0.0004051713051630365], [-0.037602955019862376, 0.0171611077308442, 0.00021672670210373668], [-0.038854859137226465, 0.021016972595451214, 0.0021546635387633926], [-0.03863576097301728, 0.01869550971316441, 0.000992460334996538], [-0.0384567957485511, 0.01935483520379911, 0.0010395883957920703], [-0.03424036584402187, 0.02363160992204481, 0.0012888961575638694], [-0.03492804352618039, 0.022806416777808133, 0.0009343127343027051], [-0.03134976969483175, 0.023481525421389676, 0.00080285520792708], [-0.03380724334975721, 0.02242970265885353, 0.0005718828090811624], [-0.034173641149403744, 0.024298023723153938, 0.001943132399519435], [-0.034496004395953586, 0.023891950399102357, 0.0015773168077366402], [-0.03593790293729686, 0.023729809936720372, 0.001995835112714072], [0.03348718305915796, 0.023524613707133612, 0.0011051265281030072], [0.033944232389269045, 0.02081606555992578, -2.000000000002e-06], [0.029725939216738082, 0.022195270343661738, -2.000000000002e-06], [0.0292920356909883, 0.022529902048583163, 8.139358781183298e-05], [0.035766056496780815, 0.024022891430067413, 0.002305283440409079], [0.032644281000535395, 0.024612732834246705, 0.0021297239353205433], [0.0351831592822697, 0.024426763859204566, 0.0026115069332912806], [0.03598589818429321, 0.024752458890618653, 0.003692278928525552], [0.03782630954746813, 0.023252297995972213, 0.0029926462277039857], [0.03753538144425338, 0.024750821697343394, 0.005661670299645417], [0.03844272879907555, 0.02445610872414862, 0.007611623603938071], [0.039913355839167344, 0.020276956038990806, 0.004622864787338098], [0.039351182779216363, 0.018278078931985214, 0.00181381503094067], [0.03965320175194026, 0.016939315793490995, 0.002120930463493126], [-0.03991181511223772, -0.01930467028213381, 0.04755538550812106], [-0.03990373030539051, -0.02030313362687354, 0.03407914603992596], [-0.03918569297661747, -0.020910428376613516, 0.039305718139079696], [-0.04004186343741417, -0.016918506729145823, 0.05146794993001265], [-0.040005242743864126, -0.017531455236263004, 0.05128425795234072], [-0.04004186343741417, 0.0073890815745666074, 0.05425681054804951], [-0.04004186343741418, -0.012239248573884103, 0.05385882371132957], [0.03357501006319974, -0.02493385549974441, 0.07005056287536225], [0.03996213477444648, -0.018850414484040463, 0.0023419778711254967], [-0.03883630938452741, -0.0199784779199347, 0.0012726118300791893], [-0.039078680452417754, -0.019924388744757784, 0.0013435986610477874], [-0.039377222372052344, -0.01990091431520215, 0.0019398712397829879], [-0.03208662690115214, -0.019689496628943773, -1.9999999999985307e-06], [-0.03891543758630927, -0.015022838825020934, 9.34483769780732e-05], [-0.038638155239644464, -0.015737162171595917, -2.0000000000054696e-06], [0.03996213477444648, 0.022958120342992543, 0.06305100556087503], [0.03771543449150411, 0.02507014338266849, 0.06668563763365683], [0.02980860540521026, 0.018876941057553897, 0.08000200193715096], [0.009274073732398178, 0.025070143382668494, 0.0783260374650953], [0.01917944404891824, 0.023508535630956895, 0.08000200193715096], [0.005219501775252684, 0.024953971891035813, 0.0786946754591194], [-0.04004186343741417, 0.0215003128800591, 0.04822455070681662], [-0.04004186343741417, 0.023480431605918384, 0.039736529948836155], [-0.03927333730722162, 0.023393900648526374, 0.0057054636916465395], [-0.03928846092830976, 0.023365908527293054, 0.005696093299768456], [-0.038686090581405326, 0.023936605734910087, 0.005111607077870877], [-0.038656135921965304, 0.023932120750227697, 0.005024707600713264], [-0.039301678548728895, 0.02244860018566883, 0.0041167032754758325], [-0.03886005896476043, 0.02289162941693871, 0.0037315390031449555], [-0.03875256294180415, 0.023079956946933336, 0.003779426024679077], [-0.04004186343741417, 0.022264780757621734, 0.009268768511169749], [-0.039677906072012964, 0.022706617119192175, 0.006302086330196077], [-0.04004186343741417, 0.018588834642192735, 0.0034646251776929253], [-0.037059409728950034, 0.020131884961285724, 0.0005498659400922992], [-0.03788805801600646, 0.020047188637625973, 0.000914069198310042], [-0.03775787123509818, 0.021902176051020166, 0.0016971286571107018], [-0.03633664235728166, 0.022659716243568415, 0.0013518322017935444], [-0.03584841015470033, 0.022466722105868142, 0.001026423850120313], [-0.039094177261590625, 0.017919213100925063, 0.0012275597754725677], [-0.03923703978019573, 0.013091772780651456, 0.0006453569785294412], [-0.03944529274992859, 0.011309184402348179, 0.0006795450134744925], [-0.03631151052059283, 0.024535237326990315, 0.003276920696818706], [-0.03613220269535719, 0.024368853777754, 0.0028731710977302587], [-0.036307522431333804, 0.023997100766303285, 0.00248676780331987], [-0.037802618557053275, 0.024224983435320358, 0.0041585537501589755], [-0.03787682273016223, 0.02349318388790036, 0.0031553406817986587], [-0.03798173318035139, 0.023539171570804, 0.0033164956653434045], [0.03576662105401604, 0.023999288865291197, 0.002278198447597222], [0.03400786701058249, 0.02342873820298723, 0.0011372989713474106], [0.0366077442285224, 0.02460737258483572, 0.003852469093388041], [0.037553472568304165, 0.024696831365415246, 0.005449176260215065], [0.0380816667078863, 0.02420693951947715, 0.004921181376217148], [0.037134951452219796, 0.024217346589764235, 0.0035856905471896225], [0.036989843232425164, 0.02428374177522459, 0.0035514101029214817], [0.030322428698805522, 0.02507014338266849, 0.002776230188506614], [0.031769645016026395, 0.024955643442924096, 0.002637080568275073], [0.03586866900777031, 0.02480714612192595, 0.0037813522954045463], [0.03695026481684458, 0.02507014338266849, 0.006727513514458761], [0.0370501243244154, 0.024990114910704537, 0.006086758863675126], [0.038582701509841014, 0.014691827081545292, 0.0003313924367693566], [0.037215669650790005, 0.017693460935246784, 0.0001780863143686917], [0.03837626657183332, 0.019029855152808823, 0.00098003726173641], [0.037528371454027855, 0.01965510273298452, 0.0006711199272425826], [0.038519691266781765, 0.023140131435142455, 0.003799525209372649], [0.03895123589530918, 0.02168019701972783, 0.002941576712360925], [0.03206109500120855, -0.019589725559581977, 0.07947924979890646], [0.03058300273539408, -0.01912110892047451, 0.07993335295274519], [0.030011171109820056, -0.02377301625797977, 0.07574993340190554], [0.027319450660173116, -0.024826897220984415, 0.07473184898727493], [0.031372715803203985, -0.024416554083813796, 0.07449897279672837], [0.02812983491061032, -0.024933855499744414, 0.07422053622805694], [0.026583718167831043, -0.024883595783525128, 0.07467989619040878], [0.022802859091077435, -0.024780315975895795, 0.07479896939035131], [-0.00048724407741772714, -0.02420654863993994, 0.07233745594740743], [-0.029441906390152536, -0.023884108423532932, 0.04927528617629311], [-0.03264227773091778, -0.024133274214884457, 0.0321521617891835], [-0.01770743629655996, -0.024933855499744414, 0.011590976781812614], [-0.023818384856663363, -0.024734600336842574, 0.013437592608602833], [-0.016028432267827998, 0.011304664722051579, 0.08000200193715096], [-0.024623844669538486, 0.022403955394255097, 0.07471280506936741], [-0.01782497243250648, -0.020527513726185047, 0.07695299845360201], [-0.01286019179664068, -0.019732519215305045, 0.07929657493534775], [-0.03942244306005034, -0.019089281463365894, 0.05254913063568389], [-0.03434277624076648, -0.020936546568963602, 0.06066254248514755], [-0.0330094479554523, -0.020165525911120773, 0.06587820640731237], [-0.03394022914700099, -0.019834903476876324, 0.06522278982324123], [-0.03363391342748708, 0.024310577058047716, 0.06490805834173438], [-0.038851035221318046, 0.022683877454992184, 0.0546888928162213], [-0.03866472410896477, 0.01900021796094472, 0.057354118593629724], [-0.0365848236347669, 0.022815934405161414, 0.06091302766522541], [-0.03658306997548409, 0.021424066844149724, 0.061729219496697955], [-0.033748611707919395, -0.01693412461110242, 0.06684184517679762], [-0.035233762408490944, -0.019336689501726092, 0.0637067894274076], [0.035114471489822864, -0.023965999963989827, 0.07071332443109621], [0.03996213477444649, -0.0063113324011856795, 0.0010070729167309563], [0.039756649440586246, 0.0027758571042353596, 0.0007906156795341006], [0.03906202536555838, 0.007656528870393232, 0.0002729774138882958], [0.03853668553468786, -0.0027142728692352374, -1.9999999999985307e-06], [0.038377370486794016, -0.015924041109341345, -2.0000000000054696e-06], [0.03821001553736263, -0.01936552337391973, 0.0002497294655297369], [0.037068984067509005, -0.01919030710598492, -1.9999999999985307e-06], [0.03778008071308161, -0.01849276293843191, -2.000000000002e-06], [0.03895227436787542, -0.019935358656111412, 0.0013165754356532847], [-0.04004186343741417, -0.019552872345819258, 0.006188728361825442], [-0.039583206899845666, -0.019696033959562056, 0.0016649635765110006], [-0.039491601534113276, -0.019821175283323493, 0.0018288226666127068], [-0.03949482534634579, -0.01980775345619181, 0.0017720687133291335], [-0.039759986886020456, -0.019655697314175824, 0.0024861282595283855], [-0.03976953778622516, -0.019575462754652884, 0.0020891627414102096], [-0.039574076245585704, -0.019653604196395117, 0.001534502402195003], [-0.04004186343741417, -0.01595151265300831, 0.0012983714539175142], [-0.03921498168676288, -0.019842698367602666, 0.0012752424105329056], [-0.03917241834670602, -0.019861240612957142, 0.0012631521450575806], [-0.03855819051880776, -0.019889864669744923, 0.0008061459146716], [-0.03777999293256525, -0.019927430918923135, 0.0006187211694093191], [-0.03721039500741414, -0.01896260143956021, -1.9999999999985307e-06], [-0.038170912791910164, -0.019342730767972764, 0.000181173299291508], [-0.0382545088846283, -0.01806971836213453, -2.000000000002e-06], [-0.038327392674269656, -0.018697558813302175, 0.00010321657222774061], [-0.038175047140910294, -0.01935218701101627, 0.00018473562319867506], [-0.03861576555954553, -0.019256834776302894, 0.0003304131255857483], [-0.03833076030088942, -0.01942261476710388, 0.00025829263188816703], [-0.039019979952558494, -0.019485262896072216, 0.0006512523988331832], [0.03996213477444649, 0.022143165474233596, 0.06783683285127944], [0.03996213477444649, 0.016139626857242743, 0.0722735268880835], [0.039647424446809146, 0.018864203103152104, 0.0725284038399423], [0.036368666137630776, 0.0016106442646040137, 0.07857471684613088], [0.035189367475730024, -0.0003133729237912059, 0.0795250066828122], [0.03785385233528361, 0.024888096551511193, 0.06964775460398], [0.032974381386538326, 0.013940378672883984, 0.07972136136995947], [0.030714914102262802, 0.018606074650030984, 0.07992165923814396], [0.0308756763482284, 0.024031895346799628, 0.0782841016038103], [0.034502838772941184, 0.02507014338266849, 0.07497773832736881], [0.027744113984323807, 0.025070143382668494, 0.07738562634763471], [0.030036855227835665, 0.024627244618807605, 0.07788726116443213], [0.038482737573237634, 0.021517955258197743, 0.07402365297561397], [0.03643786350853337, 0.02301486148566181, 0.07578004634145083], [0.035718742034950623, 0.02034666382966186, 0.07733844364145606], [0.03667843925291969, 0.01841518022777496, 0.07687602988767112], [-0.020507715426612193, 0.02405576410709821, 0.07638599457498532], [-0.020695155780244282, 0.024064793175695355, 0.07627674226713299], [-0.02308737157947207, 0.025070143382668494, 0.07348990950228139], [-0.02387248507331011, 0.02493654146639092, 0.07330046548817877], [-0.028342303386435796, 0.024415050060056634, 0.07101633795979068], [-0.028992712777121708, 0.024370997284128604, 0.070380820996855], [-0.0279370125690475, 0.02457301560085722, 0.07110295052524548], [-0.03044288887678973, 0.025070143382668494, 0.06707361084813579], [-0.033794071511733935, 0.024723229263072688, 0.06408347963811356], [-0.03783107057665769, 0.024165882632145606, 0.056046802263493484], [-0.0376869064229487, 0.023766036674757837, 0.05729491858904186], [-0.03726122059336902, 0.023654425670403635, 0.05854357873104259], [-0.03390193927972665, 0.024802687821647088, 0.06377239785140007], [-0.034048165362407776, 0.025070143382668487, 0.06282123262487763], [-0.03392230567905901, 0.02507014338266849, 0.06311519796263473], [-0.03780126242585344, 0.025070143382668497, 0.049549287436250915], [-0.03542647920781953, 0.025070143382668497, 0.058940854648122695], [-0.038015483324784036, 0.024240631840812498, 0.05520725249632703], [-0.039269446927986604, 0.02379725402376296, 0.04950211570725262], [-0.03815743988237936, 0.024656950449690938, 0.052419739659077515], [-0.03798375256788114, 0.02459012353710293, 0.05379868175260072], [-0.039224904690341914, 0.023257418287467854, 0.05225534054896329], [-0.03924942191775936, 0.02325553629511722, 0.05211263803043035], [-0.021757113993231717, 0.023951832821328026, 0.07572216192770778], [-0.02505613789888406, 0.023040460259878815, 0.07421924922915674], [-0.027916504719491444, 0.024135073304949958, 0.07159281409022514], [-0.027870377120839916, 0.024043449197508573, 0.0716871063093738], [-0.0399723418968208, 0.0188848814196104, 0.003232126421915473], [-0.03974150736041377, 0.018359354213520123, 0.002338800267588533], [-0.03953692798188855, 0.018775751695999404, 0.0020809457249129394], [-0.03924859999575388, 0.020542451542547913, 0.0024315557971780893], [-0.03957168944150434, 0.021633687790538987, 0.003897157884212882], [-0.039632934826857556, 0.021525429375289923, 0.003984277061135032], [0.03889771558400859, 0.0216120203448318, 0.0028041191137850967], [0.03903121215405697, 0.01892533781826924, 0.0015731608988411652], [0.038554750280262194, 0.01924449556562157, 0.001188862346126366], [0.036994760536814345, 0.020735486808922968, 0.0007619680999404362], [0.035307521044382735, 0.021519217517678817, 0.0004950343719430442], [0.03500531764491009, 0.022766061473474822, 0.0009916992910857132], [0.038616164530788984, 0.02336846089221753, 0.004311359779518936], [0.03851308798448538, 0.024037897234502874, 0.005697276982362994], [0.038123155554560986, 0.02417897526712997, 0.004934649165866289], [0.039330432154523254, 0.022880481583567886, 0.006149488796076866], [0.038939435389523366, 0.023287516371495363, 0.00503405712512487], [0.03892009992905282, 0.023231233607590978, 0.004837328716717458], [0.039782129461141104, 0.02080403529092394, 0.004462216838072502], [0.039463361885391166, 0.021170267992516395, 0.003616554406225346], [0.039341777938101646, 0.02144251574954232, 0.003539598194475073], [0.03348725715799634, -0.024933855499744414, 0.07081905566916893], [0.031001386452379976, -0.020218028242075123, 0.07922277554346457], [0.031958155006246235, -0.01962227801277342, 0.079485362620627], [0.02980860540521026, -0.018900044663021223, 0.08000200193715096], [0.030491729094161032, -0.019143146819638385, 0.07994144420323049], [0.03013451938280078, -0.019451989146265773, 0.07988599425267527], [0.023447524321365123, -0.019458304340250653, 0.08000200193715096], [0.029744410493732266, -0.019561566452490277, 0.07987093932714923], [0.030947911544588213, -0.024868723708115305, 0.07390928440266749], [0.03134365049903994, -0.02465387023779187, 0.0741488817870001], [0.031100037931787426, -0.024933855499744414, 0.07366210136773522], [0.032036808582833406, -0.024933855499744417, 0.0730611535173327], [0.032243524924066416, -0.024901365370270802, 0.07300316508470904], [-0.007615408081499151, -0.019956584552476116, 0.07966280852489718], [-0.01157235419340098, -0.01972432721016156, 0.07951183727782937], [-0.005507208954874338, -0.019599555235091087, 0.08000200193715096], [-0.00995361418277775, -0.019023366950141552, 0.08000200193715096], [-0.016155013566574502, 0.008243680716031413, 0.08000200193715096], [-0.021634176806723273, -0.01593067377419928, 0.07738528164232344], [-0.01147128157857517, -0.018235148118658142, 0.08000200193715096], [-0.01998105037706274, 0.023648130924548948, 0.07684296550815789], [-0.021758705485914902, 0.02224682570005213, 0.07639029319418791], [-0.016481567876945335, 0.013962526763372105, 0.07967309410408158], [-0.01981719727178346, -0.020238075508284145, 0.07677876025220653], [-0.028854919538459367, -0.0193455802200746, 0.07159645755913975], [-0.03135229818490112, -0.019938626913747655, 0.06849491778506625], [-0.02220791438500292, -0.016302509723958275, 0.07705911091087904], [-0.022323583538443426, -0.01914687577780468, 0.07633006909936155], [-0.023635804075881357, -0.019395139849244902, 0.07533856637664876], [-0.026470444118185617, -0.01412583403379863, 0.07451654103774064], [-0.028699524372075014, -0.019246750865572863, 0.07176398651659038], [-0.03945465911487399, -0.01970755497630173, 0.05066140324635898], [-0.039387718527812185, -0.019588226088855538, 0.05157640830394668], [-0.03935574030145889, -0.019872994856431854, 0.05026521994676847], [-0.0353637380028425, -0.019622323456739827, 0.06321339173452073], [-0.038130830311843125, -0.01955656045584994, 0.05687766399730148], [-0.03810498317924922, -0.019761654976606695, 0.056525980869571515], [-0.03696783493738456, -0.02008627881949621, 0.0585317072106676], [-0.03797361203719612, -0.01985524553498446, 0.05664499823803707], [-0.02925907631255359, 0.02432188656895884, 0.0701366967730127], [-0.03293944104977142, 0.023980614505094432, 0.06624659647981895], [-0.03593588183081605, 0.020965549878808092, 0.06306486572031675], [-0.03017236611445926, 0.015539657190657915, 0.07163741405920868], [-0.030837126310045346, 0.022183159331005082, 0.06965030178938608], [-0.03215899615317652, 0.005813071236185479, 0.07060650346399375], [-0.031976223439384714, 0.00603296723638808, 0.07081332805290962], [-0.032196816582482066, 0.02340949815676531, 0.06763683257696332], [-0.032599663720262795, 0.02350583617774758, 0.06708458118823905], [-0.03279620314236228, -0.019492969448852975, 0.0673586604661347], [-0.03286722081383117, -0.019426010859631093, 0.06731566844091479], [-0.0320344883751322, -0.019638924288625355, 0.0681165594911323], [-0.03271282760636116, -0.019009789154268108, 0.06770050174160247], [-0.03065710424758673, -0.01937944549072072, 0.0697585031868708], [0.039593745318311, -0.02009877247335402, 0.06910668260136109], [0.037201811603185454, -0.02303305146726502, 0.06606791665375873], [0.03996213477444649, -0.019621582739029156, 0.06550272028750924], [0.03984642307280451, -0.019781508601321714, 0.06825241037743746], [0.03679268885179367, -0.023516547136199453, 0.06258483992785212], [0.03996213477444648, -0.019241425129477473, 0.004249322200896091], [0.03939868579257168, -0.019927605444876956, 0.002723897237190031], [0.03820448703237097, -0.019378707536314654, 0.0002512386784413734], [0.037708299612662186, -0.019392050149035462, 0.0001591833410397002], [0.03797928424114058, -0.0195841527847169, 0.0002962015946516225], [0.03827801412948046, -0.019904049184803103, 0.0007272274354806936], [0.03833375017796637, -0.019826111112123446, 0.0006300067503026731], [0.038277847559285205, -0.019767839184052872, 0.000535409560973657], [0.03781729069720434, -0.019734689536711234, 0.0003853278661655052], [0.039002682704097284, -0.019880737456147386, 0.0011901247196820092], [0.038993245165408606, -0.01987394279418603, 0.0011628482352374], [0.03975086929764591, -0.019375504266910138, 0.0019009092676954796], [0.03959710208234942, -0.019678485162777665, 0.002055095758398351], [0.03972088401114541, -0.019526634698099146, 0.0022119783908121776], [-0.03946403058105882, -0.01864820632508768, 0.0007773050042645634], [-0.039531694574965796, -0.01858477829056347, 0.0008460387254054028], [-0.03963988572883924, -0.019492694130376204, 0.0014158532386844315], [-0.03906008070398111, -0.01959570630667487, 0.0007618261399028638], [-0.03898969602835585, -0.019703683788543647, 0.0008072213258614978], [-0.039335938007618386, -0.01970571108458813, 0.001170377852392216], [-0.039325902458708664, -0.01971270149926136, 0.0011676054722972312], [-0.03858741414050938, -0.01987237951477753, 0.0007866920017853309], [-0.03886265325263253, -0.019801903972055696, 0.0008441780062017241], [-0.03893408083965424, -0.019731423688743886, 0.0007974063631786196], [0.03943297429461517, 0.0221869543105208, 0.07161856456340635], [0.03949710055025363, 0.02258500577746905, 0.07086218893521974], [0.03897711304905438, 0.023054725256108695, 0.072275280347278], [0.038831694454088274, 0.02307583330810217, 0.07259170157054738], [0.03842920501955052, 0.022020883446101338, 0.07393307455567272], [0.0376832173315409, 0.02302252239345266, 0.07443967373504604], [0.038879495681691745, 0.02387933246704462, 0.07072517166561426], [0.03839428560006672, 0.024339544823453015, 0.07125930418750503], [0.039013913917844964, 0.023448635437086335, 0.07162996134891592], [0.039012637224668016, 0.02344893974647972, 0.07163451462554488], [0.03238768426773097, -0.017267797125963102, 0.07977337105515603], [0.034283000900041175, -0.010212631459868133, 0.07960535451266022], [0.034598738800266766, -0.02154232090962779, 0.07632721548877679], [0.035679342146899794, -0.021639530447714, 0.07526698079966687], [0.03981212453470022, -0.019285972145739985, 0.07082669521869106], [0.03743801935138316, 0.024864928762797715, 0.07173307816543248], [0.0351853497455926, 0.025070143382668494, 0.07422389678201666], [0.036180863357859465, 0.025070143382668497, 0.07273386406332305], [0.03746539204604768, 0.02464710199599585, 0.07262710375024259], [0.037601004010221116, 0.024705748274231415, 0.07217239464870903], [0.03844451944647783, 0.024003856397087724, 0.07220412432538233], [0.038322559749779744, 0.024343217719411753, 0.07156171966569533], [0.03027654128766821, 0.023523502639887083, 0.0788218178314262], [0.025226323221428423, 0.022390091149770967, 0.08000200193715096], [0.02198695878029362, 0.02340615273462212, 0.08000200193715096], [0.025222036695812657, 0.02369305706550552, 0.07943671338738738], [0.031484993636402815, 0.02147242752515713, 0.0792666212811113], [0.030801359059434087, 0.022596436104050373, 0.07915294734605044], [0.030862926050369607, 0.022852281808786054, 0.07903344791281489], [0.03399960147032915, 0.02077375444275774, 0.0784364162325451], [0.03394060767446369, 0.020724385225837215, 0.07847978624713908], [0.03420259697492453, 0.02441970057796048, 0.07636999179787417], [0.030550755033736627, 0.024470897935821544, 0.07794709482230089], [0.03012005323698085, 0.024619108375037276, 0.07788184133530109], [0.03440058797388472, 0.024974328251742176, 0.0752993964429311], [0.03455952967444398, 0.024723837322727014, 0.07566950555527471], [0.0342685757806735, 0.024257758725568974, 0.07651840769730811], [0.03382366444350443, 0.023121264707106255, 0.07765933108266895], [0.03590988108577357, 0.023489193577605795, 0.07598476689827569], [0.0359027385242429, 0.023629015978410946, 0.07588598457265065], [0.03322467981403618, 0.02360217707190951, 0.0777055995456388], [0.0332695898014927, 0.023474901358996547, 0.07776970631998495], [-0.008666622734516884, 0.02479754791407657, 0.07839101310469132], [-0.009533801282656285, 0.024694803166804898, 0.07843018104169318], [-0.01880962602491769, 0.025070143382668494, 0.07565976495507332], [-0.01881343756910999, 0.02507014338266849, 0.07565896019835514], [-0.017995320533740415, 0.02466525063735934, 0.07670757320203653], [-0.01651641110141975, 0.02456906384904704, 0.07722789991226463], [-0.011727739112606614, 0.023857983231646677, 0.07888995622431763], [-0.004109783731660278, 0.022431623034350173, 0.08000200193715096], [-0.00639896508513054, 0.02181624345725779, 0.08000200193715096], [0.03839110296736895, 0.02179819406365249, 0.0023165524743721556], [0.03811907979078427, 0.021338782003821544, 0.001766452017477542], [0.03780476117030478, 0.02318884365744993, 0.002902074811104601], [0.037227904251721636, 0.021597087252227747, 0.001276123859210207], [0.03652447245509608, 0.023200667221115846, 0.0019303825773847039], [0.036454874384498126, 0.022760511658063354, 0.0015496318110344923], [0.03274629624321804, -0.024830469628457648, 0.07269311371210557], [0.03268355882774932, -0.024853325155426768, 0.0727096969648045], [0.033641145180542745, -0.02481508024177692, 0.07120720509214676], [0.033525566246865944, -0.02489176698758969, 0.07109865967463663], [-0.01323210584108056, -0.01969092293227232, 0.07924838686605626], [-0.014484615376605597, -0.019383783952113664, 0.07913921559539944], [-0.021559660222114346, -0.019334213539163043, 0.07660091530372315], [-0.019647592683212936, -0.020081356723371125, 0.07702196647788859], [-0.017742843392138746, -0.019722576672930803, 0.07798880431932809], [-0.01710818601734353, -0.01934875069031345, 0.07843713992721588], [-0.017063294947514192, -0.019363342317108557, 0.07844848902594845], [-0.021761539351678922, -0.020115763764702744, 0.07576213820086289], [-0.024258752162962915, -0.019637250302567286, 0.07475316672761965], [-0.03547334371276709, -0.019460561206180158, 0.06320639592101568], [-0.035484982606722004, -0.019527833377720513, 0.06311985178991869], [-0.03785746137971459, -0.018729929220719636, 0.05826669620935926], [-0.03810821366365147, -0.019538467095936846, 0.05696688387790716], [0.03831802241336271, -0.019696156339923863, 0.00048325181691570226], [0.03821799654880915, -0.01941294595368306, 0.0002687983882427511], [0.03811971157408868, -0.01956319002715258, 0.00031477743629338556], [0.038850073213873605, -0.01910715061898453, 0.0004804204183220216], [0.038199020062841614, -0.01993465415578332, 0.0007607751142214812], [0.03820822148477721, -0.0199265235369196, 0.0007432004124573073], [0.03756652196570666, -0.019792429015118265, 0.00041031916686868133], [-0.02269196121932599, -0.021883119251222306, 0.001789873477560086], [-0.012544424127612284, -0.020829466242642824, -1.9999999999950613e-06], [0.02869856943648628, -0.019823288936588204, -1.9999999999985307e-06], [0.004361377795744148, -0.024933855499744414, 0.0100385426526131], [-0.012027944746428487, -0.024933855499744414, 0.009149732694439104], [0.03935384877124623, -0.01969713451564025, 0.0012981864068404188], [0.03952143503626448, -0.019693722263259373, 0.0017223349466825265], [0.03937740813513209, -0.019790754340020738, 0.0015902688113186118], [0.03932751887592405, -0.01977679582947054, 0.0014270114957071409], [0.03894428717132326, -0.01974253492427375, 0.0009014343969505866], [0.03902949693893361, -0.019798632234098792, 0.001056896529496039], [0.03932010670501084, -0.019685856322380554, 0.0012277089221487646], [0.039962134774446496, -0.018461121282033522, 0.002020441402326871], [0.03996213477444649, -0.011880094649378094, 0.0010532122098183654], [0.039512291537179105, -0.01943687356187204, 0.001245785219078803], [0.039524585098791434, -0.019573635270661603, 0.0014338133625027803], [0.03955011123729016, -0.01957402851964178, 0.0015006735207086994], [0.03968804018755309, -0.019400102733821464, 0.0016610760420072285], [0.039660815838644385, -0.01926711896367207, 0.0014385130003796293], [0.03962704728650793, -0.019276539239315802, 0.001361405666130834], [-0.03936698786416128, -0.01930409685645559, 0.0008668270573686208], [-0.03937440263589718, -0.018927189683848645, 0.0007511768620558738], [-0.03903066932539945, -0.019459546123186823, 0.0006447639758126213], [-0.038957636303913586, -0.019320975285210575, 0.000540920923155927], [-0.03984113926746929, -0.019127865041180747, 0.00162843249569233], [-0.04004186343741417, -0.019047081418791374, 0.0027459348195771408], [-0.04004186343741417, -0.018300227676341565, 0.001855870604014525], [-0.039839894638732144, -0.01939136598338585, 0.001934933937780263], [-0.03982366491624186, -0.01935667773729159, 0.0017954532490379455], [-0.039621651126841226, -0.01946299108149966, 0.0013375450588288644], [-0.039445686141557475, -0.019394445079215863, 0.0010014154647724656], [-0.03961811346818988, -0.019293364439512156, 0.0011967304757273117], [-0.03964018015547534, -0.019313559638917385, 0.0012481244738014007], [-0.03941453219297888, -0.01953178811840341, 0.0010610065730934683], [-0.03941778317100371, -0.019537766306384027, 0.0010709708708128776], [-0.03836779846178178, -0.01964295370474662, 0.00040317457988199926], [-0.03837055266967576, -0.0195758883139388, 0.0003515400291464743], [-0.038333808635944135, -0.01953910270910716, 0.0003148214668651783], [-0.03788838330843959, -0.019585212222170207, 0.0002529675700538095], [-0.03808092389990718, -0.01950842733361933, 0.00023541992493533032], [-0.03819217532460532, -0.01973808612501663, 0.0004387269702671136], [-0.03824494104690989, -0.01973196030377795, 0.0004455373576807836], [0.034437073459194674, -0.019874799906651527, 0.07820908883241831], [0.03485345077509535, -0.018620461510888676, 0.07842480720510814], [0.03839764481176935, 0.023545779671940815, 0.07296201484301448], [0.038386014413468485, 0.02399523308378081, 0.07235557549579623], [0.037763233758633066, 0.023464073858125437, 0.07401780652922352], [0.03755685258925292, 0.02444905680561411, 0.07295750034380957], [0.03690392372496224, 0.02413444591215723, 0.07436400727876924], [0.03692586436363982, 0.02401724544877915, 0.07449405642262166], [-0.014869057107949862, 0.02221823656526248, 0.07899540633650881], [-0.01389049602005488, 0.022783070740081082, 0.07904792384137954], [-0.019755609243047843, 0.02361936298797041, 0.07693254601411857], [-0.014063367872823664, 0.02370732522592067, 0.07860187987791459], [-0.013480959673589286, 0.023605805309220577, 0.07881981794401305], [-0.02827935798524611, -0.019696949588329706, 0.0718156370562213], [-0.027448687317815298, -0.01972148079842511, 0.07240191813521846], [-0.029576152613376188, -0.019878139989182628, 0.07033384790355862], [-0.02346578486751418, -0.020043835613341405, 0.07468458260908115], [0.03932963803348692, -0.019542271659275495, 0.0010679881832560446], [0.03945336025080838, -0.019436664222424363, 0.001146545938644515], [0.039520602424275075, -0.01907308051862644, 0.001084081683044278], [0.03940405134663574, -0.019277306046993938, 0.0009866371639759217], [0.039362712061082716, -0.01928203647599656, 0.0009408822441988249], [0.03949494529085159, -0.01739599524826377, 0.0007779588405634058], [0.03911497072846224, -0.018457673840437575, 0.0005257163763735216], [0.039097021052016394, -0.01863710888351371, 0.0005351723695817441], [0.032828426926625985, -0.01911883862153243, 0.0793596532608252], [0.03212657512374477, -0.01957925338419552, 0.07946421214168985], [0.03306338913611011, -0.019614907078713877, 0.07904701341097442], [0.03410100497218668, -0.0193535573082607, 0.07871338205949067], [0.0341447704440811, -0.019465702814858342, 0.07864671687056884], [0.03551669050795482, -0.019254484223522904, 0.07775789103685471], [0.03454880529450623, -0.019724611409307567, 0.0782665155028262], [0.0345367233243794, -0.019681538311644396, 0.07830378193305859], [0.038804256243305496, -0.006771447682526897, 0.07565831178293018], [0.03871709159186719, -0.015288063766098595, 0.07508209928329226], [0.03996213477444649, 0.004622218459277828, 0.07312925339769372], [0.03996213477444648, -0.01796561952627541, 0.07086838002185909], [0.039846537282028964, -0.018888616029403317, 0.07114233960798791], [0.03925690250571757, -0.019402756028244317, 0.0008931062546373401], [0.039266027833247084, -0.019526650886391227, 0.0009829648277769193], [0.03895856881771454, -0.019689854714924286, 0.0008541191120238698], [0.038713945150405486, -0.019574084297659985, 0.0005874606320715077], [0.038731926572574946, -0.019489467957269812, 0.0005448266847257466], [0.03886640838863909, -0.019379772597557368, 0.0005755811232526047], [0.03805452786601918, -0.01794381762152318, 0.07549443591795696], [0.03813734432045977, -0.01908120398707638, 0.0749823300013122], [0.03693661569754607, -0.019696840403439164, 0.07617660201490006], [0.036371250378470034, -0.019407676842346802, 0.07694982931486234], [0.0364218678598505, -0.019473723664664218, 0.07686597518179464], [0.036258668585861346, -0.02128443845199062, 0.07512309435414485], [0.03702422405948018, -0.019878368690767995, 0.0759096151222987], [0.03748547197785188, -0.021129042623090014, 0.07340430963566984], [0.03851189295800088, -0.019452738647344776, 0.0741621918787509], [0.03878307201872419, -0.019525719099485726, 0.07362984067676297], [0.0395674163497964, -0.01929179840507902, 0.07198655734682505], [0.03934518016963863, -0.019086756685307182, 0.0727113780256898], [0.03879989673122387, -0.019992566950693576, 0.0729402598140326], [0.0389106583172687, -0.019745098412401654, 0.07311659202782654], [0.03956040260134174, -0.019358285485161827, 0.07193973031742143], [0.03960204785556073, -0.01959866444053374, 0.07131366060255201], [0.03963779527046798, -0.019511523538716553, 0.07138567185165219]])

_NEIGHBOR_LAYER_BOUNDS = {'alarm_00000': [[[-0.0400399, -0.0199951, 0.0], [0.0399601, 0.0250681, 0.02]], [[-0.0399661, -0.0199946, 0.02], [0.0399261, 0.025068, 0.04]], [[-0.039961, -0.0199946, 0.04], [0.0398816, 0.0250287, 0.06]], [[-0.0369581, -0.0249319, 0.06], [0.0398812, 0.0250283, 0.08]], [[0.0288423, -0.0176478, 0.08], [0.0288437, -0.0176453, 0.08]]], 'garage_00005': [[[-0.0714936, -0.0710587, 0.0], [0.0710991, 0.0712767, 0.02]], [[-0.0826745, -0.0410099, 0.02], [0.0826093, 0.0647808, 0.04]], [[-0.09124, -0.0278159, 0.04], [0.0910309, 0.0463872, 0.06]], [[-0.0900515, -0.0273297, 0.06], [0.0902043, 0.0204008, 0.08]], [[-0.0859567, -0.022145, 0.08], [0.0857557, 0.0228012, 0.1]], [[-0.0803994, -0.0183618, 0.1], [0.080294, 0.0266078, 0.12]], [[-0.0718497, -0.0244403, 0.12], [0.0727273, 0.0272563, 0.14]], [[-0.0675627, -0.0294769, 0.14], [0.0689261, 0.0260922, 0.16]], [[-0.0615648, -0.0286721, 0.16], [0.0627776, 0.0249043, 0.18]], [[-0.0141343, -0.0261195, 0.18], [0.0139475, 0.0159809, 0.1909951]]], 'garage_00006': [[[-0.0425066, -0.0424968, 0.0], [0.0423259, 0.0426447, 0.02]], [[-0.0247465, -0.0048955, 0.02], [0.0245532, 0.0150921, 0.04]], [[-0.0246869, -0.00933, 0.04], [0.0245052, 0.0160916, 0.06]], [[-0.021896, -0.0122082, 0.06], [0.0217057, 0.0089873, 0.08]], [[-0.0502916, -0.0157792, 0.08], [0.0501209, 0.0134589, 0.1]], [[-0.0523416, -0.0143537, 0.1], [0.0521466, 0.0156809, 0.12]], [[-0.0499723, -0.0158355, 0.12], [0.0497706, 0.0206455, 0.14]], [[-0.0347212, -0.0158723, 0.14], [0.0347485, 0.0220497, 0.16]], [[-0.0313671, -0.0143962, 0.16], [0.0315603, 0.0220095, 0.18]], [[-0.0212805, -0.014349, 0.18], [0.0210656, 0.0179788, 0.1909622]]], 'garage_00007': [[[-0.0678528, -0.059387, 0.0], [0.0676868, 0.0595695, 0.02]], [[-0.0551474, -0.0540719, 0.02], [0.0551488, 0.0534024, 0.04]], [[-0.0530235, -0.0226519, 0.04], [0.0528678, 0.0208408, 0.06]], [[-0.0825123, -0.0255991, 0.06], [0.0823275, 0.0353863, 0.08]], [[-0.0914348, -0.0258549, 0.08], [0.0912384, 0.0396399, 0.1]], [[-0.0914467, -0.0211052, 0.1], [0.0912693, 0.0379207, 0.12]], [[-0.0888143, -0.0242985, 0.12], [0.0886513, 0.0380501, 0.14]], [[-0.0789745, -0.0396612, 0.14], [0.0787274, 0.0370742, 0.16]], [[-0.0535782, -0.0426652, 0.16], [0.0534328, 0.0332541, 0.18]], [[-0.0144837, -0.041561, 0.18], [0.0146517, -0.0071547, 0.1907309]]], 'garage_00008': [[[-0.0291747, -0.0305052, 0.0], [0.0273384, 0.0309066, 0.02]], [[-0.0354124, -0.0346492, 0.02], [0.0319699, 0.0345771, 0.04]], [[-0.0393973, -0.0371159, 0.04], [0.0361775, 0.0371159, 0.06]], [[-0.0104128, -0.0181214, 0.06], [0.007177, 0.004661, 0.08]], [[-0.0370789, -0.0149536, 0.08], [0.0269744, 0.0100262, 0.1]], [[-0.0415691, -0.0114532, 0.1], [0.0415691, 0.0111834, 0.12]], [[-0.0414623, -0.0165812, 0.12], [0.0415264, 0.0104329, 0.14]], [[-0.017714, -0.0295956, 0.14], [0.0146656, 0.013716, 0.16]], [[-0.0178609, -0.0300135, 0.16], [0.0150587, 0.0139227, 0.18]], [[-0.0082301, -0.0154474, 0.18], [0.0046565, -0.0005542, 0.181356]]], 'garage_00011': [[[-0.0486301, -0.0489475, 0.0], [0.0484404, 0.0491306, 0.02]], [[-0.0287052, -0.0179397, 0.02], [0.033371, 0.0317293, 0.04]], [[-0.0286408, -0.0185049, 0.04], [0.0336311, 0.0317442, 0.06]], [[-0.0234381, -0.0193255, 0.06], [0.0245515, 0.0215243, 0.08]], [[-0.0212251, -0.0237496, 0.08], [0.020376, 0.0172537, 0.1]], [[-0.0459654, -0.0204666, 0.1], [0.0169933, 0.0259464, 0.12]], [[-0.0440035, -0.0342773, 0.12], [0.0351788, 0.025405, 0.14]], [[-0.0225527, -0.0359087, 0.14], [0.0352561, 0.0249727, 0.16]], [[-0.0254826, -0.0361823, 0.16], [0.0286376, 0.0177336, 0.18]], [[-0.0211551, -0.0125273, 0.18], [0.0221156, 0.0144493, 0.1909689]]], 'keyboard_00000': [[[-0.1598797, -0.0764465, 0.0], [0.1598797, 0.0764465, 0.02]], [[-0.1598797, 0.001803, 0.02], [0.1598797, 0.074845, 0.0265173]]], 'keyboard_00001': [[[-0.1635774, -0.0800248, 0.0], [0.1635774, 0.0800248, 0.02]], [[-0.1562398, 0.058564, 0.02], [0.1562397, 0.0797421, 0.0221483]]], 'keyboard_00002': [[[-0.1837696, -0.0761152, 0.0], [0.1837696, 0.0761152, 0.012992]]]}

# Public collision triangles conservatively covered by 5 mm voxels, then
# solid material filled and merged into 24 boxes; lower cabinet void retained.
_NEIGHBOR_LAYER_BOUNDS['drawer_00000'] = [[[-0.165, -0.06, 0.0], [-0.15, 0.165, 0.005]], [[0.175, -0.06, 0.0], [0.19, 0.16, 0.005]], [[0.165, 0.17, 0.0], [0.185, 0.175, 0.1]], [[-0.165, 0.17, 0.0], [-0.14, 0.175, 0.1]], [[0.17, 0.16, 0.0], [0.19, 0.17, 0.1]], [[-0.165, 0.165, 0.0], [-0.145, 0.17, 0.1]], [[0.175, -0.065, 0.005], [0.19, 0.16, 0.1]], [[-0.165, -0.065, 0.005], [-0.15, 0.165, 0.1]], [[-0.045, -0.08, 0.14], [0.07, -0.055, 0.165]], [[-0.165, 0.17, 0.1], [0.185, 0.175, 0.205]], [[-0.165, -0.065, 0.1], [-0.15, -0.055, 0.205]], [[0.175, -0.065, 0.1], [0.19, -0.055, 0.205]], [[-0.165, -0.055, 0.1], [0.19, 0.17, 0.205]], [[0.175, -0.065, 0.205], [0.19, 0.16, 0.225]], [[-0.165, -0.065, 0.205], [-0.15, 0.165, 0.225]], [[-0.155, 0.18, 0.0], [0.18, 0.185, 0.23]], [[-0.15, 0.185, 0.0], [0.175, 0.19, 0.23]], [[-0.16, 0.175, 0.0], [0.185, 0.18, 0.23]], [[0.165, 0.17, 0.205], [0.185, 0.175, 0.23]], [[-0.165, 0.17, 0.205], [-0.14, 0.175, 0.23]], [[0.17, 0.16, 0.205], [0.19, 0.17, 0.23]], [[-0.165, 0.165, 0.205], [-0.145, 0.17, 0.23]], [[-0.165, -0.055, 0.225], [-0.15, 0.165, 0.23]], [[0.175, -0.055, 0.225], [0.19, 0.16, 0.23]]]

def object_world_bounds(ctx, item, yaw=None):
    info, _ = models()
    box = np.asarray(info[item['model_key']]['bounds'])
    corners = np.array([[x, y, z] for x in box[:, 0] for y in box[:, 1] for z in box[:, 2]])
    root, rotation = object_frame(ctx, item, yaw)
    points = corners @ rotation.T + root
    return np.array([points.min(0), points.max(0)])


def center(item):
    info, _ = models()
    bounds = np.asarray(info[item['model_key']]['bounds'])
    if 'world_rotation' in item and 'object_root' in item:
        return np.asarray(item['object_root'], float) + np.asarray(
            item['world_rotation'], float) @ bounds.mean(0)
    yaw = float(item['yaw'])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    position = np.asarray(item['position'], float).copy()
    position[:2] += rotation @ bounds.mean(0)[:2]
    return position


def object_frame(ctx, item, yaw=None):
    """Return the complete observed source frame or an upright target frame."""
    angle = float(item['yaw'] if yaw is None else yaw)
    rotation = Rotation.from_euler('z', angle).as_matrix()
    root = np.r_[np.asarray(item.get('object_root',item['position']), float)[:2], ctx.table_z]
    if yaw is None and 'world_rotation' in item:
        rotation = np.asarray(item['world_rotation'], float)
        root = np.asarray(item['object_root'], float)
    return root, rotation


def alarm_grasp_offset(item):
    """Contact the middle casing, away from the narrow decorative upper edge."""
    _, data = models()
    cloud = data[item['model_key']+'_points']
    section = cloud[(np.abs(cloud[:, 0]) < .010) &
                    (cloud[:, 2] > .030) & (cloud[:, 2] < .052)]
    if len(section) < 10:
        raise RuntimeError('Public alarm casing section missing')
    middle = (section.min(0)+section.max(0))*.5
    return np.r_[middle[:2], .043]


def garage_grasp_geometry(item):
    """Return a public head-shell section, above shoulders and the neck."""
    _, data = models()
    cloud = data[item['model_key']+'_points']
    section = cloud[(cloud[:, 2] > item['height']*.83) &
                    (cloud[:, 2] < item['height']*.95)]
    if len(section) < 8:
        raise RuntimeError('Public figurine head-shell section missing')
    return section, (section.min(0)+section.max(0))*.5


def measure_carried_garage(ctx, item, expected_xy, yaw_hint=0.0):
    """Fit the held object's base and tilt from two calibrated RGB views.

    The head camera uses only the exposed lower silhouette. Neutral robot
    pixels censor the wrist silhouette where fingers hide the public CAD.
    The public desk plane bounds the lowest point; no simulator pose is used.
    """
    _, data = models()
    cloud = np.asarray(data[item['model_key']+'_points'], float)
    expected_xy = np.asarray(expected_xy, float)
    expected = np.r_[expected_xy, ctx.table_z+.001]
    rz = Rotation.from_euler('z', yaw_hint).as_matrix()
    reference = cloud @ rz.T+expected
    views, diagnostics = [], {}
    for camera in ('cam_left_wrist', 'cam_head'):
        rgb = image(ctx, camera)
        height, width = rgb.shape[:2]
        uv = project_points(ctx, reference, camera)
        low = np.maximum(uv.min(0)-50, 0).astype(int)
        high = np.minimum(uv.max(0)+50, [width-1, height-1]).astype(int)
        if camera == 'cam_head':
            lower = reference[cloud[:, 2] < min(.10, item['height']*.60)]
            low[1] = max(0, int(project_points(ctx, lower, camera)[:, 1].min()-3))
        if np.any(high <= low):
            raise RuntimeError('Held figurine RGB search is outside the camera')
        roi = np.zeros(rgb.shape[:2], bool)
        roi[low[1]:high[1]+1, low[0]:high[0]+1] = True
        masks = color_masks(rgb)
        seed = (masks['green'] | masks['yellow']) & roi
        ys, xs = np.where(seed)
        if len(xs) < (100 if camera == 'cam_left_wrist' else 15):
            raise RuntimeError('Held figurine color landmarks are missing in '+camera)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        margin_x = max(6, int((x1-x0)*.10))
        margin_down = max(14, int((y1-y0)*.18))
        labels = np.zeros(rgb.shape[:2], np.uint8)
        labels[max(0, y0-6):min(height, y1+margin_down),
               max(0, x0-margin_x):min(width, x1+margin_x)] = cv2.GC_PR_FGD
        labels[seed] = cv2.GC_FGD
        cv2.grabCut(rgb, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    5, cv2.GC_INIT_WITH_MASK)
        foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) & roi
        ys, xs = np.where(foreground)
        polygon = cv2.convexHull(np.column_stack((xs, ys)).astype(np.float32)).reshape(-1, 2)
        observed = boundary_samples(polygon, 80)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        occluded = ((hsv[..., 1] < 40) | (hsv[..., 2] < 70)).astype(np.float32)
        occluded = cv2.GaussianBlur(occluded, (3, 3), .7)
        def visibility(pixels, field=occluded):
            hidden = cv2.remap(field, pixels[:, 0].astype(np.float32).reshape(-1, 1),
                               pixels[:, 1].astype(np.float32).reshape(-1, 1),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=1).ravel()
            return 1.-hidden
        weights = visibility(observed)
        rectangle = np.array([[low[0], low[1]], [high[0], low[1]],
                              [high[0], high[1]], [low[0], high[1]]], np.float32)
        views.append((camera, polygon, observed, weights, visibility, rectangle))
        diagnostics[camera] = dict(bbox=list(cv2.boundingRect(polygon)),
                                   foreground_pixels=int(foreground.sum()),
                                   visible_boundary_samples=float(weights.sum()))

    def world_cloud(values):
        rotation = Rotation.from_euler('xyz', values[3:]).as_matrix()
        local = cloud @ rotation.T
        position = np.r_[values[:2], ctx.table_z+values[2]-local[:, 2].min()]
        return local+position, rotation, position

    def residual(values):
        points, _, _ = world_cloud(values)
        result = []
        for camera, polygon, observed, weights, visibility, rectangle in views:
            uv = project_points(ctx, points, camera)
            predicted = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            _, clipped = cv2.intersectConvexConvex(predicted, rectangle)
            if clipped is not None and len(clipped) >= 3:
                predicted = clipped.reshape(-1, 2)
            sampled = boundary_samples(predicted, 80)
            result.extend((line_distance(observed, predicted)*weights,
                           line_distance(sampled, polygon)*visibility(sampled)))
        result.append(np.array([values[2]*200., values[3]*2., values[4]*2.]))
        return np.concatenate(result)

    candidates = []
    for angle in yaw_hint+np.array([0., np.pi/2, np.pi, -np.pi/2]):
        initial = np.r_[expected_xy, .001, 0., 0., angle]
        fitted = least_squares(residual, initial,
            bounds=(np.r_[expected_xy-.08, 0., -.6, -.6, angle-1.3],
                    np.r_[expected_xy+.08, .012, .6, .6, angle+1.3]),
            diff_step=1e-3, loss='soft_l1', f_scale=1.5, max_nfev=160)
        rms = float(np.sqrt(np.mean(fitted.fun**2)))
        candidates.append((rms, fitted.x))
    rms, values = min(candidates, key=lambda candidate: candidate[0])
    points, rotation, position = world_cloud(values)
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
    if rms > 3.0 or np.linalg.norm(position[:2]-expected_xy) > .065 or tilt > .40:
        raise RuntimeError(f'Held figurine CAD fit is uncertain: rms={rms:.2f}, tilt={tilt:.3f}')
    return dict(position=position, rotation=rotation, yaw=float(values[5]),
                tilt=tilt, pixel_rms=rms, min_z=float(points[:, 2].min()), views=diagnostics)


def attached_object_region(ctx, item, point, quat, arm):
    """Project the public shape where an attached object must appear.

    This uses the RGB fit before grasping and the measured wrist pose. An
    object that stayed on the desk must not count as an attached object merely
    because it remains visible somewhere in the wrist camera.
    """
    _, data = models()
    cloud = np.asarray(data[item['model_key']+'_points'], float)
    root, rotation = object_frame(ctx, item)
    original = cloud @ rotation.T + root
    relative = (original-np.asarray(point)) @ quat_matrix(quat)
    pose = np.asarray(ctx.state[f'{arm}_ee_pose'], float)
    carried = relative @ quat_matrix(pose[3:]).T + ctx.tcp(arm)
    camera = f'cam_{arm}_wrist'
    rgb = image(ctx, camera)
    origin, camera_rotation, _, _ = ctx.perception.camera_geometry(camera, ctx.state, rgb.shape)
    local = (carried-origin) @ camera_rotation
    carried = carried[local[:, 2] < -.015]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    if len(carried) < 4:
        return mask > 0
    pixels = project_points(ctx, carried, camera)
    pixels = pixels[np.isfinite(pixels).all(1) & (np.abs(pixels) < 5000).all(1)]
    if len(pixels) >= 4:
        polygon = cv2.convexHull(pixels.astype(np.float32))
        cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    mask[:15] = mask[-15:] = 0
    mask[:, :15] = mask[:, -15:] = 0
    return mask > 0


def free_alarm_buffer(ctx, alarm, measured_scene, return_candidates=False):
    """Select a visible wood patch using measured object rectangles."""
    candidates = []
    height = .064
    for x in np.linspace(-.29, -.18, 4):
        for y in (-.285, -.255, -.225):
            point = np.array([x, y, ctx.table_z+height])
            footprint = region_mask(ctx, [x-.050, y-.040, x+.050, y+.040], z=ctx.table_z) > 0
            rgb = image(ctx)
            pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[footprint]
            if not len(pixels):
                continue
            wood = (pixels[:, 0] > 5) & (pixels[:, 0] < 26) & (pixels[:, 1] > 75) & (pixels[:, 2] > 45)
            ratio = float(wood.mean())
            if ratio < .82:
                continue
            blocked = False
            for kind, item in measured_scene.items():
                if not isinstance(item, dict) or kind in ('alarm', 'drawer'):
                    continue
                size = np.asarray(item['size'])[:2]
                distance = np.abs(point[:2]-center(item)[:2])
                if np.all(distance < size/2+[.055, .045]):
                    blocked = True
                    break
            if not blocked:
                candidates.append((ratio-.15*np.linalg.norm(point[:2]-alarm['position'][:2]), point))
    if not candidates:
        raise RuntimeError('No visually clear tabletop buffer for alarm')
    ordered = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
    return [p for _, p in ordered] if return_candidates else ordered[0][1]


def _clip_source_polygon(points, width, height):
    polygon = np.asarray(points, float).reshape(-1, 2).copy()
    for axis, boundary, sign in ((0, 0., 1), (0, width-1., -1),
                                  (1, 0., 1), (1, height-1., -1)):
        fa = (polygon[:, axis]-boundary)*sign
        inside = fa >= 0
        if inside.all():
            if len(polygon) < 3: return None
            continue
        second = np.roll(polygon, -1, axis=0)
        fb = (second[:, axis]-boundary)*sign
        crossing = inside != (fb >= 0)
        # The two slots retain the original per-edge append order: first,
        # then intersection. Keep subtract/multiply/divide/add separate.
        result = np.empty((len(polygon), 2, 2), dtype=float)
        result[:, 0] = polygon
        first = polygon[crossing]
        result[crossing, 1] = (first+(second[crossing]-first)*fa[crossing, None]/
                              (fa[crossing]-fb[crossing])[:, None])
        polygon = result[np.column_stack((inside, crossing))]
        if len(polygon) < 3: return None
    return polygon



def _source_contour(ctx, camera, measured=None, hint=None):
    rgb = image(ctx, camera)
    height, width = rgb.shape[:2]
    hh, ss, vv = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    rr, gg, bb = cv2.split(rgb.astype(float))
    masks = color_masks(rgb)
    workspace = np.zeros(rgb.shape[:2], bool)
    if camera == 'cam_head':
        for z in (.015, .095, .175):
            workspace |= region_mask(ctx, [-.55, -.34, .09, .235], z=ctx.table_z+z) > 0
        workspace[:140] = False
    else:
        workspace[:] = True
    excluded = np.zeros(rgb.shape[:2], bool)
    if camera == 'cam_head' and measured is not None:
        for key in ('alarm', 'keyboard', 'drawer'):
            item = measured.get(key)
            if item is None or 'bbox' not in item: continue
            if key == 'alarm': excluded |= _alarm_background_mask(ctx, item)
            else:
                x, y, w, h = item['bbox']
                excluded[max(0, y-3):y+h+3, max(0, x-3):x+w+3] = True
    neutral = (ss < 70) & (rr < gg*1.06) & (vv > 45)
    primary = (masks['green'] | masks['yellow']) & workspace & ~excluded
    if camera=='cam_head':
        primary |= neutral & workspace & ~excluded
    # Both broad figures contain green material. Neutral material remains a
    # secondary source seed when green is cropped out of a wrist observation.
    if primary.sum() < 80: primary = neutral & workspace & ~excluded
    cs = contours(primary, area=70, close=3)
    if not cs: return None
    seed = (np.vstack(cs) if camera!='cam_head' else
            max(cs,key=lambda c:cv2.contourArea(c.astype(np.float32))))
    x, y, w, h = cv2.boundingRect(seed.astype(np.float32))
    roi = np.zeros(rgb.shape[:2], bool)
    roi[max(0,y-40):min(height,y+h+50),max(0,x-45):min(width,x+w+45)] = True
    roi &= workspace & ~excluded
    wood = (rr > gg*1.10) & (gg > bb*1.10) & (ss > 25) & (vv > 100)
    labels = np.zeros(rgb.shape[:2], np.uint8)
    labels[roi] = cv2.GC_PR_FGD
    labels[(primary | neutral) & roi] = cv2.GC_FGD
    labels[wood | excluded] = cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(rgb, labels, None, np.zeros((1,65)), np.zeros((1,65)),
                4, cv2.GC_INIT_WITH_MASK)
    cs = contours((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD), area=100, close=3)
    if not cs: return None
    return max(cs, key=lambda c: cv2.contourArea(c.astype(np.float32)))


@lru_cache(maxsize=16)
def _source_public_hull(key):
    # A camera silhouette is unchanged by removing interior points from the
    # public convex hull. Retain the full cloud for the visibility certificate.
    _, data = models()
    points = np.asarray(data[key+'_points'], float)
    return points[ConvexHull(points).vertices]


def _fit_source_rows(ctx, cameras=('cam_head','cam_left_wrist'), measured=None,
                     seeds=None, keys=None):
    info, data = models()
    views, diagnostics = [], {}
    for camera in cameras:
        contour = _source_contour(ctx, camera, measured)
        if contour is None: continue
        polygon = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2)
        rgb = image(ctx, camera)
        geometry = ctx.perception.camera_geometry(camera,ctx.state,rgb.shape)
        views.append((camera,polygon,boundary_samples(polygon,80),geometry,rgb.shape))
        diagnostics[camera] = dict(bbox=list(cv2.boundingRect(polygon)),
             area=float(cv2.contourArea(polygon)))

    if not views: raise RuntimeError('No source silhouette')
    seed_camera, poly, _, _, _ = views[0]
    center_uv = poly.mean(0)
    results=[]
    for key in (keys or [k for k in info if k.startswith('garage_')]):
        midpoint = np.array([0., 0., info[key]['height']/2])
        full_cloud = np.asarray(data[key+'_points'], float)-midpoint
        cloud = _source_public_hull(key)-midpoint
        approx = ctx.perception.project(center_uv, ctx.table_z+info[key]['height']/2,
                                        seed_camera, ctx.state, image(ctx,seed_camera).shape)
        def world(p,points=cloud):
            rot=Rotation.from_euler('xyz',p[2:]).as_matrix()
            pts=points@rot.T; root=np.r_[p[:2],ctx.table_z-pts[:,2].min()]
            return pts+root,root-rot@midpoint,rot
        def pixels(pts,geometry,shape):
            origin,rotation,fx,fy=geometry
            local=(pts-origin)@rotation
            return np.column_stack((shape[1]/2-fx*local[:,0]/local[:,2],
                                    shape[0]/2+fy*local[:,1]/local[:,2]))
        def residual(p):
            pts,_,_=world(p); out=[]
            for camera, polygon, observed,geometry,shape in views:
                uv=pixels(pts,geometry,shape)
                pred=cv2.convexHull(uv.astype(np.float32)).reshape(-1,2)
                pred=_clip_source_polygon(pred,shape[1],shape[0])
                if pred is None:
                    out.append(np.full(160,1000.));continue
                out.extend((line_distance(observed,pred),
                            line_distance(boundary_samples(pred,80),polygon)))
            return np.r_[tuple(out)]
        starts = [np.r_[approx[:2],0.,0.,yaw] for yaw in (0.,math.pi/2,math.pi,3*math.pi/2)]
        head = next((v for v in views if v[0]=='cam_head'), None)
        if head is not None:
            coarse = fit_model(ctx,head[1],key)
            if coarse is not None:
                starts.append(np.r_[coarse['position'][:2],0.,0.,coarse['yaw']])
        if seeds:
            for seed in seeds:
                if seed['model_key'] != key: continue
                rotation = Rotation.from_euler('xyz',seed['euler_xyz']).as_matrix()
                middle = np.asarray(seed['root'])+rotation@midpoint
                starts.append(np.r_[middle[:2],seed['euler_xyz']])
        for start in starts:
            start[:2] = np.clip(start[:2],approx[:2]-.159,approx[:2]+.159)
            f=least_squares(residual,start,
              bounds=(np.r_[approx[:2]-.16,-.90,-.90,start[-1]-1.4],
                      np.r_[approx[:2]+.16,.90,.90,start[-1]+1.4]),
              diff_step=.001,loss='soft_l1',f_scale=1.5,max_nfev=180)
            pts,root,rot=world(f.x)
            row=dict(model_key=key,rms=float(np.sqrt(np.mean(f.fun**2))),
                root=root.tolist(),euler_xyz=f.x[2:].tolist(),rotation=rot.tolist(),
                nfev=f.nfev,view_rms={},visibility={})
            full_points,_,_=world(f.x,full_cloud)
            for i,(camera,_,_,geometry,shape) in enumerate(views):
                uv=pixels(full_points,geometry,shape)
                row['view_rms'][camera]=float(np.sqrt(np.mean(f.fun[i*160:(i+1)*160]**2)))
                row['visibility'][camera]=float(((uv[:,0]>0)&(uv[:,0]<shape[1]-1)&
                                               (uv[:,1]>0)&(uv[:,1]<shape[0]-1)).mean())
            results.append(row)

    return dict(views=diagnostics,fits=results,best=min(results,key=lambda r:r['rms']))


def garage_source_hint(ctx, measured):
    """Trigger complete source recovery from a clipped or uncertain RGB fit."""
    contour = _source_contour(ctx, 'cam_head', measured)
    if contour is None:
        return None
    x, y, width, height = cv2.boundingRect(contour.astype(np.float32))
    clipped = x <= 1 or x+width >= image(ctx).shape[1]-1
    item = measured.get('garage')
    uncertain_broad = width>95 and (item is None or item['fit_error']>3.)
    if not clipped and not uncertain_broad:
        return None
    mask = np.zeros(image(ctx).shape[:2], np.uint8)
    cv2.fillPoly(mask,[contour.astype(np.int32)],1)
    rgb = image(ctx); h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    gray = float((((s<85)&(v>70))&(mask>0)).sum()/max(1,mask.sum()))
    green = float((color_masks(rgb)['green']&(mask>0)).sum()/max(1,mask.sum()))
    keys = ('garage_00005','garage_00007') if width>95 and (gray>.35 or green>.50) else None
    # The public textured armor projects a predominantly neutral shell in
    # this view. Use conservative material separation only for obstacle identity;
    # current grasp identity is still fitted against both complete CAD models.
    obstacle_keys = (('garage_00005',) if keys is not None and gray>.52 and green<.35
                     else keys)
    return dict(bbox=[x,y,width,height], gray_fraction=gray, green_fraction=green,
                obstacle_keys=obstacle_keys,
                keys=keys, trigger='observed_source_boundary_clipped' if clipped else
                                   'observed_source_pose_uncertain')


def source_pose_hypotheses(ctx, measured, hint):
    """Keep plausible full CAD frames as obstacles, never as grasp poses."""
    result = _fit_source_rows(ctx, ('cam_head',), measured,
                              keys=hint.get('obstacle_keys',hint['keys']))
    rows = sorted(result['fits'], key=lambda r:r['rms'])
    good = [r for r in rows if r['rms']<4.0]
    if not good:
        raise RuntimeError('Clipped source pose hypotheses are uncertain')
    selected=[]
    for row in good:
        if any(row['model_key']==prior['model_key'] and
               np.linalg.norm(np.asarray(row['root'])-prior['root'])<.003 and
               Rotation.from_matrix(np.asarray(row['rotation'])@np.asarray(prior['rotation']).T).magnitude()<.03
               for prior in selected):
            continue
        selected.append(row)
    return [_source_item(ctx,row,result['views']) for row in selected], result


def _source_item(ctx, fitted, views):
    info, _ = models(); key=fitted['model_key']; model=info[key]
    root, rotation=np.asarray(fitted['root']),np.asarray(fitted['rotation'])
    bounds=np.asarray(model['bounds'])
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    world=corners@rotation.T+root
    item=dict(kind='garage', model_key=key, object_root=root,
        world_rotation=rotation, position=root+rotation@[0.,0.,model['height']/2],
        height=model['height'], yaw=float(fitted['euler_xyz'][2]),
        size=world.max(0)-world.min(0), fit_error=fitted['rms'],
        source_views=views, source_view_rms=fitted['view_rms'],
        source_visibility=fitted['visibility'])
    if 'cam_head' in views:
        item['bbox']=views['cam_head']['bbox']; item['camera']='cam_head'
    return item


def measure_source_garage(ctx, measured, hint, hypotheses=(), required=True):
    """Fit identity and full current source pose from synchronized RGB views."""
    seeds=[]
    for item in hypotheses:
        seeds.append(dict(model_key=item['model_key'], root=item['object_root'],
            euler_xyz=Rotation.from_matrix(item['world_rotation']).as_euler('xyz')))
    result=_fit_source_rows(ctx, measured=measured, keys=hint['keys'], seeds=seeds)
    rows=sorted(result['fits'], key=lambda row:row['rms']); best=rows[0]
    alternative=min([r['rms'] for r in rows if r['model_key']!=best['model_key']],default=float('inf'))
    tilt=Rotation.from_matrix(best['rotation']).as_euler('xyz')[:2]
    wrist=best['view_rms'].get('cam_left_wrist',float('inf'))
    visible=best['visibility'].get('cam_left_wrist',0.)
    ambiguous=any(r['model_key']==best['model_key'] and r['rms']<best['rms']+.50 and
        (np.linalg.norm(np.asarray(r['root'])-best['root'])>.005 or
         Rotation.from_matrix(np.asarray(r['rotation'])@np.asarray(best['rotation']).T).magnitude()>.08)
        for r in rows[1:])
    valid=(best['rms']<3.0 and wrist<2.0 and visible>.25 and
           alternative-best['rms']>1.25 and np.linalg.norm(tilt)<.40 and not ambiguous)
    result.update(valid=bool(valid), identity_margin=float(alternative-best['rms']),
                  ambiguous_pose=bool(ambiguous))
    if not valid:
        if required:
            raise RuntimeError(f'Current figurine source CAD is uncertain: rms={best["rms"]:.2f}, wrist={wrist:.2f}, visible={visible:.3f}, identity_margin={alternative-best["rms"]:.2f}')
        return None,result
    item=_source_item(ctx,best,result['views']);item['source_pose_validated']=True
    return item,result
