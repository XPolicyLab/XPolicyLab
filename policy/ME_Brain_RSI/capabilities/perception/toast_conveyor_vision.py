"""RGB measurements for toast insertion and delayed conveyor matching.

Coordinates below describe the public object assets, never an episode layout.
The scene transforms are estimated from the currently observed RGB pixels.
"""
from dataclasses import dataclass
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


BELT_Z = .718


def _learned_toast_mask(ctx,label,camera):
    if ctx.task_name not in ('make_toast_random', 'make_toast') or not os.environ.get('COAP_PUBLIC_RGB_MODELS'):
        return None
    from .containers_vision import _learned_pack_masks
    proposals=_learned_pack_masks(ctx,label,camera)
    if label=='toaster':
        # The detector's top box is sometimes another appliance-like object; the public task
        # randomises the toaster inside x [0.04, 0.35], y [-0.12, -0.03]: take the first
        # proposal whose mask centroid projects into that band (with slack), trying other
        # prompts when the plain one only found clutter or nothing at all (a toaster beside
        # the parked right gripper got no 'toaster' box, but a 'white toaster' one).
        rgb=_toast_rgb(ctx,camera)
        def in_range(proposal):
            ys,xs=np.nonzero(proposal['best_mask'])
            if len(xs)<200:return False
            pixel=np.array([xs.mean(),ys.mean()])
            point=np.asarray(ctx.perception.project(pixel,ctx.table_z+.10,camera,ctx.state,rgb.shape),float)
            return -.03<point[0]<.45 and -.22<point[1]<.08
        for query in (None,'white toaster','kitchen toaster appliance','bread toaster'):
            found=proposals if query is None else _learned_pack_masks(ctx,label,camera,query=query)
            for proposal in found:
                if in_range(proposal):
                    if query is not None and not proposals:
                        from ..executor import _log
                        events=ctx.memory.get('task_events',[])
                        if not any(e.get('event')=='toaster_prompt_fallback' and e.get('step')==ctx.steps for e in events):
                            _log(ctx,'toaster_prompt_fallback',query=query,score=round(float(proposal['score']),3))
                    return proposal['best_mask']
        return proposals[0]['best_mask'] if proposals else None
    if not proposals:return None
    if label=='bread':
        return np.logical_or.reduce([p['best_mask'] for p in proposals])
    return proposals[0]['best_mask']


def _toast_rgb(ctx,camera):
    rgb=np.asarray(ctx.vision[camera]['color'],np.uint8)[...,:3]
    gain=ctx.memory.get('toast_white_gain') if ctx.task_name=='make_toast_random' else None
    if gain is None:return rgb
    return np.clip(rgb.astype(float)*np.asarray(gain),0,255).astype(np.uint8)


def head_image(ctx):
    name = next((name for name in ctx.vision if "head" in name), next(iter(ctx.vision)))
    return _toast_rgb(ctx,name), name


def world_pixels(ctx, xyz, image_shape, camera="cam_head"):
    return np.asarray([ctx.perception.world_to_pixel(point, camera, ctx.state, image_shape)
                       for point in np.asarray(xyz, float).reshape(-1, 3)])


def world_roi(ctx, image_shape, bounds, z, camera="cam_head"):
    xmin, xmax, ymin, ymax = bounds
    polygon = world_pixels(ctx, [[xmin, ymin, z], [xmax, ymin, z],
                                 [xmax, ymax, z], [xmin, ymax, z]], image_shape, camera)
    mask = np.zeros(image_shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    if tuple(bounds)==(-.01,.48,-.30,.18) and 'head' in camera:
        learned=_learned_toast_mask(ctx,'toaster',camera)
        if learned is not None:
            pieces,_=cv2.findContours(learned.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if pieces:
                support=np.zeros(image_shape[:2],np.uint8)
                cv2.fillConvexPoly(support,cv2.convexHull(np.concatenate(pieces)),255)
                mask&=cv2.dilate(support,np.ones((5,5),np.uint8))
    return mask


def contour_record(ctx, image, contour, z, camera):
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    pixel = np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    points = contour[:, 0, :].astype(float)
    try:
        world = np.asarray([ctx.perception.project(p, z, camera, ctx.state, image.shape) for p in points])
        position = ctx.perception.project(pixel, z, camera, ctx.state, image.shape)
    except ValueError:
        return None
    rect = cv2.minAreaRect(world[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    index = int(np.argmax(np.linalg.norm(edges, axis=1)))
    axis = edges[index] / max(np.linalg.norm(edges[index]), 1e-8)
    length = float(max(rect[1]))
    width = float(min(rect[1]))
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return dict(position=position, pixel=pixel, yaw=math.atan2(axis[1], axis[0]),
                length=length, width=width, area=cv2.contourArea(contour),
                bbox=(x, y, w, h), contour=points, world_contour=world,
                mask=mask, crop=image[y:y + h, x:x + w].copy(), camera=camera,
                mean_rgb=image[mask > 0].mean(axis=0))


@dataclass
class ToastScene:
    toaster: np.ndarray
    toaster_yaw: float
    variant: int
    slots: list
    lever: np.ndarray
    breads: list
    rack_yaw: float
    confidence: float


def toast_body_asset(variant):
    """Public body and slot geometry; the pose is always estimated from RGB."""
    if variant==2:
        return dict(footprint=(-.127079,.119925,-.082755,.078748),roof=.081116,
                    slots=[(-.074713,.058699,-.043804,-.009604),
                           (-.074713,.058699,.005596,.039797)],
                    floor=-.061386,half_length=.067,half_width=.0185)
    if variant==4:
        return dict(footprint=(-.120400,.106340,-.062489,.062445),roof=.078564,
                    slots=[(-.068653,.054593,-.026707,-.007348),
                           (-.068653,.054593,.007304,.026663)],
                    floor=-.030893,half_length=.0616,half_width=.00965)
    if variant == 0:
        return dict(footprint=(-.105056, .102198, -.078268, .077172), roof=.080233,
                    slots=[(-.079149, .076291, -.039408, -.013502),
                           (-.079149, .076291, .012405, .038312)],
                    floor=-.049301, half_length=.077720, half_width=.012953)
    if variant == 1:
        return dict(footprint=(-.110779, .115714, -.070570, .069101), roof=.075138,
                    slots=[(-.057931, .062866, -.047920, -.014324),
                           (-.057931, .062866, .012855, .046452)],
                    floor=-.019234, half_length=.060399, half_width=.016798)
    raise ValueError(f'Unsupported toaster variant: {variant}')


def toast_lever_asset(variant):
    """Public USD slider collision box, support height and prismatic limits."""
    if variant==2:
        # Public link_1 mesh has authored q=-.085m; undo it to obtain q=0.
        return dict(x0=.115175,x1=.142982,y0=-.016254,y1=.012247,
                    z0=-.077060,z1=-.055685,support=.0948,
                    lower=-.0852,upper=.0012,tcp_x=.139,
                    pitch=math.pi/4,tip_depth=.012523,target=-.0020)
    if variant==4:
        # Public link_0 mesh has authored q=.002841776m, positive downward.
        return dict(x0=.051637,x1=.126308,y0=-.034361,y1=.034316,
                    z0=-.019121,z1=.036785,support=.0859,
                    lower=0.,upper=.0432,tcp_x=.122,
                    pitch=math.pi/4,tip_depth=.012523,target=.041)
    if variant == 0:
        # The authored USD has q=.000642617; these Z bounds undo that offset.
        return dict(x0=.102198, x1=.115152, y0=-.013502, y1=.012405,
                    z0=.028419, z1=.041373, support=.0818,
                    lower=-.0138, upper=.0642, tcp_x=.1205,
                    pitch=math.radians(25), tip_depth=.009971, target=.0588)
    return dict(x0=.115735, x1=.138363, y0=-.013569, y1=.012100,
                z0=.005769, z1=.011808, support=.0759,
                lower=-.0234, upper=.0516, tcp_x=.1370,
                pitch=math.pi / 4, tip_depth=.012523, target=.0465)


def measure_toast_lever(ctx, scene, reference=None):
    """Fit the visible slider's public box to wrist RGB, including its height.

    The fitted XY translation is independent of the earlier roof estimate.
    Positive q moves down. Its datum is the public support height above the
    known table, not an observed simulator joint or object pose.
    """
    arm = ctx.arm_for(scene.lever)
    camera = next((n for n in ctx.vision if arm in n and "wrist" in n), None)
    if camera is None:
        return None
    image = _toast_rgb(ctx,camera)
    asset = toast_lever_asset(scene.variant)
    yaw = scene.toaster_yaw
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw), 0],
                           [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    root = np.r_[scene.toaster[:2], scene.toaster[2] if scene.variant in (2,4)
                 else ctx.table_z + asset["support"]]
    if reference is not None:
        root[:2] = np.asarray(reference["root"])[0:2]
    box = np.asarray([[x, y, z] for x in (asset["x0"], asset["x1"])
                      for y in (asset["y0"], asset["y1"])
                      for z in (asset["z0"], asset["z1"])])

    def projected(parameters):
        points = box.copy()
        points[:, 2] -= parameters[2]
        points = points @ rotation.T + root + [parameters[0], parameters[1], 0]
        return world_pixels(ctx, points, image.shape, camera)

    extent = np.r_[projected([0, 0, asset["lower"]]),
                   projected([0, 0, asset["upper"]])]
    height, width = image.shape[:2]
    if (extent[:, 0].min() < 4 or extent[:, 0].max() > width - 4 or
            extent[:, 1].min() < 4 or extent[:, 1].max() > height - 4):
        return None
    lower_pixel, upper_pixel = extent.min(0) - [40, 45], extent.max(0) + [55, 45]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = ((hsv[..., 1] < 40) & (hsv[..., 2] > 88) &
            (hsv[..., 2] < 245)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c[:, 0] for c in contours if cv2.contourArea(c) > 20]
    if not contours:
        return None
    observed = np.concatenate(contours).astype(float)
    observed = observed[((observed > lower_pixel) & (observed < upper_pixel)).all(1)]
    if len(observed) < 40:
        return None
    tree = cKDTree(observed)

    def perimeter(parameters):
        pixels = projected(parameters)
        hull = cv2.convexHull(pixels.astype(np.float32), returnPoints=False).ravel()
        segments = []
        for first, second in zip(hull, np.roll(hull, -1)):
            # The rear face joins the body, so its outline can be occluded.
            if box[first, 0] == asset["x0"] and box[second, 0] == asset["x0"]:
                continue
            segments.append((pixels[first], pixels[second]))
        segments = np.asarray(segments)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        cumulative = np.r_[0., np.cumsum(lengths)]
        # Hull topology can change during optimisation. Uniform arc-length
        # samples keep the residual dimension fixed for every candidate.
        distances = np.linspace(0, cumulative[-1], 128, endpoint=False)
        indices = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"),
                             len(segments) - 1)
        fractions = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-8)
        return segments[indices, 0] + fractions[:, None] * (segments[indices, 1] - segments[indices, 0])

    def residual(parameters):
        return np.r_[tree.query(perimeter(parameters))[0], parameters[:2] / .04]

    shift_limit = .025 if reference is None else .006
    choices = []
    for seed in (-.01, .01, .032, .045):
        result = least_squares(residual, [0, 0, np.clip(seed, asset["lower"], asset["upper"])],
                               bounds=([-shift_limit, -shift_limit, asset["lower"]],
                                       [shift_limit, shift_limit, asset["upper"]]),
                               loss="soft_l1", f_scale=2, max_nfev=100, diff_step=.003)
        errors = tree.query(perimeter(result.x))[0]
        choices.append((float(errors.mean()), result.x, errors))
    score, parameters, errors = min(choices, key=lambda row: row[0])
    pixels = projected(parameters)
    hull = cv2.convexHull(pixels.astype(np.float32))[:, 0]
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 255)
    area = int((mask > 0).sum())
    coverage = float((gray[mask > 0] > 0).mean()) if area else 0.
    if score > 2.5 or np.quantile(errors, .9) > 4 or area < 80 or coverage < .70:
        return None
    fitted_root = root + [parameters[0], parameters[1], 0]
    q = float(parameters[2])
    centre = np.r_[[(asset["x0"] + asset["x1"]) / 2,
                    (asset["y0"] + asset["y1"]) / 2], asset["z1"] - q]
    return dict(root=fitted_root, position=fitted_root + rotation @ centre,
                joint_q=q, joint_fraction=(q - asset["lower"]) / (asset["upper"] - asset["lower"]),
                pixel_error=score, coverage=coverage, camera=camera,
                pixel=ctx.perception.world_to_pixel(fitted_root + rotation @ centre,
                                                    camera, ctx.state, image.shape))


# Public bread/00000 mesh's planar convex profile (metres), simplified with a
# 0.6 mm tolerance. Its 11.8 mm thickness comes from the public aligned box.
_BREAD_PROFILE = np.asarray([
    [-.057061, .048389], [-.058369, .043446], [-.051044, -.049677],
    [-.050195, -.053266], [-.047891, -.056785], [-.042111, -.058227],
    [.052095, -.050249], [.057162, -.048330], [.058412, -.045497],
    [.050975, .050899], [.048791, .056522], [.043651, .057768],
    [-.053815, .050343],
])


def centre_source_grasp(ctx, contact, arm="left", shelf_end=False, prefer_rack_end=False):
    """Align a safe, open top pinch to the visible gold edge near its pads.

    The RGB measurement is made above the rack. Corrections follow the
    observed jaw axis, with scale supplied by public wrist camera geometry.
    """
    from ..motion import quat_matrix
    camera = next((name for name in ctx.vision if arm in name and "wrist" in name), None)
    if camera is None:
        return None
    image = _toast_rgb(ctx,camera)
    frame = quat_matrix(ctx.state[f"{arm}_ee_pose"][3:])
    pad = np.asarray(contact, float) + frame[:, 0] * .005
    pixel = ctx.perception.world_to_pixel(pad, camera, ctx.state, image.shape)
    shifted = ctx.perception.world_to_pixel(pad + frame[:, 1] * .010,
                                            camera, ctx.state, image.shape)
    gradient = float((shifted[0] - pixel[0]) / .010)
    row = int(round(pixel[1]))
    if abs(gradient) < 300 or not 3 <= row < image.shape[0] - 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if shelf_end:
        # A single keypad rectangle can mimic a pinch edge. Require the
        # observed parallel slice planes to agree with public 25mm seats.
        expected_width=abs(gradient)*.0118
        count=int(ctx.memory.get('toast_direct_rack_count',4));count=min(4,max(3,count))
        groups=[]
        for saturation_floor in (30,15):
            mask=((hsv[...,0]>=14)&(hsv[...,0]<42)&(hsv[...,1]>saturation_floor)&(hsv[...,2]>115))
            for sample_row in (row,row-20,row-40,row-60,row-80):
                if not 3<=sample_row<image.shape[0]-3:continue
                probability=mask[sample_row-2:sample_row+3].mean(axis=0)
                line=cv2.morphologyEx((probability>.4).astype(np.uint8)[None,:],
                                      cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))[0]
                columns=np.flatnonzero(line);runs=np.split(columns,np.flatnonzero(np.diff(columns)>1)+1)
                runs=[run for run in runs if max(9,expected_width*.42)<=len(run)<=expected_width*2.2]
                for start in range(len(runs)-count+1):
                    group=runs[start:start+count];centres=[]
                    for run in group:
                        u=float((run[0]+run[-1])/2)
                        # The fingers can cover the contact row. Extrapolate
                        # exposed upright edges through the public camera,
                        # rather than treating another image row as that plane.
                        point=ctx.perception.project([u,sample_row],pad[2],camera,ctx.state,image.shape)
                        up=ctx.perception.world_to_pixel(point+[0,0,.01],camera,ctx.state,image.shape)
                        if abs(up[1]-sample_row)<.5:break
                        centres.append(u+(up[0]-u)*(row-sample_row)/(up[1]-sample_row))
                    if len(centres)!=count:continue
                    centres=np.asarray(centres);spacing=np.diff(centres)/abs(gradient)
                    error=float(np.max(abs(spacing-.025)))
                    if error>.025/2-.0025:continue
                    stripe_support=[float(mask[max(0,sample_row-24):min(image.shape[0],sample_row+25),
                                               run[0]:run[-1]+1].mean()) for run in group]
                    if min(stripe_support)<.40:continue
                    groups.append((error,group,centres,stripe_support,sample_row))
        if not groups:return None
        consistent=[g for g in groups if any(abs(g[4]-other[4])>=20 and
                    np.max(abs(g[2]-other[2]))<max(4,abs(gradient)*.002)
                    for other in groups)]
        at_contact=[g for g in groups if g[4]==row]
        chosen=at_contact or consistent
        if not chosen:return None
        error,group,centres,stripe_support,sample_row=min(chosen,key=lambda g:g[0])
        rack=ctx.memory.get('toast_geometry',{}).get('rack_yaw',0.)
        row_axis=np.array([math.cos(rack),math.sin(rack),0.])
        direction=float(frame[:,1]@row_axis)
        correction=(centres-pixel[0])/gradient
        index=int(np.argmax(correction*direction))
        raw=float(correction[index])
        if abs(raw)>.10:return None
        run=group[index];bounded=float(np.clip(raw,-.010,.010))
        ctx.memory['toast_wrist_shelf']=dict(count=count,spacing_error=error,
             raw_correction=raw,stripe_support=stripe_support,endpoint_pixel=float(centres[index]),
             sample_row=sample_row,contact_row=row)
        return dict(contact=np.asarray(contact,float)+frame[:,1]*bounded,
                    correction=bounded,raw_correction=raw,pixel=pixel,
                    centre=float(centres[index]),run=(int(run[0]),int(run[-1])))
    # Gold crust remains visible under the approaching hand's shadow. The
    # saturated condition excludes the grey rack and the black fingers.
    mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
            (hsv[..., 1] > 50) & (hsv[..., 2] > 115))
    probability = mask[row - 2:row + 3].mean(axis=0)
    line = cv2.morphologyEx((probability > .4).astype(np.uint8)[None, :],
                            cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0]
    columns = np.flatnonzero(line)
    runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    expected_width = abs(gradient) * .0118
    choices = []
    for run in runs:
        if len(run) < max(9, expected_width * .45) or len(run) > expected_width * 2.2:
            continue
        centre = float((run[0] + run[-1]) / 2)
        correction = (centre - pixel[0]) / gradient
        if abs(correction) < .010:
            choices.append((abs(correction), correction, centre, (int(run[0]), int(run[-1]))))
    if not choices:
        return None
    if prefer_rack_end:
        rack_yaw = ctx.memory.get('toast_geometry', {}).get('rack_yaw')
        if rack_yaw is None:
            return None
        row_axis = np.array([math.cos(rack_yaw), math.sin(rack_yaw), 0.])
        direction = float(frame[:, 1] @ row_axis)
        if abs(direction) < .5:
            return None
        # The task selects the outermost remaining slice. A nearest-strip
        # rule can alternate between its face and the adjacent slice; retain
        # the endpoint identity using the measured rack direction instead.
        _, correction, centre, run = max(choices, key=lambda choice: choice[1] * direction)
    else:
        _, correction, centre, run = min(choices)
    # The next image closes the residual; a single uncertain strip cannot
    # move the hand by more than 6 mm into the neighbouring slice's plane.
    correction = float(np.clip(correction, -.006, .006))
    return dict(contact=np.asarray(contact, float) + frame[:, 1] * correction,
                correction=correction, pixel=pixel, centre=centre, run=run)


def held_bread_view(ctx, arm="left"):
    """Record a visible bread silhouette with the measured donor/camera pose."""
    from ..motion import quat_matrix
    image, camera = head_image(ctx)
    tcp = ctx.tcp(arm)
    frame = quat_matrix(ctx.state[f"{arm}_ee_pose"][3:])
    predicted = ctx.perception.world_to_pixel(tcp + frame[:, 0] * .055,
                                              camera, ctx.state, image.shape)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
            (hsv[..., 1] > 4) & (hsv[..., 1] < 150) &
            (hsv[..., 2] > 185)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if 50 < cv2.contourArea(c) < 8000]
    if not contours:
        return None
    contour = min(contours, key=lambda c: np.linalg.norm(c[:, 0, :].mean(axis=0) - predicted))
    if np.linalg.norm(contour[:, 0, :].mean(axis=0) - predicted) > 60:
        return None
    observed = contour[:, 0, :].astype(float)
    observed = observed[::max(1, len(observed) // 120)]
    origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
    return dict(tcp=tcp, frame=frame, pixels=observed, origin=origin,
                rotation=rotation, fx=fx, fy=fy, centre=np.asarray(image.shape[1::-1]) / 2)


# A secondary load check compares a real exposed bread template with held
# and world-stationary motion hypotheses. Public camera geometry bounds4mm
# relative contact compliance; constrained affine registration handles small
# perspective changes. Development RGB pairs calibrated correlation>=.85 and
# held advantage>=.10. Native task success must be established separately.
def _direct_template_reference(ctx,arm,centre,identity_mask):
 from ..motion import quat_matrix
 if identity_mask is None:return None
 camera=f'cam_{arm}_wrist';rgb=_toast_rgb(ctx,camera);h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 centre=np.asarray(centre,float)
 frame=quat_matrix(ctx.state[f'{arm}_ee_pose'][3:]);apex=ctx.state[f'{arm}_ee_pose'][:3]+frame@np.array([.157570,0.,.00485992]);apex_pixel=ctx.perception.world_to_pixel(apex,camera,ctx.state,rgb.shape)
 cx=w*.5
 mask=((hsv[...,0]>10)&(hsv[...,0]<48)&(hsv[...,1]>20)&(hsv[...,2]>110)).astype(np.uint8)*255
 mask[:,:max(0,int(cx-w*.047))]=0;mask[:,min(w,int(cx+w*.047)):]=0;mask[max(0,int(apex_pixel[1])-10):]=0
 mask &= identity_mask
 mask=cv2.erode(mask,np.ones((5,5),np.uint8));ys,xs=np.nonzero(mask)
 if len(xs)<300:return None
 x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
 template=gray[y0:y1,x0:x1].copy();m=mask[y0:y1,x0:x1].copy()
 if np.std(template[m>0])<8:return None
 origin,rot,fx,fy=ctx.perception.camera_geometry(camera,ctx.state,rgb.shape);depth=float(-((centre-origin)@rot)[2])
 if not .05<depth<.8:return None
 return dict(camera=camera,template=template,mask=m,origin_pixel=[x0,y0],centre=centre,old_centre_pixel=world_pixels(ctx,[centre],rgb.shape,camera)[0],depth=depth,fx=fx,fy=fy,tcp=ctx.tcp(arm).copy(),mask_pixels=len(xs),finger_apex_pixel=apex_pixel)


def _direct_template_follows(ctx,arm,ref):
 if ref is None:return False,dict(reason='no_discriminative_foreground_template')
 rgb=_toast_rgb(ctx,ref['camera']);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape;th,tw=ref['template'].shape;x0,y0=ref['origin_pixel'];limit=.004
 radius=max(3,int(math.ceil(max(ref['fx'],ref['fy'])*limit/ref['depth'])))
 static=world_pixels(ctx,[ref['centre']],rgb.shape,ref['camera'])[0]-ref['old_centre_pixel'];movement=ctx.tcp(arm)-ref['tcp']
 proof=dict(mask_pixels=ref['mask_pixels'],depth_m=ref['depth'],search_radius_pixels=radius,allowed_relative_motion_m=limit,stationary_prediction_pixels=static,actual_tcp_delta_m=movement)
 if np.linalg.norm(movement)<.010 or np.linalg.norm(static)<2*radius+3:return False,dict(**proof,reason='probe_not_discriminative')
 def score(shift):
  cx=x0+shift[0];cy=y0+shift[1];x=max(0,int(math.floor(cx-radius)));y=max(0,int(math.floor(cy-radius)));xx=min(w,int(math.ceil(cx+radius+tw)));yy=min(h,int(math.ceil(cy+radius+th)))
  search=gray[y:yy,x:xx]
  if search.shape[0]<th or search.shape[1]<tw:return dict(score=-1.,shift=[None,None])
  corr=cv2.matchTemplate(search,ref['template'],cv2.TM_CCOEFF_NORMED,mask=ref['mask']);corr=np.nan_to_num(corr,nan=-1.,posinf=-1.,neginf=-1.);_,value,_,xy=cv2.minMaxLoc(corr)
  displacement=np.array([float(x+xy[0]-x0),float(y+xy[1]-y0)])
  result=dict(score=float(value),shift=displacement.tolist(),raw_ncc=float(value),affine_accepted=False)
  px=x+xy[0];py=y+xy[1];patch=gray[py:py+th,px:px+tw]
  if patch.shape!=(th,tw):return result
  warp=np.eye(2,3,dtype=np.float32)
  try:
   correlation,warp=cv2.findTransformECC(ref['template'],patch,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-5),inputMask=ref['mask'],gaussFiltSize=3)
  except cv2.error:return result
  u,scales,vt=np.linalg.svd(warp[:,:2]);rotation=u@vt;angle=float(np.arctan2(rotation[1,0],rotation[0,0]))
  midpoint=np.array([tw/2,th/2]);measured=displacement+warp[:,:2]@midpoint+warp[:,2]-midpoint
  # Correct only small perspective/compliance changes, with the same bounds
  # for the held and stationary hypotheses. Never allow a free image warp.
  accepted=bool(np.linalg.det(warp[:,:2])>0 and np.min(scales)>=.90 and np.max(scales)<=1.10 and abs(angle)<=.05 and np.linalg.norm(measured-shift)<=radius*1.5)
  result.update(ecc=float(correlation),affine_matrix=warp.tolist(),affine_scales=scales.tolist(),affine_angle_rad=angle,affine_accepted=accepted)
  if accepted:result.update(score=float(correlation),shift=measured.tolist())
  return result
 held=score([0,0]);stationary=score(static);proof.update(held=held,stationary=stationary)
 if held['score']<.85:return False,dict(**proof,reason='held_template_not_identifiable')
 shift=np.asarray(held['shift']);relative=float(np.linalg.norm(shift/[ref['fx'],ref['fy']])*ref['depth']);proof['relative_translation_proxy_m']=relative
 ok=relative<=limit and held['score']-stationary['score']>=.10
 return ok,dict(**proof,reason='held_hypothesis_dominates' if ok else 'ambiguous_or_excessive_relative_motion')


def direct_bread_reference(ctx,arm,rack_yaw,centre_offset=None):
    """Track visible bread grain in head and active wrist RGB."""
    _,head=head_image(ctx)
    centre=ctx.tcp(arm)+(np.array([0,0,-.0464]) if centre_offset is None else np.asarray(centre_offset,float))
    c,s=math.cos(rack_yaw),math.sin(rack_yaw)
    frame=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    points=np.array([[x,y,z] for x in (-.0059,.0059) for y in (-.059,.059)
                     for z in (-.059,.059)])@frame.T+centre
    views=[]
    template_identity_mask = None
    for camera in (f'cam_{arm}_wrist',head):
        if camera not in ctx.vision:continue
        image=_toast_rgb(ctx,camera);pixels=world_pixels(ctx,points,image.shape,camera)
        if not np.isfinite(pixels).all():continue
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(pixels.astype(np.float32)).astype(np.int32),255)
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        saturation=20 if 'wrist' in camera else 8
        mask&=((hsv[...,0]>10)&(hsv[...,0]<48)&(hsv[...,1]>saturation)&(hsv[...,2]>110)).astype(np.uint8)*255
        if camera == f'cam_{arm}_wrist':template_identity_mask = mask.copy()
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        features=cv2.goodFeaturesToTrack(gray,maxCorners=100,qualityLevel=.02,
                                       minDistance=3,mask=mask,blockSize=3)
        minimum=12 if 'wrist' in camera else 6
        if features is None or len(features)<minimum:continue
        views.append(dict(gray=gray,features=features,camera=camera,
                          old_pixel=world_pixels(ctx,[centre],image.shape,camera)[0]))
    template = _direct_template_reference(ctx, arm, centre, template_identity_mask)
    if not views and template is None:return None
    return dict(views=views,centre=centre,tcp=ctx.tcp(arm).copy(),template=template)


def _direct_feature_follows(ctx, arm, reference):
    """Verify bread texture with independent, bidirectional SIFT matches.

    Match both possible motion regions before estimating a bounded transform;
    a stationary background pattern must not be interpreted as a held slice.
    """
    if reference is None:
        return False, dict(reason="no_target_texture_reference")
    rgb = _toast_rgb(ctx, reference["camera"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    movement = ctx.tcp(arm) - reference["tcp"]
    stationary = (world_pixels(ctx, [reference["centre"]], rgb.shape, reference["camera"])[0]
                  - reference["old_centre_pixel"])
    radius = max(3, int(math.ceil(max(reference["fx"], reference["fy"]) * .004 / reference["depth"])))
    proof = dict(stationary_prediction_pixels=stationary, search_radius_pixels=radius,
                 actual_tcp_delta_m=movement)
    if np.linalg.norm(movement) < .010 or np.linalg.norm(stationary) < 2 * radius + 3:
        return False, dict(**proof, reason="probe_not_discriminative")
    height, width = gray.shape
    th, tw = reference["template"].shape
    origin = np.asarray(reference["origin_pixel"])
    mask = np.zeros_like(gray, np.uint8)
    for shift in (np.zeros(2), stationary):
        x, y = origin + shift
        x0, x1 = max(0, int(math.floor(x - radius))), min(width, int(math.ceil(x + tw + radius)))
        y0, y1 = max(0, int(math.floor(y - radius))), min(height, int(math.ceil(y + th + radius)))
        mask[y0:y1, x0:x1] = 255
    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=.006, edgeThreshold=12)
    old, old_descriptors = sift.detectAndCompute(reference["template"], reference["mask"])
    new, new_descriptors = sift.detectAndCompute(gray, mask)
    proof.update(reference_features=len(old), current_features=len(new))
    if old_descriptors is None or new_descriptors is None:
        return False, dict(**proof, reason="insufficient_texture")
    matcher = cv2.BFMatcher()
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(new_descriptors, old_descriptors)}
    before, after = [], []
    for pair in matcher.knnMatch(old_descriptors, new_descriptors, k=2):
        if len(pair) != 2:
            continue
        match, second = pair
        if match.distance >= .75 * second.distance or reverse.get(match.trainIdx) != match.queryIdx:
            continue
        a = np.asarray(old[match.queryIdx].pt) + origin
        b = np.asarray(new[match.trainIdx].pt)
        if any(np.linalg.norm(a - previous) < 2 for previous in before):
            continue
        before.append(a)
        after.append(b)
    proof['independent_matches'] = len(before)
    if len(before) < 8:
        return False, dict(**proof, reason="too_few_independent_matches")
    before, after = np.asarray(before), np.asarray(after)
    transform, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
            ransacReprojThreshold=2., maxIters=1000, confidence=.995, refineIters=10)
    if transform is None or inliers is None:
        return False, dict(**proof, reason="no_consistent_texture_transform")
    keep = inliers.ravel().astype(bool)
    count = int(keep.sum())
    fraction = float(keep.mean())
    span = np.ptp(before[keep], axis=0) if count else np.zeros(2)
    scale = float(np.linalg.norm(transform[:, 0]))
    angle = float(math.atan2(transform[1, 0], transform[0, 0]))
    observed = np.median(after[keep] - before[keep], axis=0)
    residual = after[keep] - (before[keep] @ transform[:, :2].T + transform[:, 2])
    error = float(np.median(np.linalg.norm(residual, axis=1)))
    relative = float(np.linalg.norm(observed / [reference['fx'], reference['fy']]) * reference['depth'])
    advantage = float(np.linalg.norm(observed - stationary) - np.linalg.norm(observed))
    proof.update(inliers=count, inlier_fraction=fraction, feature_span_pixels=span,
                 scale=scale, angle_rad=angle, observed_motion_pixels=observed,
                 fit_error_pixels=error, relative_translation_proxy_m=relative,
                 held_advantage_pixels=advantage)
    accepted = (count >= 8 and fraction >= .65 and min(span) >= 8 and max(span) >= 20
                and .90 <= scale <= 1.10 and abs(angle) <= .05 and error <= 1.5
                and relative <= .004 and advantage >= 6.)
    return bool(accepted), dict(**proof, reason="held_texture_features" if accepted else "motion_or_texture_ambiguous")


def direct_bread_follows(ctx,arm,reference):
    if reference is None:return False,dict(reason='no_positive_bread_features')
    centre=reference['centre'];delta=ctx.tcp(arm)-reference['tcp']
    proofs=[]
    for view in reference['views']:
        camera=view['camera'];image=_toast_rgb(ctx,camera);gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        end,valid,_=cv2.calcOpticalFlowPyrLK(view['gray'],gray,view['features'],None,
                                          winSize=(17,17),maxLevel=2)
        if end is None:continue
        valid=valid.ravel().astype(bool);before=view['features'][:,0][valid];after=end[:,0][valid]
        minimum=12 if 'wrist' in camera else 6
        if len(before)<minimum:continue
        if 'wrist' in camera:
            # A held pattern stays fixed in the moving wrist. A stationary
            # table object must move appreciably under this same camera motion.
            expected=np.zeros(2)
            static=world_pixels(ctx,[centre],image.shape,camera)[0]-view['old_pixel']
            discriminative=np.linalg.norm(static)>6
        else:
            expected=world_pixels(ctx,[centre+delta],image.shape,camera)[0]-view['old_pixel']
            discriminative=np.linalg.norm(expected)>3
        flow=after-before;observed=np.median(flow,axis=0)
        error=float(np.linalg.norm(observed-expected))
        dispersion=float(np.median(np.linalg.norm(flow-observed,axis=1)))
        fraction=float(np.mean(np.linalg.norm(flow-expected,axis=1)<2.5))
        proof=dict(camera=camera,tracks=len(before),expected=expected.tolist(),
                   observed=observed.tolist(),pixel_error=error,dispersion=dispersion,
                   inlier_fraction=fraction,discriminative=bool(discriminative))
        proofs.append(proof)
        if discriminative and error<2.5 and dispersion<2.5 and fraction>.55:
            return True,dict(accepted_camera=camera,views=proofs)
    template_ok, template_proof = _direct_template_follows(ctx, arm, reference.get('template'))
    if template_ok:
        return True, dict(accepted_camera=f'cam_{arm}_wrist', views=proofs,
                          template_match=template_proof)
    features_ok, feature_proof = _direct_feature_follows(ctx, arm, reference.get('template'))
    if features_ok:
        return True, dict(accepted_camera=f'cam_{arm}_wrist', views=proofs,
                          template_match=template_proof, feature_match=feature_proof)
    return False,dict(views=proofs,reason='no_view_confirmed_rigid_follow',
                      template_match=template_proof,feature_match=feature_proof)


def fit_held_bread(views, frame_mapping=None):
    """Estimate the rigid slice offset from two RGB silhouettes and public CAD.

    The views must share the same grip and expose different head-camera
    perspectives. This fits observations; it never accesses object state.
    """
    from scipy.spatial.transform import Rotation
    if len(views) < 2 or any(view is None for view in views):
        return None
    # A top pinch followed by the donor's in-plane quarter turn maps bread
    # local X to tool X, local Y to minus tool Z, thickness to tool Y.
    profile = _BREAD_PROFILE - np.asarray([.0000864, -.0002499])
    mapping = (np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
               if frame_mapping is None else np.asarray(frame_mapping, float))
    model = np.asarray([[x, y, normal] for normal in (-.0059, .0059)
                        for x, y in profile]) @ mapping.T

    def perimeter(parameters, view):
        local = model @ Rotation.from_rotvec(parameters[3:]).as_matrix().T + parameters[:3]
        world = local @ view["frame"].T + view["tcp"]
        camera = (world - view["origin"]) @ view["rotation"]
        u0, v0 = view["centre"]
        pixels = np.c_[u0 - view["fx"] * camera[:, 0] / camera[:, 2],
                       v0 + view["fy"] * camera[:, 1] / camera[:, 2]]
        hull = cv2.convexHull(pixels.astype(np.float32))[:, 0, :]
        return np.concatenate([a + (b - a) * t for a, b in zip(hull, np.roll(hull, -1, axis=0))
                               for t in np.linspace(0, 1, 12)]).reshape(-1, 2)

    def residual(parameters):
        # Lower edges may be hidden by the donor's fingers. Visible observed
        # edges constrain the model without treating the occluder as bread.
        return np.concatenate([cKDTree(perimeter(parameters, view)).query(view["pixels"])[0]
                               for view in views])

    result = least_squares(residual, [.055, .002, -.010, 0, .05, 0],
                           bounds=([.010, -.015, -.040, -.20, -.40, -.25],
                                   [.090, .020, .035, .20, .40, .25]),
                           max_nfev=120, diff_step=.002, loss="soft_l1", f_scale=1)
    error = float(np.mean(abs(result.fun)))
    if error > 4.5:
        return None
    offset = result.x[:3]
    local = model @ Rotation.from_rotvec(result.x[3:]).as_matrix().T + offset
    # Intersect the observed upper profile with the centre of the broad face.
    # A bounding-box highest corner overestimates the top at this grasp point.
    hull = cv2.convexHull(local[:, (0, 2)].astype(np.float32))[:, 0, :]
    upper = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        if min(a[0], b[0]) <= offset[0] <= max(a[0], b[0]) and abs(b[0] - a[0]) > 1e-7:
            upper.append(float(a[1] + (b[1] - a[1]) * (offset[0] - a[0]) / (b[0] - a[0])))
    if not upper:
        return None
    receive = np.asarray([offset[0], offset[1], max(upper) - .014])
    return dict(offset=offset, receive=receive, rotation=result.x[3:], pixel_error=error,
                local_bounds=np.asarray([local.min(axis=0), local.max(axis=0)]))


def donor_bootstrap_views(ctx, frame_mapping=None):
    """Stopped head/donor views, retaining original foreground bread edges."""
    from ..motion import quat_matrix
    mapping = (np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
               if frame_mapping is None else np.asarray(frame_mapping, float))
    offset = (np.asarray([.055, .002, -.010]) if frame_mapping is None else
              mapping[:, 0] * .054)
    tcp = ctx.tcp("left")
    frame = quat_matrix(ctx.state["left_ee_pose"][3:])
    centre = tcp + frame @ offset
    cameras = [next((name for name in ctx.vision if "head" in name), None),
               next((name for name in ctx.vision if "left" in name and "wrist" in name), None)]
    if any(camera is None for camera in cameras):
        return None
    views = []
    for camera in cameras:
        image = _toast_rgb(ctx,camera).copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
                (hsv[..., 1] > 4) & (hsv[..., 1] < 195) & (hsv[..., 2] > 130) &
                ((hsv[..., 1] > 45) | (hsv[..., 2] > 190))).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, component_labels = cv2.connectedComponents(mask)
        predicted = ctx.perception.world_to_pixel(centre, camera, ctx.state, image.shape)
        # The public finger apex supplies a row just above the gripping pad.
        # Measure the real colour run there before considering the centroid
        # of a component that can also contain distant rack slices.
        foreground_lane = None
        anchor = None
        if "left" in camera:
            apex = (np.asarray(ctx.state["left_ee_pose"][:3]) +
                    frame @ np.asarray([.157570, 0., .00485992]))
            apex_pixel = ctx.perception.world_to_pixel(apex, camera, ctx.state, image.shape)
            anchor = int(math.floor(apex_pixel[1])) - 9
            bounds = []
            for row in range(anchor - 3, anchor + 4):
                if not 3 < row < image.shape[0] - 3:
                    continue
                line = cv2.morphologyEx(mask[row:row + 1], cv2.MORPH_CLOSE,
                                       np.ones((1, 5), np.uint8))[0] > 0
                columns = np.flatnonzero(line)
                runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
                candidates = [run for run in runs if 12 <= len(run) < image.shape[1] * .22 and
                              abs(float((run[0] + run[-1]) / 2) - predicted[0]) < 65]
                if candidates:
                    run = min(candidates, key=lambda r: abs((r[0] + r[-1]) / 2 - predicted[0]))
                    bounds.append([run[0], run[-1]])
            if len(bounds) >= 4:
                lower, upper = np.median(bounds, axis=0)
                foreground_lane = (float(lower - 6), float(upper + 6))
        fragments = []
        unknown_outline = np.zeros(image.shape[:2], np.uint8)
        for contour in contours:
            if not 80 < cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .65:
                continue
            pixels = contour[:, 0, :].astype(float)
            near_centre = np.linalg.norm(pixels.mean(axis=0) - predicted) < 65
            if foreground_lane is None:
                if near_centre:
                    fragments.append(pixels)
                continue
            relevant = ((pixels[:, 0] >= foreground_lane[0]) &
                        (pixels[:, 0] <= foreground_lane[1]))
            at_neck = relevant & (abs(pixels[:, 1] - anchor) <= 4)
            if relevant.sum() < 12 or not at_neck.any():
                continue
            left_merge = pixels[:, 0].min() < foreground_lane[0] - image.shape[1] * .03
            right_merge = pixels[:, 0].max() > foreground_lane[1] + image.shape[1] * .03
            if left_merge or right_merge:
                integer = np.rint(pixels).astype(int)
                relevant &= hsv[integer[:, 1], integer[:, 0], 1] > 45
                middle = (foreground_lane[0] + foreground_lane[1]) / 2
                # Keep every retained original facing edge, including the
                # visible rounded tip. Only a real colour run continuing
                # from the stem into the distant component can hide an edge;
                # do not cancel reverse evidence over an entire half-image.
                label = component_labels[integer[0, 1], integer[0, 0]]
                for row in range(image.shape[0]):
                    columns = np.flatnonzero(component_labels[row] == label)
                    runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
                    for run in runs:
                        if not len(run) or run[0] > middle + 2 or run[-1] < middle - 2:
                            continue
                        if left_merge and run[0] < foreground_lane[0] - 3:
                            lower = max(0, int(math.floor(foreground_lane[0] - 12)))
                            upper = min(image.shape[1], int(math.ceil(middle)))
                            unknown_outline[row, lower:upper] = 255
                        if right_merge and run[-1] > foreground_lane[1] + 3:
                            lower = max(0, int(math.floor(middle)))
                            upper = min(image.shape[1], int(math.ceil(foreground_lane[1] + 12)))
                            unknown_outline[row, lower:upper] = 255
                unknown_outline = cv2.dilate(unknown_outline, np.ones((5, 5), np.uint8))
            if relevant.sum() >= 12:
                fragments.append(pixels[relevant])
        if not fragments:
            return None
        pixels = np.concatenate(fragments)
        hull = cv2.convexHull(pixels.astype(np.float32))[:, 0, :]
        black_distance = cv2.distanceTransform((image.max(axis=2) > 70).astype(np.uint8),
                                              cv2.DIST_L2, 3)
        height, width = image.shape[:2]
        visible = []
        for pixel in pixels:
            x, y = np.rint(pixel).astype(int)
            if (3 < x < width - 3 and 3 < y < height - 3 and black_distance[y, x] > 4 and
                    abs(cv2.pointPolygonTest(hull, tuple(pixel), True)) < 1.4):
                visible.append(pixel)
        if len(visible) < 12:
            return None
        visible = np.asarray(visible)[::max(1, len(visible) // 100)]
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        views.append(dict(tcp=tcp, frame=frame, pixels=visible, hull=hull, image=image,
                          origin=origin, rotation=rotation, fx=fx, fy=fy,
                          centre=np.asarray(image.shape[1::-1]) / 2, camera=camera,
                          unknown_outline=unknown_outline))
    if np.linalg.norm(views[0]["origin"] - views[1]["origin"]) < .03:
        return None
    return views


def _donor_head_plane_prior(views, mapping, offset):
    """Measure a broad face axis in RGB, using the public gripping plane.

    The head exposes that face at near; the donor wrist supplies a second,
    independent view for the eventual rigid fit. Projecting a rack merge or
    a grazing silhouette onto this plane cannot give a plausible bread size.
    """
    if views is None or len(views) != 2:
        return None
    head = views[0]
    normal = head["frame"] @ mapping[:, 2]
    centre = head["tcp"] + head["frame"] @ offset
    sign = np.sign((head["origin"] - centre) @ normal)
    plane = centre + normal * sign * .0059
    pixels = head["pixels"]
    u0, v0 = head["centre"]
    rays = np.c_[-(pixels[:, 0] - u0) / head["fx"],
                 (pixels[:, 1] - v0) / head["fy"], np.ones(len(pixels))] @ head["rotation"].T
    denominator = rays @ normal
    if np.any(abs(denominator) < .01):
        return None
    distance = ((plane - head["origin"]) @ normal) / denominator
    if not np.isfinite(distance).all() or np.any(distance >= 0):
        return None
    world = head["origin"] + rays * distance[:, None]
    projected = (world - centre) @ head["frame"] @ mapping[:, :2]
    rectangle = cv2.minAreaRect(projected.astype(np.float32))
    extent = np.sort(np.asarray(rectangle[1]))
    public_extent = np.sort(np.ptp(_BREAD_PROFILE, axis=0))
    if np.any(extent < public_extent * .65) or np.any(extent > public_extent * 1.35):
        return None
    corners = cv2.boxPoints(rectangle)
    edges = np.roll(corners, -1, axis=0) - corners
    angles = (np.arctan2(edges[:, 1], edges[:, 0]) + math.pi / 2) % math.pi - math.pi / 2
    angle = float(angles[np.argmin(abs(angles))])
    # Near a square's diagonal, the two identifiable source axes compete.
    if abs(angle) >= math.pi / 4 - .10:
        return None
    return dict(angle=angle, extent=extent)


def fit_donor_bootstrap(views, frame_mapping=None):
    """Return a donor prior only after a stopped rigid two-camera fit."""
    from scipy.spatial.transform import Rotation
    mapping = (np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
               if frame_mapping is None else np.asarray(frame_mapping, float))
    offset = (np.asarray([.055, .002, -.010]) if frame_mapping is None else
              mapping[:, 0] * .054)
    prior = _donor_head_plane_prior(views, mapping, offset)
    if prior is None:
        return None
    solutions = []
    for variation in (-.06, 0., .06):
        reference = Rotation.from_rotvec(mapping[:, 2] * (prior["angle"] + variation)).as_matrix() @ mapping
        fitted = fit_receiver_bread(views, reference_frame=reference, reference_offset=offset)
        if fitted is None:
            return None
        change = abs(np.asarray(fitted["offset"]) - offset)
        if (max(fitted["pixel_errors"]) > 4.5 or
                np.any(change > np.asarray([.0195, .0145, .0195])) or
                np.max(abs(fitted["rotation"])) > .39):
            return None
        solutions.append(fitted)
    centre_spread = max(float(np.linalg.norm(a["offset"] - b["offset"]))
                        for a in solutions for b in solutions)
    angle_spread = max(float(Rotation.from_matrix(a["frame"] @ b["frame"].T).magnitude())
                       for a in solutions for b in solutions)
    if centre_spread > .004 or angle_spread > .06:
        return None
    fitted = min(solutions, key=lambda candidate: np.mean(candidate["pixel_errors"]))
    profile = _BREAD_PROFILE - np.asarray([.0000864, -.0002499])
    local = np.asarray([[x, y, normal] for normal in (-.0059, .0059)
                        for x, y in profile]) @ fitted["frame"].T + fitted["offset"]
    hull = cv2.convexHull(local[:, (0, 2)].astype(np.float32))[:, 0, :]
    upper = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        if (min(a[0], b[0]) <= fitted["offset"][0] <= max(a[0], b[0]) and
                abs(b[0] - a[0]) > 1e-7):
            upper.append(float(a[1] + (b[1] - a[1]) *
                               (fitted["offset"][0] - a[0]) / (b[0] - a[0])))
    if not upper:
        return None
    receive = np.asarray([fitted["offset"][0], fitted["offset"][1], max(upper) - .014])
    world = local @ views[0]["frame"].T + views[0]["tcp"]
    return dict(offset=fitted["offset"], receive=receive,
                rotation=Rotation.from_matrix(fitted["frame"] @ mapping.T).as_rotvec(),
                pixel_error=float(np.mean(fitted["pixel_errors"])),
                pixel_errors=fitted["pixel_errors"],
                contact_plane_distance=fitted["contact_plane_distance"],
                head_axis_angle=prior["angle"], head_plane_extent=prior["extent"],
                centre_spread=centre_spread, angle_spread=angle_spread,
                lowest_point=float(world[:, 2].min()),
                local_bounds=np.asarray([local.min(axis=0), local.max(axis=0)]))


def donor_bread_views(ctx, expected_offset):
    """Current, simultaneous wrist silhouettes anchored to the stopped donor.

    The receiving arm occludes the head camera at contact. Texture and fingers
    can split one narrow bread silhouette into several components, so retain
    their joint convex exterior instead of fitting an individual fragment.
    """
    from ..motion import quat_matrix
    tcp = ctx.tcp("left")
    frame = quat_matrix(ctx.state["left_ee_pose"][3:])
    world_centre = tcp + frame @ np.asarray(expected_offset, float)
    cameras = [next((name for name in ctx.vision if arm in name and "wrist" in name), None)
               for arm in ("left", "right")]
    if any(camera is None for camera in cameras):
        return None
    views = []
    for camera in cameras:
        image = _toast_rgb(ctx,camera)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
                (hsv[..., 1] > 4) & (hsv[..., 1] < 195) & (hsv[..., 2] > 130) &
                ((hsv[..., 1] > 45) | (hsv[..., 2] > 190))).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        predicted = ctx.perception.world_to_pixel(world_centre, camera, ctx.state, image.shape)
        fragments = [contour for contour in contours
                     if 80 < cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .65 and
                     np.linalg.norm(contour[:, 0, :].mean(axis=0) - predicted) < 65]
        if not fragments:
            return None
        pixels = np.concatenate(fragments)[:, 0, :].astype(float)
        hull = cv2.convexHull(pixels.astype(np.float32))[:, 0, :]
        black_distance = cv2.distanceTransform((image.max(axis=2) > 70).astype(np.uint8),
                                              cv2.DIST_L2, 3)
        height, width = image.shape[:2]
        visible = []
        for pixel in pixels:
            x, y = np.rint(pixel).astype(int)
            if (3 < x < width - 3 and 3 < y < height - 3 and black_distance[y, x] > 4 and
                    abs(cv2.pointPolygonTest(hull, tuple(pixel), True)) < 1.4):
                visible.append(pixel)
        if len(visible) < 12:
            return None
        visible = np.asarray(visible)[::max(1, len(visible) // 100)]
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        views.append(dict(tcp=tcp, frame=frame, pixels=visible, hull=hull, image=image,
                          origin=origin, rotation=rotation, fx=fx, fy=fy,
                          centre=np.asarray(image.shape[1::-1]) / 2, camera=camera))
    if np.linalg.norm(views[0]["origin"] - views[1]["origin"]) < .03:
        return None
    return views


def current_donor_bread_views(ctx, expected_offset, expected_frame, cameras=None, stereo_plane=None,
                            broad_silhouette=False):
    """Retain visible fragments anywhere on the RGB-fitted public profile.

    The oblique receiving camera can expose an upper broad-face fragment far
    from the slice centre. Its projected near-view prior supplies a local ROI;
    current image agreement and independent pose fits remain mandatory.
    """
    from ..motion import quat_matrix
    tcp = ctx.tcp("left")
    frame = quat_matrix(ctx.state["left_ee_pose"][3:])
    world_centre = tcp + frame @ np.asarray(expected_offset, float)
    profile = _BREAD_PROFILE - np.asarray([.0000864, -.0002499])
    local_profile = np.asarray([[x, y, normal] for normal in (-.0059, .0059)
                                for x, y in profile]) @ np.asarray(expected_frame).T
    world_profile = (local_profile + expected_offset) @ frame.T + tcp
    if cameras is None:
        cameras = [next((name for name in ctx.vision if arm in name and "wrist" in name), None)
                   for arm in ("left", "right")]
    if (len(cameras) not in (1, 2) or (len(cameras) == 1 and stereo_plane is None) or
            any(camera is None or camera not in ctx.vision for camera in cameras)):
        return None
    views = []
    for camera in cameras:
        image = _toast_rgb(ctx,camera)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
                (hsv[..., 1] > 4) & (hsv[..., 1] < 195) & (hsv[..., 2] > 130) &
                ((hsv[..., 1] > 45) | (hsv[..., 2] > 190))).astype(np.uint8) * 255
        if stereo_plane is not None or (broad_silhouette and len(cameras) == 2):
            # Blue table illumination shifts the pale crust toward green.
            # Two independent current contours may bootstrap rectification.
            # A bootstrap is never actionable: original current stereo depth
            # and a new constrained final fit must still pass afterward.
            mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 85) &
                    (hsv[..., 1] > 8) & (hsv[..., 1] < 195) &
                    (hsv[..., 2] > 110)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        predicted = ctx.perception.world_to_pixel(world_centre, camera, ctx.state, image.shape)
        projected = world_pixels(ctx, world_profile, image.shape, camera)
        if not np.isfinite(projected).all():
            return None
        expected_hull = cv2.convexHull(projected.astype(np.float32))[:, 0, :]
        # At the donor's own wrist, the held edge passes immediately above
        # the finger apex. A row run measured in the original RGB separates
        # that foreground stem from same-coloured rack slices behind it.
        # The row is supplied by the public link7 mesh, not episode geometry.
        foreground_lane = None
        anchor = None
        unknown_outline = np.zeros(image.shape[:2], np.uint8)
        if "left" in camera:
            apex = (np.asarray(ctx.state["left_ee_pose"][:3]) +
                    frame @ np.asarray([.157570, 0., .00485992]))
            apex_pixel = ctx.perception.world_to_pixel(apex, camera, ctx.state, image.shape)
            anchor = int(math.floor(apex_pixel[1])) - 9
            bounds = []
            for row in range(anchor - 3, anchor + 4):
                if not 3 < row < image.shape[0] - 3:
                    continue
                line = cv2.morphologyEx(mask[row:row + 1], cv2.MORPH_CLOSE,
                                       np.ones((1, 5), np.uint8))[0] > 0
                columns = np.flatnonzero(line)
                runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
                candidates = [run for run in runs if len(run) >= 12 and
                              len(run) < image.shape[1] * .22 and
                              abs(float((run[0] + run[-1]) / 2) - predicted[0]) < 65]
                if candidates:
                    run = min(candidates, key=lambda r: abs((r[0] + r[-1]) / 2 - predicted[0]))
                    bounds.append([run[0], run[-1]])
            if len(bounds) >= 4:
                lower, upper = np.median(bounds, axis=0)
                foreground_lane = (float(lower - 6), float(upper + 6))
        fragments = []
        for contour in contours:
            if not 80 < cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .65:
                continue
            # A connected component can contain a rack slice behind the held
            # bread. Keep only its original colour-boundary pixels near the
            # projected prior; never contour a cropped mask or synthesize a
            # boundary along the ROI itself.
            pixels = contour[:, 0, :].astype(float)
            relevant = np.asarray([cv2.pointPolygonTest(expected_hull, tuple(pixel), True) >= -12
                                    for pixel in pixels])
            on_profile = float(relevant.mean()) >= .75
            near_centre = np.linalg.norm(pixels.mean(axis=0) - predicted) < 65
            if on_profile:
                fragments.append(pixels)
            elif near_centre and int(relevant.sum()) >= 12:
                background_merge = (foreground_lane is not None and "left" in camera and
                                    pixels[:, 0].min() < foreground_lane[0] - image.shape[1] * .03)
                if background_merge:
                    relevant &= ((pixels[:, 0] >= foreground_lane[0]) &
                                 (pixels[:, 0] <= foreground_lane[1]))
                    integer = np.rint(pixels).astype(int)
                    # Warm illumination can connect the white rack foot to
                    # bread. The gold/cream foreground stem is saturated;
                    # reject only those original low-saturation foot edges.
                    relevant &= hsv[integer[:, 1], integer[:, 0], 1] > 45
                    # The upper side facing the merged rack has no original
                    # colour exterior. Its far component outline belongs to
                    # background bread; retain the opposite upper edge and
                    # both exposed lower edges of the measured foreground.
                    middle = (foreground_lane[0] + foreground_lane[1]) / 2
                    relevant &= ~((pixels[:, 1] < anchor) & (pixels[:, 0] < middle))
                    unknown_outline[:max(0, anchor), :max(0, int(math.ceil(middle)))] = 255
                if int(relevant.sum()) >= 12:
                    fragments.append(pixels[relevant])
        if not fragments:
            return None
        pixels = np.concatenate(fragments)
        hull = cv2.convexHull(pixels.astype(np.float32))[:, 0, :]
        black_distance = cv2.distanceTransform((image.max(axis=2) > 70).astype(np.uint8),
                                              cv2.DIST_L2, 3)
        height, width = image.shape[:2]
        visible = []
        for pixel in pixels:
            x, y = np.rint(pixel).astype(int)
            if (3 < x < width - 3 and 3 < y < height - 3 and black_distance[y, x] > 4 and
                    abs(cv2.pointPolygonTest(hull, tuple(pixel), True)) < 1.4):
                visible.append(pixel)
        if len(visible) < 12:
            return None
        visible = np.asarray(visible)[::max(1, len(visible) // 100)]
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        views.append(dict(tcp=tcp, frame=frame, pixels=visible, hull=hull, image=image,
                          origin=origin, rotation=rotation, fx=fx, fy=fy,
                          centre=np.asarray(image.shape[1::-1]) / 2, camera=camera,
                          unknown_outline=unknown_outline))
    if len(views) > 1 and np.linalg.norm(views[0]["origin"] - views[1]["origin"]) < .03:
        return None
    return views


def receiver_texture_view(ctx, expected_offset, expected_frame):
    """Sample real bread texture across its RGB-fitted complete CAD profile.

    Equal vertical regions retain exposed upper texture as well as the lower
    gripping region. The public CAD projection defines only a search mask;
    every selected feature still comes from the original RGB and colour mask.
    """
    from ..motion import quat_matrix
    camera = next((name for name in ctx.vision if "right" in name and "wrist" in name), None)
    if camera is None:
        return None
    image = _toast_rgb(ctx,camera)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bread = ((hsv[..., 0] > 14) & (hsv[..., 0] < 42) &
             (hsv[..., 1] > 8) & (hsv[..., 2] > 115)).astype(np.uint8) * 255
    centre = ctx.tcp("right") + quat_matrix(ctx.state["right_ee_pose"][3:]) @ expected_offset
    pixel = ctx.perception.world_to_pixel(centre, camera, ctx.state, image.shape)
    height, width = image.shape[:2]
    frame = quat_matrix(ctx.state["right_ee_pose"][3:]) @ np.asarray(expected_frame, float)
    profile = _BREAD_PROFILE - np.asarray([.0000864, -.0002499])
    model = np.asarray([[x, y, normal] for normal in (-.0059, .0059)
                        for x, y in profile])
    projected = world_pixels(ctx, model @ frame.T + centre, image.shape, camera)
    if not np.isfinite(projected).all():
        return None
    hull = cv2.convexHull(projected.astype(np.float32))[:, 0, :]
    roi = np.zeros_like(bread)
    cv2.fillConvexPoly(roi, np.rint(hull).astype(int), 255)
    roi = cv2.erode(roi, np.ones((7, 7), np.uint8))
    # Keep the same central aperture used by the original load test. CAD and
    # RGB must agree here; finger or rack pixels outside it supply no features.
    roi[:, :int(width * .453)] = 0
    roi[:, int(width * .547):] = 0
    vertical = np.nonzero(roi)[0]
    if not len(vertical):
        return None
    bands = np.linspace(int(vertical.min()), int(vertical.max()) + 1, 7).astype(int)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    candidates = []
    search = bread & roi
    for lower, upper in zip(bands, bands[1:]):
        mask = np.zeros_like(search)
        mask[lower:upper] = search[lower:upper]
        band_points = cv2.goodFeaturesToTrack(gray, maxCorners=13, qualityLevel=.025,
                                             minDistance=4, mask=mask)
        candidates.append([] if band_points is None else band_points[:, 0, :])
    # Allocate by feature rank across every band. Enforce the original 4 px
    # separation globally, including neighbours on either side of a band cut.
    features = []
    for rank in range(13):
        for band_points in candidates:
            if rank >= len(band_points):
                continue
            point = band_points[rank]
            if all(np.linalg.norm(point - other) >= 4 for other in features):
                features.append(point)
    points = np.asarray(features, np.float32).reshape(-1, 1, 2)
    if points is None or len(points) < 6:
        return None
    origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
    depth = float(-((centre - origin) @ rotation)[2])
    if depth <= .025:
        return None
    # A public-CAD depth fitted from RGB supplies the stationary alternative.
    # The narrow edge view cannot reliably intersect a broad-face plane.
    camera_points = np.c_[(points[:, 0, 0] - width / 2) / fx * depth,
                          -(points[:, 0, 1] - height / 2) / fy * depth,
                          np.full(len(points), -depth)]
    world_points = camera_points @ rotation.T + origin
    return dict(gray=gray, points=points, camera=camera, pixel=pixel,
                world_points=world_points)


def receiver_texture_follows(ctx, view):
    """Compare held and stationary texture hypotheses after a measured probe."""
    if view is None:
        return False, dict(features=0, tracked=0, reason="no_bread_texture")
    image = np.asarray(ctx.vision[view["camera"]]["color"], np.uint8)[..., :3]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    points = view["points"]
    tracked, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
        view["gray"], gray, points, None, winSize=(15, 15), maxLevel=3)
    if tracked is None:
        return False, dict(features=len(points), tracked=0, reason="no_tracking")
    backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
        gray, view["gray"], tracked, None, winSize=(15, 15), maxLevel=3)
    if backward is None or backward_ok is None:
        return False, dict(features=len(points), tracked=0, reason="no_reverse_tracking")
    valid = ((forward_ok[:, 0] > 0) & (backward_ok[:, 0] > 0) &
             np.isfinite(tracked[:, 0]).all(axis=1) &
             (np.linalg.norm(backward[:, 0] - points[:, 0], axis=1) < 1.5))
    xy = np.rint(np.nan_to_num(tracked[:, 0], nan=-1000.)).astype(int)
    height, width = image.shape[:2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bread = ((hsv[..., 0] > 14) & (hsv[..., 0] < 42) &
             (hsv[..., 1] > 8) & (hsv[..., 2] > 115))
    valid &= inside
    valid[inside] &= bread[xy[inside, 1], xy[inside, 0]]
    count = int(valid.sum())
    evidence = dict(features=len(points), tracked=count)
    if count < 6 or count < len(points) * .4:
        return False, dict(**evidence, reason="lost_bread_texture")
    movement = np.linalg.norm(tracked[:, 0][valid] - points[:, 0][valid], axis=1)
    median, p80 = float(np.median(movement)), float(np.quantile(movement, .8))
    origin, rotation, fx, fy = ctx.perception.camera_geometry(view["camera"], ctx.state, image.shape)
    stationary_camera = (view["world_points"] - origin) @ rotation
    stationary_pixels = np.c_[width / 2 - fx * stationary_camera[:, 0] / stationary_camera[:, 2],
                              height / 2 + fy * stationary_camera[:, 1] / stationary_camera[:, 2]]
    predicted = float(np.median(np.linalg.norm(stationary_pixels[valid] - points[:, 0][valid], axis=1)))
    stationary_error = float(np.median(np.linalg.norm(stationary_pixels[valid] - tracked[:, 0][valid], axis=1)))
    evidence.update(median_pixels=median, p80_pixels=p80,
                    stationary_predicted_pixels=predicted, stationary_error_pixels=stationary_error)
    # A camera-axis translation can leave even an unsupported slice almost
    # stationary in the image. Require a visually distinguishable alternative.
    return median < 7 and p80 < 12 and predicted > 2.5 and median < .55 * stationary_error, evidence


def receiver_bread_views(ctx, expected_offset=None):
    """Visible outer bread edges at the receiver's measured pose.

    Brown texture is not a silhouette boundary. Keep only the convex outer
    band, and discard contours next to the fingers or the image boundary.
    """
    from ..motion import quat_matrix
    tcp = ctx.tcp("right")
    frame = quat_matrix(ctx.state["right_ee_pose"][3:])
    cameras = [next((name for name in ctx.vision if "head" in name), None),
               next((name for name in ctx.vision if "right" in name and "wrist" in name), None)]
    if any(camera is None for camera in cameras):
        return None
    views = []
    for camera in cameras:
        image = _toast_rgb(ctx,camera)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) &
                (hsv[..., 1] > 4) & (hsv[..., 1] < 195) & (hsv[..., 2] > 130) &
                ((hsv[..., 1] > 45) | (hsv[..., 2] > 190))).astype(np.uint8) * 255
        # An edge-on upright slice can have only four CHAIN_APPROX_SIMPLE
        # vertices. Retain real perimeter pixels before visibility filtering.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if "head" in camera:
            predicted_centre = tcp + frame @ (np.asarray(expected_offset, float)
                                                if expected_offset is not None else [ .045, 0, 0])
            predicted = ctx.perception.world_to_pixel(predicted_centre,
                                                      camera, ctx.state, image.shape)
            contours = [c for c in contours if 100 < cv2.contourArea(c) < 9000]
            if not contours:
                return None
            contour = min(contours, key=lambda c: np.linalg.norm(c[:, 0, :].mean(axis=0) - predicted))
            if np.linalg.norm(contour[:, 0, :].mean(axis=0) - predicted) > 85:
                return None
        else:
            contours = [c for c in contours if 500 < cv2.contourArea(c) < image.shape[0] * image.shape[1] * .65]
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
        pixels = contour[:, 0, :].astype(float)
        hull = cv2.convexHull(pixels.astype(np.float32))[:, 0, :].astype(float)
        black_distance = cv2.distanceTransform((image.max(axis=2) > 70).astype(np.uint8),
                                              cv2.DIST_L2, 3)
        height, width = image.shape[:2]
        visible = []
        for pixel in pixels:
            x, y = np.rint(pixel).astype(int)
            if (3 < x < width - 3 and 3 < y < height - 3 and black_distance[y, x] > 4 and
                    abs(cv2.pointPolygonTest(hull.astype(np.float32), tuple(pixel), True)) < 1.4):
                visible.append(pixel)
        if len(visible) < 12:
            return None
        visible = np.asarray(visible)[::max(1, len(visible) // 100)]
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        views.append(dict(tcp=tcp, frame=frame, pixels=visible, hull=hull, image=image,
                          origin=origin, rotation=rotation, fx=fx, fy=fy,
                          centre=np.asarray(image.shape[1::-1]) / 2, camera=camera))
    return views


def donor_stereo_plane(ctx, expected_offset, expected_frame):
    """Triangulate matched current bread texture, with no object-state input.

    The earlier RGB plane only rectifies the head image for SIFT. Every depth
    is then triangulated from the original pixels and calibrated cameras.
    """
    from itertools import combinations
    from ..motion import quat_matrix
    cameras = [next((c for c in ctx.vision if "head" in c), None),
               next((c for c in ctx.vision if "right" in c and "wrist" in c), None)]
    if any(c is None for c in cameras):
        return None
    frame = quat_matrix(ctx.state["left_ee_pose"][3:])
    centre = ctx.tcp("left") + frame @ np.asarray(expected_offset)
    reference = frame @ np.asarray(expected_frame)
    plane = np.asarray([[-.045, -.045, 0], [.045, -.045, 0],
                        [.045, .045, 0], [-.045, .045, 0]]) @ reference.T + centre
    profile = _BREAD_PROFILE - [.0000864, -.0002499]
    body = np.asarray([[x, y, n] for n in (-.0059, .0059) for x, y in profile])
    body = body @ reference.T + centre
    images, projections, rotations, origins, intrinsics, corners = [], [], [], [], [], []
    for camera in cameras:
        image = _toast_rgb(ctx, camera)
        points = world_pixels(ctx, plane, image.shape, camera)
        if not np.isfinite(points).all() or cv2.contourArea(points.astype(np.float32)) < 80:
            return None
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, image.shape)
        rcv = np.diag([1., -1., -1.]) @ rotation.T
        k = np.asarray([[fx, 0, image.shape[1] / 2],
                        [0, fy, image.shape[0] / 2], [0, 0, 1.]])
        images.append(image)
        projections.append(k @ np.c_[rcv, -rcv @ origin])
        rotations.append(rcv)
        origins.append(origin)
        intrinsics.append(k)
        corners.append(points)
    if np.linalg.norm(origins[1] - origins[0]) < .04:
        return None
    homography = cv2.getPerspectiveTransform(corners[0].astype(np.float32),
                                            corners[1].astype(np.float32))
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-9:
        return None
    inverse = np.linalg.inv(homography)
    rectified = cv2.warpPerspective(images[0], homography, images[1].shape[1::-1])
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=.008, edgeThreshold=15)
    features = []
    for i, image in enumerate((rectified, images[1])):
        mask = np.zeros(image.shape[:2], np.uint8)
        projected = (corners[1] if i == 0 else
                     world_pixels(ctx, body, image.shape, cameras[1]))
        if not np.isfinite(projected).all():
            return None
        cv2.fillConvexPoly(mask, cv2.convexHull(projected.astype(np.float32)).astype(int), 255)
        mask = cv2.dilate(mask, np.ones((35 if i == 0 else 25,) * 2, np.uint8))
        if i == 0:
            mask[image.max(axis=2) < 100] = 0
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            mask[(hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 45) | (hsv[:, :, 2] < 100)] = 0
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask)
        if descriptors is None or len(keypoints) < 12:
            return None
        features.append((keypoints, descriptors))
    relative = rotations[1] @ rotations[0].T
    translation = rotations[1] @ (origins[0] - origins[1])
    x, y, z = translation
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    fundamental = np.linalg.inv(intrinsics[1]).T @ cross @ relative @ np.linalg.inv(intrinsics[0])
    matcher = cv2.BFMatcher()
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(features[1][1], features[0][1])}
    points = []
    for pair in matcher.knnMatch(features[0][1], features[1][1], k=2):
        if len(pair) != 2:
            continue
        match, second = pair
        if match.distance >= .75 * second.distance or reverse.get(match.trainIdx) != match.queryIdx:
            continue
        u = inverse @ np.r_[features[0][0][match.queryIdx].pt, 1.]
        u /= u[2]
        v = np.r_[features[1][0][match.trainIdx].pt, 1.]
        line = fundamental @ u
        if abs(v @ line) / max(np.linalg.norm(line[:2]), 1e-12) > 1.5:
            continue
        point = cv2.triangulatePoints(*projections, u[:2, None], v[:2, None])
        if abs(point[3, 0]) < 1e-10:
            continue
        point = (point[:3] / point[3]).ravel()
        if (not np.isfinite(point).all() or np.linalg.norm(point - centre) > .10 or
                any((rotation @ (point - origin))[2] <= 0
                    for rotation, origin in zip(rotations, origins))):
            continue
        # Multiple SIFT orientations at one pixel do not add independent depth.
        if any(np.linalg.norm(point - previous) < .0015 for previous in points):
            continue
        points.append(point)
    if len(points) < 10:
        return None
    points = np.asarray(points)
    best = []
    for indices in combinations(range(len(points)), 3):
        triple = points[list(indices)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        length = np.linalg.norm(normal)
        if length < 1e-5:
            continue
        normal /= length
        inliers = np.flatnonzero(abs((points - triple[0]) @ normal) < .003)
        if len(inliers) > len(best):
            best = inliers
    if len(best) < 10 or len(best) < .75 * len(points):
        return None
    selected = points[best]
    origin = selected.mean(axis=0)
    _, singular, axes = np.linalg.svd(selected - origin)
    normal = axes[-1]
    if normal @ reference[:, 2] < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((selected - origin) @ normal) ** 2)))
    if (singular[1] / np.sqrt(len(selected)) < .015 or rms > .0018 or
            normal @ reference[:, 2] < math.cos(.30) or
            abs(normal @ (centre - origin)) > .025):
        return None
    return dict(points=selected, origin=origin, normal=normal, rms=rms,
                independent_matches=len(points), inliers=len(selected),
                second_axis_spread=float(singular[1] / np.sqrt(len(selected))),
                cameras=cameras)


def fit_receiver_bread(views, reference_frame=None, reference_offset=None, stereo_plane=None):
    """Fit a rigid held slice across fresh head/wrist views and RGB memory.

    Donor RGB supplies the reference axes for a square whose partially hidden
    silhouette is ambiguous. The receiver's images must still agree with the
    complete public outline at every observed robot pose.
    """
    from scipy.spatial.transform import Rotation
    if views is None or len(views) < 1:
        return None
    if len(views) == 1 and (stereo_plane is None or stereo_plane.get("inliers", 0) < 10 or
            stereo_plane.get("rms", 1.) > .0018 or
            stereo_plane.get("second_axis_spread", 0.) < .015 or
            len(set(stereo_plane.get("cameras", []))) < 2):
        return None
    profile = _BREAD_PROFILE - np.asarray([.0000864, -.0002499])
    use_reference = reference_frame is not None and reference_offset is not None
    mapping = (np.asarray(reference_frame, float) if use_reference else
               np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float))
    reference_offset = np.asarray(reference_offset, float) if use_reference else None
    model = np.asarray([[x, y, normal] for normal in (-.0059, .0059)
                        for x, y in profile]) @ mapping.T

    def projected_hull(parameters, view):
        local = model @ Rotation.from_rotvec(parameters[3:]).as_matrix().T + parameters[:3]
        world = local @ view["frame"].T + view["tcp"]
        camera = (world - view["origin"]) @ view["rotation"]
        u0, v0 = view["centre"]
        pixels = np.c_[u0 - view["fx"] * camera[:, 0] / camera[:, 2],
                       v0 + view["fy"] * camera[:, 1] / camera[:, 2]]
        return cv2.convexHull(pixels.astype(np.float32))[:, 0, :].astype(float)

    def closest_segments(points, hull):
        delta = np.roll(hull, -1, axis=0) - hull
        difference = points[:, None, :] - hull[None, :, :]
        fraction = np.clip(np.sum(difference * delta[None, :, :], axis=2) /
                           np.maximum(np.sum(delta * delta, axis=1), 1e-9), 0, 1)
        residual = difference - fraction[:, :, None] * delta[None, :, :]
        edge = np.argmin(np.sum(residual * residual, axis=2), axis=1)
        return residual[np.arange(len(edge)), edge, :]

    def residual(parameters):
        values = []
        for view in views:
            hull = projected_hull(parameters, view)
            observed = closest_segments(view["pixels"], hull)
            values.append(observed.ravel() * np.sqrt(50 / len(observed)))
            # The reverse term prevents an oversized or shifted CAD outline
            # from containing every observed edge. Black finger occlusions
            # and the clipped wrist view provide no reverse-edge evidence.
            delta = np.roll(hull, -1, axis=0) - hull
            lengths = np.linalg.norm(delta, axis=1)
            cumulative = np.r_[0, np.cumsum(lengths)]
            samples = np.linspace(0, cumulative[-1], 64, endpoint=False)
            edge = np.searchsorted(cumulative[1:], samples)
            outline = hull[edge] + delta[edge] * ((samples - cumulative[edge]) / lengths[edge])[:, None]
            reverse = closest_segments(outline, view["hull"])
            height, width = view["image"].shape[:2]
            for i, point in enumerate(outline):
                x, y = np.rint(point).astype(int)
                if not (3 < x < width - 3 and 3 < y < height - 3):
                    reverse[i] = 0
                elif view.get("unknown_outline") is not None and view["unknown_outline"][y, x]:
                    # A colour merge hides this true edge. The convex hull of
                    # retained original pixels cannot manufacture its closure.
                    reverse[i] = 0
                elif (view["image"][y - 2:y + 3, x - 2:x + 3].max(axis=2) < 70).any():
                    reverse[i] = 0
            values.append(reverse.ravel() * .6)
        if stereo_plane is not None:
            normal = views[0]["frame"] @ Rotation.from_rotvec(parameters[3:]).as_matrix() @ mapping[:, 2]
            centre = views[0]["tcp"] + views[0]["frame"] @ parameters[:3]
            side = 1. if normal @ (views[-1]["origin"] - centre) > 0 else -1.
            distances = (stereo_plane["points"] - centre) @ normal
            values.append((distances - side * .0059) / .001 * 15.)
        if use_reference:
            # This is a weak tie-breaker for occluded edges, not a substitute
            # for fresh image agreement. A temporal slip cannot be concealed
            # by choosing a different pose for each receiver observation.
            values.append((parameters[:3] - reference_offset) / .012 * 1.5)
            values.append(parameters[3:] / .18 * 1.5)
        return np.concatenate(values)

    def jacobian(parameters):
        columns = []
        # Absolute steps avoid zero rotation seeds disappearing in float32
        # convex-hull projection; translations and rotations use metre/radian.
        for index, step in enumerate((.00001, .00001, .00001, .0002, .0002, .0002)):
            delta = np.zeros(6)
            delta[index] = step
            columns.append((residual(parameters + delta) - residual(parameters - delta)) / (2 * step))
        return np.asarray(columns).T

    results = []
    if use_reference:
        extent = np.asarray([.020, .015, .020])
        bounds = (np.r_[reference_offset - extent, [-.4, -.4, -.4]],
                  np.r_[reference_offset + extent, [.4, .4, .4]])
        seeds = [np.r_[reference_offset, [0., pitch, yaw]]
                 for pitch in (-.12, 0., .12) for yaw in (-.12, 0., .12)]
    else:
        bounds = ([.005, -.06, -.07, -.8, -.9, -.9],
                  [.12, .06, .07, .8, .9, .9])
        seeds = [[.05, .002, 0, .03, pitch, yaw]
                 for pitch in (-.4, .05, .4) for yaw in (-.55, .05, .55)]
    for seed in seeds:
        result = least_squares(residual, seed, jac=jacobian, bounds=bounds,
                               max_nfev=100, loss="soft_l1", f_scale=2)
        results.append(result)
    result = min(results, key=lambda item: np.mean(abs(item.fun)))
    errors = [float(np.linalg.norm(closest_segments(view["pixels"], projected_hull(result.x, view)),
                                  axis=1).mean()) for view in views]
    if max(errors) > 7 or float(np.mean(errors)) > 5:
        return None
    bread_frame = Rotation.from_rotvec(result.x[3:]).as_matrix() @ mapping
    offset = result.x[:3]
    # Its broad plane must still intersect the small gripping faces. A good
    # silhouette fit to an unrelated bright surface cannot certify a hold.
    contact_plane_distance = float(abs(bread_frame[:, 2] @ offset))
    if contact_plane_distance > .014:
        return None
    if stereo_plane is not None:
        normal = views[0]["frame"] @ bread_frame[:, 2]
        centre = views[0]["tcp"] + views[0]["frame"] @ offset
        side = 1. if normal @ (views[-1]["origin"] - centre) > 0 else -1.
        depth_rms = float(np.sqrt(np.mean(((stereo_plane["points"] - centre) @ normal - side * .0059) ** 2)))
        if depth_rms > .002 or normal @ stereo_plane["normal"] < math.cos(.08):
            return None
    return dict(offset=offset, frame=bread_frame, rotation=result.x[3:],
                pixel_errors=errors, contact_plane_distance=contact_plane_distance,
                reference_offset_change=None if not use_reference else offset - reference_offset)


def _angle_delta(a, b):
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def _rotate_xy(xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def _triangulate(ctx, pixels, cameras, shapes):
    matrices, targets = [], []
    for pixel, camera, shape in zip(pixels, cameras, shapes):
        origin, rotation, fx, fy = ctx.perception.camera_geometry(camera, ctx.state, shape)
        height, width = shape[:2]
        u, v = pixel
        ray = rotation @ np.asarray([(u - width / 2) / fx, -(v - height / 2) / fy, -1])
        ray /= np.linalg.norm(ray)
        normal = np.eye(3) - np.outer(ray, ray)
        matrices.append(normal)
        targets.append(normal @ origin)
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    point = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return point, float(np.linalg.norm(matrix @ point - target))


def _wrist_bread_peaks(ctx, camera, count=4):
    image = _toast_rgb(ctx,camera)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= 14) & (hsv[..., 0] < 42) & (hsv[..., 1] > 4) &
            (hsv[..., 1] < 140) & (hsv[..., 2] > 200)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 300]
    if not contours:
        return [], image.shape
    contour = max(contours, key=cv2.contourArea)
    mask[:] = 0
    cv2.drawContours(mask, [contour], -1, 255, -1)
    columns = np.flatnonzero(mask.any(axis=0))
    top = np.asarray([np.flatnonzero(mask[:, column])[0] for column in columns], float)
    peaks, _ = find_peaks(-top, distance=18, prominence=2)
    points = sorted(([int(columns[i]), float(top[i])] for i in peaks), key=lambda p: p[1])[:count]
    points.sort(key=lambda p: p[0])
    return points, image.shape


def _random_toaster_slots(ctx,image,camera):
    """Fit observed openings at the publicly configured articulation height.

    make_toast_random.yml fixes root Z at80mm relative to Table for both
    variants. Their different visual mesh floors do not set this fixed-root
    articulation height. XY, yaw and variant still come from current RGB.
    This constant is public task geometry, never an episode pose lookup.
    """
    roof=ctx.table_z+.16
    roi=world_roi(ctx,image.shape,(-.01,.48,-.30,.18),roof,camera)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    candidates=[]
    for threshold in (55,78,105,135,180,205):
        mask=((gray<threshold)&(roi>0)).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not 25<cv2.contourArea(contour)<7000:continue
            item=contour_record(ctx,image,contour,roof,camera)
            if item is None or not (.065<item['length']<.205 and .006<item['width']<.055):continue
            if abs(_angle_delta(item['yaw'],-math.pi/2))>math.radians(32):continue
            if any(np.linalg.norm(item['pixel']-old['pixel'])<4 for old in candidates):continue
            candidates.append(item)
    pairs=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            axis=np.array([math.cos(a['yaw']),math.sin(a['yaw'])]);delta=b['position'][:2]-a['position'][:2]
            separation=abs(axis[0]*delta[1]-axis[1]*delta[0]);longitudinal=abs(axis@delta)
            if (abs(_angle_delta(a['yaw'],b['yaw']))>.18 or
                    not .024<separation<.072 or longitudinal>.025):continue
            equal=abs(a['length']-b['length'])
            pairs.append((equal+longitudinal,a,b,separation))
    choices=[]
    for _,a,b,separation in sorted(pairs,key=lambda row:row[0])[:12]:
        # At yaw near -pi/2 the negative local-Y opening lies at lower X.
        observed=sorted((a,b),key=lambda item:item['position'][0])
        outlines=[_uniform_polygon(cv2.convexHull(item['contour'].astype(np.float32))[:,0],48)
                  for item in observed]
        trees=[cKDTree(points) for points in outlines]
        for variant in ((2,) if separation>.041 else (4,)):
            asset=toast_body_asset(variant)
            local=[np.array([[x0,y0,asset['roof']],[x1,y0,asset['roof']],
                             [x1,y1,asset['roof']],[x0,y1,asset['roof']]])
                   for x0,x1,y0,y1 in asset['slots']]
            initial=(a['position']+b['position'])/2
            lower=np.array([.015,-.17,-1.92]);upper=np.array([.39,.025,-1.22])
            seed=np.clip([initial[0],initial[1],-math.pi/2],
                         lower+1e-5,upper-1e-5)
            def projected(parameters):
                x,y,angle=parameters;z=.080;c,s=math.cos(angle),math.sin(angle)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                return [_uniform_polygon(world_pixels(ctx,points@rotation.T+[x,y,ctx.table_z+z],
                                                      image.shape,camera),48) for points in local]
            def residual(parameters):
                predicted=projected(parameters)
                return np.concatenate([np.r_[tree.query(p)[0],cKDTree(p).query(o)[0]]
                                       for p,o,tree in zip(predicted,outlines,trees)])
            fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=70,
                              diff_step=.0005,loss='soft_l1',f_scale=1.5)
            score=float(np.mean(residual(fit.x)))
            if score<3.5:choices.append((score,variant,fit.x))
    if not choices:
        ctx.memory['random_toaster_detection']=dict(components=len(candidates),pairs=len(pairs))
        return None
    score,variant,parameters=min(choices,key=lambda row:row[0]);x,y,yaw=parameters;z=.080
    root=np.array([x,y,ctx.table_z+z]);asset=toast_body_asset(variant)
    slots=[np.r_[root[:2]+_rotate_xy([(x0+x1)/2,(y0+y1)/2],yaw),
                    root[2]+asset['floor']+.0584] for x0,x1,y0,y1 in asset['slots']]
    lever_asset=toast_lever_asset(variant)
    lever=np.r_[root[:2]+_rotate_xy([(lever_asset['x0']+lever_asset['x1'])/2,
                                    (lever_asset['y0']+lever_asset['y1'])/2],yaw),
                root[2]+lever_asset['z1']-lever_asset['lower']]
    ctx.memory['random_toaster_detection']=dict(components=len(candidates),pairs=len(pairs),
        pixel_error=score,variant=variant,root=root.tolist(),yaw=float(yaw),
        public_task_root_height=.080)
    return root,float(yaw),variant,slots,lever


def _toast_slots(ctx, image, camera):
    """Measure the two long dark openings on the observed toaster roof."""
    if ctx.task_name=='make_toast_random':
        return _random_toaster_slots(ctx,image,camera)
    roof = ctx.table_z + .158
    roi = world_roi(ctx, image.shape, (.005, .46, -.28, .17), roof, camera)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1]
    candidates = []
    for threshold in (55, 78, 105, 135, 180, 205):
        mask = ((gray < threshold) & (saturation < 170) & (roi > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 25:
                continue
            item = contour_record(ctx, image, contour, roof, camera)
            if item is None or not (.068 < item["length"] < .20 and .006 < item["width"] < .054):
                continue
            if abs(_angle_delta(item["yaw"], -math.pi / 2)) > math.radians(32):
                continue
            if not any(np.linalg.norm(item["position"][:2] - other["position"][:2]) < .008
                       for other in candidates):
                candidates.append(item)
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            delta = second["position"][:2] - first["position"][:2]
            axis = np.asarray([math.cos(first["yaw"]), math.sin(first["yaw"])])
            separation = abs(axis[0] * delta[1] - axis[1] * delta[0])
            if (abs(_angle_delta(first["yaw"], second["yaw"])) < .20 and
                    .032 < separation < .072 and abs(np.dot(axis, delta)) < .026):
                score = abs(separation - .053) + abs(first["length"] - second["length"])
                pairs.append((score, first, second))
    if not pairs:
        return _toast_merged_slots(ctx, image, camera)
    _, first, second = min(pairs, key=lambda row: row[0])
    yaw = math.atan2(math.sin(2 * first["yaw"]) + math.sin(2 * second["yaw"]),
                     math.cos(2 * first["yaw"]) + math.cos(2 * second["yaw"])) / 2
    if math.cos(yaw + math.pi / 2) < 0:
        yaw += math.pi
    variant = 0 if np.mean([first["length"], second["length"]]) > .136 else 1
    root = (first["position"] + second["position"]) / 2
    root[:2] -= _rotate_xy([-.003, .00025] if variant == 0 else [.0015, -.001], yaw)
    root[2] = ctx.table_z + .080
    fitted = _fit_toaster_outline(ctx, image, camera, root, yaw)
    if fitted is not None:
        root, yaw, variant = fitted
    local_y = (-.026, .0265) if variant == 0 else (-.0285, .0265)
    slots = [np.r_[root[:2] + _rotate_xy([-.003 if variant == 0 else .0015, y], yaw),
                    root[2] + (-.051 if variant == 0 else -.020) + .058] for y in local_y]
    lever = np.r_[root[:2] + _rotate_xy([.119, 0.0 if variant == 0 else -.0007], yaw),
                  root[2] + (.035 if variant == 0 else .009)]
    return root, yaw, variant, slots, lever


def _uniform_polygon(vertices, count=128):
    vertices = np.asarray(vertices, float)
    ends = np.roll(vertices, -1, axis=0)
    lengths = np.linalg.norm(ends - vertices, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    distances = np.linspace(0, cumulative[-1], count, endpoint=False)
    indices = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"),
                         len(vertices) - 1)
    fractions = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-8)
    return vertices[indices] + fractions[:, None] * (ends[indices] - vertices[indices])


def _toast_merged_slots(ctx, image, camera):
    """Recognise toaster1's single rim and its lower white central divider.

    Its public mesh has a 120.8 by 94.4 mm outer aperture. The divider is
    11.3 mm below the rim, so both dark slots can form one image component.
    A rim fit alone is insufficient: the visible divider, both dark halves,
    and the silver roof must agree with this same observed rigid transform.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    roof = ctx.table_z + .158
    roi = world_roi(ctx, image.shape, (.005, .46, -.28, .17), roof, camera)
    rim = np.asarray([[-.057931, -.047920, .075138],
                      [.062866, -.047920, .075138],
                      [.062866, .046452, .075138],
                      [-.057931, .046452, .075138]])
    choices = []
    for threshold in (135, 180, 205):
        mask = ((gray < threshold) & (hsv[..., 1] < 170) & (roi > 0)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 300:
                continue
            item = contour_record(ctx, image, contour, roof, camera)
            if item is None or not (.105 < item["length"] < .136 and .077 < item["width"] < .110):
                continue
            yaw = item["yaw"]
            if math.cos(yaw + math.pi / 2) < 0:
                yaw -= math.pi
            if abs(_angle_delta(yaw, -math.pi / 2)) > math.radians(32):
                continue
            observed = _uniform_polygon(cv2.convexHull(contour.astype(np.float32))[:, 0])
            tree = cKDTree(observed)

            def projected(parameters, local=rim):
                angle = parameters[2]
                c, s = math.cos(angle), math.sin(angle)
                rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                root = np.r_[parameters[:2], ctx.table_z + .080]
                return world_pixels(ctx, local @ rotation.T + root, image.shape, camera)

            def residual(parameters):
                predicted = _uniform_polygon(projected(parameters))
                return np.r_[tree.query(predicted)[0], cKDTree(predicted).query(observed)[0]]

            seed = np.r_[item["position"][:2] - _rotate_xy([.00247, -.00073], yaw), yaw]
            seed = np.clip(seed, [.0051, -.2799, -1.8999], [.4599, .1699, -1.2201])
            fit = least_squares(residual, seed,
                                bounds=([.005, -.28, -1.90], [.46, .17, -1.22]),
                                max_nfev=100, diff_step=.002, loss="soft_l1", f_scale=2)
            errors = residual(fit.x)
            if errors.mean() > 2.5 or np.quantile(errors, .9) > 4:
                continue

            def colors(local):
                pixels = projected(fit.x, np.asarray(local))
                h, w = image.shape[:2]
                if not ((pixels > [3, 3]) & (pixels < [w - 3, h - 3])).all():
                    return None
                pixels = np.rint(pixels).astype(int)
                return hsv[pixels[:, 1], pixels[:, 0]]

            # Sample the actual public divider's narrow central top, away
            # from its rounded ends. The two cavities are sampled below it.
            divider = colors([[x, y, .063813] for x in np.linspace(-.045, .050, 12)
                              for y in np.linspace(-.00473, .00327, 5)])
            halves = [colors([[x, y, .055] for x in np.linspace(-.045, .050, 12)
                              for y in np.linspace(center - .004, center + .004, 5)])
                      for center in (-.0285, .0265)]
            silver = colors([[x, y, .075138] for x in np.linspace(-.095, .095, 14)
                             for y in (-.063, .061)] +
                            [[x, y, .075138] for x in (-.093, .096)
                             for y in np.linspace(-.06, .06, 10)])
            if divider is None or silver is None or any(part is None for part in halves):
                continue
            white = ((divider[:, 1] < 35) & (divider[:, 2] > 180)).mean()
            dark = [((part[:, 1] < 170) & (part[:, 2] < 180)).mean() for part in halves]
            roof_gray = ((silver[:, 1] < 35) & (silver[:, 2] > 88)).mean()
            if white < .70 or min(dark) < .65 or roof_gray < .70:
                continue
            choices.append((float(errors.mean()), fit.x, white, dark, roof_gray))
    if not choices:
        return None
    error, parameters, white, dark, roof_gray = min(choices, key=lambda row: row[0])
    root, yaw = np.r_[parameters[:2], ctx.table_z + .080], float(parameters[2])
    slots = [np.r_[root[:2] + _rotate_xy([.0015, y], yaw), root[2] + .038]
             for y in (-.0285, .0265)]
    lever = np.r_[root[:2] + _rotate_xy([.119, -.0007], yaw), root[2] + .009]
    ctx.memory["toast_merged_rim"] = dict(pixel_error=error, divider_white=float(white),
                                            dark_halves=[float(x) for x in dark],
                                            roof_gray=float(roof_gray))
    return root, yaw, 1, slots, lever


def _fit_toaster_outline(ctx, image, camera, root, yaw):
    """Compare the observed outline against the two public rigid body boxes."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    roi = world_roi(ctx, image.shape, (.02, .46, -.28, .15), ctx.table_z + .10, camera)
    mask = ((hsv[..., 1] < 24) & (hsv[..., 2] > 88) & (roi > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1400]
    if not contours:
        return None
    contour = min(contours, key=lambda c: np.linalg.norm(
        np.mean(c[:, 0, :], axis=0) - ctx.perception.world_to_pixel(root, camera, ctx.state, image.shape)))
    observed = contour[:, 0, :].astype(float)
    observed = observed[::max(1, len(observed) // 160)]
    observed_tree = cKDTree(observed)
    boxes = [(-.1051, .1022, -.0783, .0772, -.0817, .0802),
             (-.1108, .1157, -.0706, .0691, -.0759, .0751)]
    choices = []
    for variant, box in enumerate(boxes):
        x0, x1, y0, y1, z0, z1 = box
        vertices = np.asarray([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])

        def perimeter(parameters):
            x, y, angle = parameters
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            points = vertices @ rotation.T + [x, y, ctx.table_z + .080]
            projected = world_pixels(ctx, points, image.shape, camera)
            hull = cv2.convexHull(projected.astype(np.float32))[:, 0, :]
            # A convex hull can change its vertex count during optimisation.
            # Fixed arc-length samples keep the least-squares residual fixed.
            return _uniform_polygon(hull, count=128)

        def residual(parameters):
            predicted = perimeter(parameters)
            return np.r_[observed_tree.query(predicted)[0], cKDTree(predicted).query(observed)[0]]

        angle = (yaw + math.pi) % (2 * math.pi) - math.pi
        result = least_squares(residual, [root[0], root[1], angle],
                               bounds=([.025, -.22, -1.90], [.38, .05, -1.22]),
                               max_nfev=100, diff_step=.002, loss="soft_l1", f_scale=2)
        score = float(np.mean(np.minimum(residual(result.x), 8)))
        choices.append((score, result.x, variant))
    score, parameters, variant = min(choices, key=lambda row: row[0])
    if score > 5:
        return None
    return np.asarray([parameters[0], parameters[1], ctx.table_z + .080]), float(parameters[2]), variant


def _rack_apparent_length_limit(ctx, item, camera, image_shape, count):
    """Project a public bread/rack sphere onto the measuring plane.

    Each slice has a 74.8 mm asset radius, and rack seats are 25 mm apart.
    A horizontal projection of upright geometry is larger than its physical
    footprint. The oblique-view bound accounts for that without fitting an
    upper limit to one episode's contour length.
    """
    radius = .0748 + .025 * (max(1, int(count)) - 1) / 2
    origin, _, _, _ = ctx.perception.camera_geometry(camera, ctx.state, image_shape)
    try:
        centre = ctx.perception.project(item["pixel"], ctx.table_z + .0731,
                                         camera, ctx.state, image_shape)
    except ValueError:
        return math.hypot(.1169, .1161) + .025 * (count - 1) + .035
    delta = np.asarray(centre) - origin
    denominator = delta[2] ** 2 - radius ** 2
    if denominator <= 1e-6:
        return math.hypot(.1169, .1161) + .025 * (count - 1) + .035
    distance = math.sqrt(max(float(delta @ delta) - radius ** 2, 0.0))
    plane_distance = abs(origin[2] - (ctx.table_z + .101))
    diameter = 2 * plane_distance * radius * distance / denominator
    # Colour centroids need not coincide with the 3D stack centre. Allow a
    # small perspective variation plus subpixel boundary uncertainty.
    return float(diameter * 1.10 + .005)


def bread_candidates(ctx, camera=None, count=4):
    image, head = head_image(ctx)
    camera = head if camera is None else camera
    if camera != head:
        image = _toast_rgb(ctx,camera)
    learned=_learned_toast_mask(ctx,'bread',camera) if camera==head else None
    if learned is not None:
        pieces,_=cv2.findContours(learned.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        found=[]
        for contour in pieces:
            if not 20<cv2.contourArea(contour)<image.shape[0]*image.shape[1]*.10:continue
            item=contour_record(ctx,image,contour,ctx.table_z+.101,camera)
            if item is None or not .035<item['length']<_rack_apparent_length_limit(ctx,item,camera,image.shape,count):continue
            if item['position'][0]>.05 or not -.31<item['position'][1]<.16:continue  # shelf x range [-0.35, 0] plus slack
            found.append(item)
        # A detector may correctly segment only one slice. Never infer four
        # seats from that partial envelope; require the visible public row.
        if found and _random_bread_grid(ctx,found,image,camera,count) is not None:
            return found
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    roi = (world_roi(ctx, image.shape, (-.46, .07, -.31, .16), ctx.table_z + .10, camera)
           if "head" in camera else np.full(image.shape[:2], 255, np.uint8))
    mask = ((h >= 14) & (h < 42) & (s > 4) & (s < 140) & (v > 200) & (roi > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        if not 20 < cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .10:
            continue
        item = contour_record(ctx, image, contour, ctx.table_z + .101, camera)
        if item is None or not .035 < item["length"] < _rack_apparent_length_limit(
                ctx, item, camera, image.shape, count):
            continue
        if item["position"][0] > .05 or not -.31 < item["position"][1] < .16:
            continue
        if not any(np.linalg.norm(item["pixel"] - other["pixel"]) < 10 for other in found):
            found.append(item)
    return found


def _random_bread_grid(ctx,breads,image,camera,count):
    """Choose an observed shelf row, not the highest unrelated warm blobs."""
    from itertools import combinations
    if not breads:return None
    pixels=[]
    # A taller clutter object can share image columns with the rack. Its
    # unrelated top edge must not erase the lower bread component's peaks.
    for item in breads:
        mask=item['mask'];columns=np.flatnonzero(mask.any(axis=0))
        if len(columns)<4:continue
        top=np.asarray([np.flatnonzero(mask[:,c])[0] for c in columns],float)
        peaks,_=find_peaks(-top,distance=5,prominence=1.)
        proposed=[[float(columns[i]),float(top[i])] for i in peaks]
        for run in np.split(columns,np.flatnonzero(np.diff(columns)>1)+1):
            if len(run)<3:continue
            height=[np.flatnonzero(mask[:,c])[0] for c in run]
            proposed.append([float(run[int(np.argmin(height))]),float(min(height))])
        for pixel in proposed:
            if any(np.linalg.norm(np.array(pixel)-old[0])<3 for old in pixels):continue
            pixels.append((np.array(pixel),mask))
    top_z=ctx.table_z+.0381+.035+.0584
    measured=[]
    for pixel,mask in pixels:
        point=ctx.perception.project(pixel,top_z,camera,ctx.state,image.shape)
        if not (-.46<point[0]<.055 and -.24<point[1]<.09):continue
        column=int(round(pixel[0]));rows=np.flatnonzero(mask[:,column])
        bottom=ctx.perception.world_to_pixel(point-[0.,0.,.11],camera,ctx.state,image.shape)
        required=abs(bottom[1]-pixel[1])*.45
        nearby=rows[rows>=pixel[1]]
        if not len(nearby) or nearby[-1]-pixel[1]<required:continue
        measured.append(point)
    if len(measured)<count:return None
    # Bound clutter complexity without selecting only the highest objects.
    measured=sorted(measured,key=lambda p:p[0])[:28]
    best=None
    for points in combinations(measured,count):
        points=np.asarray(points);axis=points[-1,:2]-points[0,:2]
        yaw=math.atan2(axis[1],axis[0])
        if abs(yaw)>.30:continue
        normal=np.array([math.cos(yaw),math.sin(yaw)])
        side=np.array([-normal[1],normal[0]])
        along=points[:,:2]@normal;cross=points[:,:2]@side
        spacing=float(np.max(np.abs(np.diff(along)-.025)))
        straight=float(np.ptp(cross))
        if spacing>.007 or straight>.009:continue
        score=spacing+straight
        if best is None or score<best[0]:best=(score,points,yaw)
    if best is None:return None
    score,points,yaw=best
    offsets=(np.arange(count)-(count-1)/2)*.025
    def row_error(parameters):
        x,y,angle=parameters
        return (np.c_[x+offsets*math.cos(angle),y+offsets*math.sin(angle)]-points[:,:2]).ravel()
    fit=least_squares(row_error,[*points[:,:2].mean(0),yaw],
                      bounds=([-.47,-.25,-.30],[.06,.10,.30]))
    x,y,yaw=fit.x;side=np.array([-math.sin(yaw),math.cos(yaw)])
    fitted=np.c_[x+offsets*math.cos(yaw),y+offsets*math.sin(yaw),np.full(count,top_z)]
    centres=[point-np.r_[side*.048,.012] for point in fitted]
    ctx.memory['random_bread_grid']=dict(count=count,residual=score,points=points.tolist(),yaw=yaw)
    return centres,yaw,.8


def measure_breads(ctx, count=4):
    image, camera = head_image(ctx)
    breads = bread_candidates(ctx, count=count)
    if not breads:
        ctx.memory["bread_detection"] = dict(bread_components=0, expected_count=count)
        return None
    if ctx.task_name in ('make_toast_random', 'make_toast'):
        grid=_random_bread_grid(ctx,breads,image,camera,count)
        if grid is not None:return grid
    mask = np.maximum.reduce([item["mask"] for item in breads])
    columns = np.flatnonzero(mask.any(axis=0))
    edge_peaks = []
    if len(columns) > 20:
        top = np.asarray([np.flatnonzero(mask[:, column])[0] for column in columns], float)
        peaks, _ = find_peaks(-top, distance=8, prominence=2)
        edge_peaks = sorted(([int(columns[i]), float(top[i])] for i in peaks), key=lambda p: p[0])
    if len(edge_peaks) >= count:
        # Upper silhouette peaks belong to the upright slices. A
        # known bread/rack model supplies the upper edge's elevation and width.
        edge_peaks = sorted(edge_peaks, key=lambda p: p[1])[:count]
        edge_peaks.sort(key=lambda p: p[0])
        top_z = ctx.table_z + .0381 + .035 + .0584
        measured = np.asarray([ctx.perception.project(p, top_z, camera, ctx.state, image.shape)
                               for p in edge_peaks])
        wrist = next((name for name in ctx.vision if "left" in name and "wrist" in name), None)
        if wrist is not None:
            wrist_peaks, wrist_shape = _wrist_bread_peaks(ctx, wrist, count)
            if len(wrist_peaks) == count:
                triangulated = [_triangulate(ctx, [head, close], [camera, wrist], [image.shape, wrist_shape])
                                for head, close in zip(edge_peaks, wrist_peaks)]
                if all(error < .006 and ctx.table_z + .10 < point[2] < ctx.table_z + .18
                       for point, error in triangulated):
                    measured = np.asarray([point for point, error in triangulated])
                    ctx.memory["bread_stereo"] = dict(points=measured.tolist(),
                                                       ray_errors=[error for point, error in triangulated])
        row_axis = measured[-1, :2] - measured[0, :2]
        rack_yaw = float(np.clip(math.atan2(row_axis[1], row_axis[0]), -.27, .27))
        side = np.asarray([-math.sin(rack_yaw), math.cos(rack_yaw)])
        # This contact convention is retained from the observed successful
        # edge pinch. Colour-profile peaks need not be the same CAD vertex
        # in both cameras, so a face-thickness adjustment is not inferred.
        centers = [point - np.r_[side * .048, .012] for point in measured]
        confidence = .8
    elif len(breads) >= 3:
        centers = np.asarray([item["position"][:2] for item in breads])
        _, vectors = np.linalg.eigh(np.cov(centers.T))
        row_axis = vectors[:, -1]
        rack_yaw = math.atan2(row_axis[1], row_axis[0])
        rack_yaw = _angle_delta(rack_yaw, 0)
        if abs(rack_yaw) > .45:
            rack_yaw = _angle_delta(float(np.median([b["yaw"] for b in breads])) + math.pi / 2, 0)
        candidates = sorted(breads, key=lambda item: item["position"][0])
        centers = [item["position"].copy() for item in candidates]
        confidence = .7
    else:
        # A slice can hide the other three faces. The observed stack envelope
        # anchors a public rack model; its four 25 mm spacings are asset geometry.
        points = np.concatenate([item["world_contour"][:, :2] for item in breads])
        rect = cv2.minAreaRect(points.astype(np.float32))
        corners = cv2.boxPoints(rect)
        edges = np.roll(corners, -1, axis=0) - corners
        axis = edges[np.argmin(np.linalg.norm(edges, axis=1))]
        rack_yaw = _angle_delta(math.atan2(axis[1], axis[0]), 0)
        rack_yaw = float(np.clip(rack_yaw, -.27, .27))
        center = np.asarray(rect[0])
        offsets = (np.arange(count) - (count - 1) / 2) * .025
        centers = [np.r_[center + _rotate_xy([offset, 0], rack_yaw), ctx.table_z + .101]
                   for offset in offsets]
        confidence = .45
    if len(edge_peaks) < count:
        for center in centers:
            center[2] = ctx.table_z + .120
    return centers, rack_yaw, confidence


def measure_toast(ctx):
    image, camera = head_image(ctx)
    toaster = _toast_slots(ctx, image, camera)
    if ctx.task_name=='make_toast_random' and toaster is not None and 'toast_white_gain' not in ctx.memory:
        root,yaw,variant,_,_=toaster;asset=toast_body_asset(variant)
        x0,x1,y0,y1=asset['footprint']
        vertices=np.array([[x,y,z] for x in (x0,x1) for y in (y0,y1)
                           for z in (-.075,asset['roof'])])
        c,s=math.cos(yaw),math.sin(yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        polygon=cv2.convexHull(world_pixels(ctx,vertices@rotation.T+root,image.shape,camera).astype(np.float32))
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),255)
        mask=cv2.erode(mask,np.ones((9,9),np.uint8))>0
        paint=image[mask];paint=paint[(paint.min(1)>95)&(paint.max(1)>175)]
        if len(paint)>400:
            white=np.median(paint,axis=0)
            gain=np.clip(float(white.mean())/np.maximum(white,1),.70,1.40)
            ctx.memory['toast_white_gain']=gain.tolist()
    breads = measure_breads(ctx)
    if toaster is None or breads is None:
        ctx.memory["toast_detection"] = dict(slots_found=toaster is not None,
                                              bread_found=breads is not None)
        return None
    root, yaw, variant, slots, lever = toaster
    centers, rack_yaw, confidence = breads
    return ToastScene(root, yaw, variant, slots, lever, centers, rack_yaw, confidence)


def appearance_descriptor(image, contour, bbox):
    x, y, width, height = bbox
    crop = image[y:y + height, x:x + width]
    if not crop.size:
        return None
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
    local = mask[y:y + height, x:x + width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], local, [18, 8], [0, 180, 0, 256]).reshape(-1)
    hist = hist / max(hist.sum(), 1)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    thumbnail = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
    thumbnail -= thumbnail.mean()
    thumbnail /= max(np.linalg.norm(thumbnail), 1e-6)
    hu = cv2.HuMoments(cv2.moments(local)).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-12))
    pixels = crop[local > 0]
    return dict(hist=hist, thumbnail=thumbnail, hu=hu, rgb=pixels.mean(axis=0),
                aspect=width / max(height, 1), area=cv2.contourArea(contour))


def appearance_distance(first, second):
    if first is None or second is None:
        return float("inf")
    hist = np.sqrt(np.maximum(first["hist"], 0) * np.maximum(second["hist"], 0)).sum()
    color = np.linalg.norm(first["rgb"] - second["rgb"]) / 255
    aspect = abs(math.log(max(first["aspect"], .01) / max(second["aspect"], .01)))
    shape = np.mean(np.minimum(np.abs(first["hu"][:4] - second["hu"][:4]), 5)) / 5
    correlation = max(float(np.sum(first["thumbnail"] * np.rot90(second["thumbnail"], rotation)))
                      for rotation in range(4))
    return .48 * (1 - hist) + .18 * color + .12 * min(aspect, 1.5) + .12 * shape + .10 * (1 - correlation)


def measure_reference_stereo(ctx, reference_item):
    """Match the sole reference's colour/shape in a second RGB camera."""
    head, head_camera = head_image(ctx)
    descriptor = reference_item["descriptor"]
    hue_hist = descriptor["hist"].reshape(18, 8).sum(axis=1)
    hue = float(np.argmax(hue_hist) * 10 + 5)
    ref_hsv = cv2.cvtColor(np.asarray(descriptor["rgb"], np.uint8).reshape(1, 1, 3),
                           cv2.COLOR_RGB2HSV)[0, 0]
    choices = []
    for camera in ctx.vision:
        if "wrist" not in camera:
            continue
        image = _toast_rgb(ctx,camera)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        distance = np.minimum(abs(hsv[..., 0].astype(float) - hue),
                              180 - abs(hsv[..., 0].astype(float) - hue))
        if ref_hsv[1] > 55:
            mask = (distance < 18) & (hsv[..., 1] > 45) & (hsv[..., 2] > 45)
        else:
            mask = ((hsv[..., 1] < 60) & (abs(hsv[..., 2].astype(float) - ref_hsv[2]) < 50))
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not 30 < cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .14:
                continue
            moments = cv2.moments(contour)
            pixel = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            other = appearance_descriptor(image, contour, cv2.boundingRect(contour))
            similarity = appearance_distance(descriptor, other)
            if similarity > .30:
                continue
            point, error = _triangulate(ctx, [reference_item["pixel"], pixel],
                                        [head_camera, camera], [head.shape, image.shape])
            if error > .012 or not .73 < point[2] < 1.04 or abs(point[0]) > 1:
                continue
            choices.append((similarity + error * 4, point, error, camera, pixel))
    if not choices:
        return None
    _, point, error, camera, pixel = min(choices, key=lambda row: row[0])
    ctx.memory["conveyor_reference_stereo"] = dict(point=point.tolist(), ray_error=error,
                                                   camera=camera, pixel=pixel)
    return point


def conveyor_objects(ctx, previous_image=None):
    """Segment object colours against the currently measured belt colour."""
    image, camera = head_image(ctx)
    roi = world_roi(ctx, image.shape, (-1.1, 1.1, -.17, .56), BELT_Z + .022, camera)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    flat = lab[(roi > 0) & (gradient < 12)]
    if len(flat) < 100:
        return [], image
    # The most populated 16-level RGB bin measures the belt rather than an
    # arbitrary fixed colour. This permits bright, grey, and textured belts.
    quantized = (flat / 16).astype(np.int32)
    encoded = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
    values, counts = np.unique(encoded, return_counts=True)
    dominant = values[np.argmax(counts)]
    background = np.median(flat[encoded == dominant], axis=0)
    difference = np.linalg.norm((lab - background) * [.65, 1.0, 1.0], axis=2)
    foreground = (difference > 22) & (roi > 0)
    if previous_image is not None and previous_image.shape == image.shape:
        motion = cv2.absdiff(image, previous_image).max(axis=2) > 10
        motion = cv2.dilate(motion.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # Keep textureless object interiors, while rejecting static belt rails.
        foreground &= cv2.dilate(motion.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    mask = foreground.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coloured = []
    for lower, upper in ((0, 12), (12, 35), (35, 90), (90, 135), (135, 180)):
        region = ((hsv[..., 0] >= lower) & (hsv[..., 0] < upper) &
                  (hsv[..., 1] > 80) & (hsv[..., 2] > 65) & (roi > 0)).astype(np.uint8) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        pieces, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coloured.extend(piece for piece in pieces if cv2.contourArea(piece) > 30)
    contours = coloured + list(contours)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 30 < area < image.shape[0] * image.shape[1] * .09:
            continue
        item = contour_record(ctx, image, contour, BELT_Z + .028, camera)
        if item is None or not .015 < item["length"] < .23 or item["width"] > .19:
            continue
        if not (-.95 < item["position"][0] < .95 and -.14 < item["position"][1] < .48):
            continue
        if hasattr(ctx.perception, "on_robot") and ctx.perception.on_robot(item["pixel"], ctx.state, image.shape):
            continue
        item["descriptor"] = appearance_descriptor(image, contour, item["bbox"])
        item["height"] = .048
        item["position"][2] = BELT_Z + item["height"] * .48
        if not any(np.linalg.norm(item["pixel"] - other["pixel"]) < 10 for other in found):
            found.append(item)
    return sorted(found, key=lambda item: item["position"][0]), image


class ConveyorTracker:
    """Measured identities and velocity; elapsed time comes from action steps."""
    def __init__(self, frequency=25):
        self.frequency = frequency
        self.tracks = {}
        self.next_id = 0
        self.previous_image = None
        self.reference = None
        self.reference_id = None
        self.reference_gone = False
        self.velocity = np.zeros(2)
        self.reference_z = None
        self.contact_z = .783
        self.reference_velocity_samples = []
        self.belt_velocity = None

    def observe(self, ctx):
        items, image = conveyor_objects(ctx, self.previous_image)
        if not items and self.previous_image is not None:
            items, _ = conveyor_objects(ctx)
        self.previous_image = image.copy()
        if self.reference_z is not None:
            for item in items:
                item["position"] = ctx.perception.project(item["pixel"], self.reference_z,
                                                          item["camera"], ctx.state, image.shape)
        step = int(ctx.steps)
        unused = set(self.tracks)
        visible = []
        for item in items:
            possible = []
            for key in unused:
                track = self.tracks[key]
                dt = (step - track["last_step"]) / self.frequency
                predicted = track["position"][:2] + track["velocity"] * dt
                distance = np.linalg.norm(item["position"][:2] - predicted)
                similarity = appearance_distance(track["descriptor"], item["descriptor"])
                if distance < .10 + abs(dt) * .06 and similarity < .40:
                    possible.append((distance + similarity * .10, key))
            if possible:
                _, key = min(possible)
                old = self.tracks[key]
                dt = (step - old["last_step"]) / self.frequency
                velocity = old["velocity"].copy()
                if dt > 0:
                    measured = (item["position"][:2] - old["position"][:2]) / dt
                    if np.linalg.norm(measured) < .65:
                        velocity = .4 * velocity + .6 * measured
                first_step = old["first_step"]
                unused.discard(key)
            else:
                key = self.next_id
                self.next_id += 1
                velocity = self.velocity.copy()
                first_step = step
            item.update(id=key, velocity=velocity, last_step=step, first_step=first_step)
            self.tracks[key] = item
            visible.append(item)
        good_velocities = [t["velocity"] for t in visible if np.linalg.norm(t["velocity"]) > .015]
        if good_velocities:
            self.velocity = np.median(good_velocities, axis=0)
        if self.reference is None and visible and np.linalg.norm(self.velocity) > .015:
            direction = self.velocity / np.linalg.norm(self.velocity)
            # The first object is the leading object in the observed direction,
            # not a simulator label or a seed-specific object identity.
            first = max(visible, key=lambda item: np.dot(item["position"][:2], direction))
            self.reference = first["descriptor"]
            self.reference_id = first["id"]
            measured = measure_reference_stereo(ctx, first)
            if measured is not None:
                self.reference_z = float(measured[2])
                # The public jaw gripping face is centred 5 mm in front of
                # Context's TCP, so place the TCP slightly above the body centre.
                self.contact_z = max(.781, self.reference_z + .012)
                for track in self.tracks.values():
                    track["position"] = ctx.perception.project(track["pixel"], self.reference_z,
                                                              track["camera"], ctx.state, image.shape)
        if self.reference_id is not None:
            reference_track = self.tracks.get(self.reference_id)
            if (reference_track is not None and reference_track["last_step"] == step and
                    20 < reference_track["pixel"][0] < image.shape[1] - 20 and
                    np.linalg.norm(reference_track["velocity"]) > .04):
                self.reference_velocity_samples.append(reference_track["velocity"].copy())
                if len(self.reference_velocity_samples) >= 4:
                    self.belt_velocity = np.median(self.reference_velocity_samples[-20:], axis=0)
                    self.belt_velocity[np.abs(self.belt_velocity) < .012] = 0.0
            if (reference_track is not None and step - reference_track["last_step"] > 20):
                self.reference_gone = True
        return visible

    def matching(self, visible):
        if self.reference is None:
            return []
        choices = [(appearance_distance(self.reference, item["descriptor"]), item) for item in visible
                   if item["id"] != self.reference_id and .025 < item["width"] < .14]
        choices.sort(key=lambda row: row[0])
        return [item for score, item in choices if score < .26 and
                item["area"] > self.reference["area"] * .55]
