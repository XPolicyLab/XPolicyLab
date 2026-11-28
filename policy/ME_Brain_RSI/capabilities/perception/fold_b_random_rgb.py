"""Fuse corresponding RGB-derived landmarks or hem sources without contour-index mixing."""
from copy import copy, deepcopy
import math

import cv2
import numpy as np

from coap.tasks.fold_candidate_b import landmarks
from coap.tasks.fold_candidate_a import contour_world


def _texture_features(image):
    """Reuse the fixed chroma and isotropic-texture representation from the prior round."""
    gray = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.GaussianBlur(gx*gx, (0, 0), 2.)
    xy = cv2.GaussianBlur(gx*gy, (0, 0), 2.)
    yy = cv2.GaussianBlur(gy*gy, (0, 0), 2.)
    eigenvalue = .5*(xx+yy-np.sqrt((xx-yy)**2+4.*xy*xy))
    texture = np.clip(np.sqrt(np.maximum(eigenvalue, 0.))/.16, 0., 1.)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return np.stack([lab[:, :, 1], lab[:, :, 2], np.uint8(255.*texture)], axis=-1)


def _shape_cost(ctx, contour, expected_width, initial):
    """Use the unchanged broad footprint model to quantify alternative RGB evidence."""
    cloud = np.asarray([ctx.project(pixel, ctx.table_z+.006, 'cam_head')[:2]
                        for pixel in contour], float)
    extent = np.ptp(cloud, axis=0)
    area = float(abs(cv2.contourArea(cloud.astype(np.float32))))
    _, vectors = np.linalg.eigh(np.cov(cloud.T))
    axis = vectors[:, -1]
    angle = (math.atan2(axis[1], axis[0])+math.pi/2.) % math.pi-math.pi/2.
    orientation = max(abs(angle)-math.radians(25.), 0.)/math.radians(65.)
    expected_area = .105 if initial else .060
    cost = ((extent[0]-expected_width)/.18)**2+((extent[1]-.33)/.14)**2
    cost += ((area-expected_area)/.055)**2+(3. if initial else 0.)*orientation**2
    return float(cost), dict(world_extent_m=extent.tolist(), world_area_m2=area,
                             major_axis_angle_degrees=math.degrees(angle))


def _texture_item(ctx, seed_world, initial):
    """Estimate one alternative on the same real frame and original pre-measurement seed."""
    camera = 'cam_head'
    image = np.asarray(ctx.vision[camera]['color'], np.uint8)[..., :3]
    seed_world = np.asarray(seed_world, float)
    seed = np.rint(ctx.perception.world_to_pixel(seed_world, camera, ctx.state,
                                               image.shape)).astype(int)
    if initial:
        corners = [[x, y, ctx.table_z+.006]
                   for x in (-.38, .38) for y in (-.39, .09)]
    else:
        center = seed_world[:2]
        corners = [[center[0]+dx, center[1]+dy, ctx.table_z+.006]
                   for dx in (-.40, .40) for dy in (-.26, .28)]
    pixels = np.asarray([ctx.perception.world_to_pixel(point, camera, ctx.state,
                                                      image.shape) for point in corners])
    lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int), 0)
    upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int),
                       [image.shape[1]-1, image.shape[0]-1])
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[lower[1]:upper[1], lower[0]:upper[0]] = cv2.GC_PR_BGD
    cv2.circle(mask, tuple(seed), 15, cv2.GC_FGD, -1)
    cv2.setRNGSeed(0)
    cv2.grabCut(_texture_features(image), mask, None, np.zeros((1, 65)),
                np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    material = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))*255
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(material, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)[:, 0].astype(float)
    if len(contour) < 3:
        return None
    x, y, width, height = cv2.boundingRect(contour.astype(np.float32))
    moments = cv2.moments(contour.astype(np.float32))
    pixel = [moments['m10']/max(moments['m00'], 1.),
             moments['m01']/max(moments['m00'], 1.)]
    return dict(bbox=(x, y, width, height), contour=contour, camera=camera,
                pixel=pixel, position=ctx.project(pixel, ctx.table_z+.006, camera))


def _evidence(ctx, raw, alternative, initial):
    """Continuously attenuate small cost advantages; never accept or reject an action."""
    expected_width = .52 if initial else .26
    raw_cost, raw_geometry = _shape_cost(ctx, raw['contour'], expected_width, initial)
    alt_cost, alt_geometry = _shape_cost(ctx, alternative['contour'], expected_width, initial)
    improvement = max(raw_cost-alt_cost, 0.)
    alpha = improvement/(improvement+.25)
    return alpha, dict(raw_cost=raw_cost, alternative_cost=alt_cost,
        positive_cost_improvement=improvement, alpha=alpha,
        raw_bbox=list(raw['bbox']), alternative_bbox=list(alternative['bbox']),
        raw_pixel_area=float(cv2.contourArea(np.asarray(raw['contour'], np.float32))),
        alternative_pixel_area=float(cv2.contourArea(np.asarray(alternative['contour'], np.float32))),
        raw_geometry=raw_geometry, alternative_geometry=alt_geometry)


def blend_initial_landmarks(ctx, raw_item):
    """Blend matching named semantic points while leaving all raw measurement state intact."""
    raw_points = landmarks(ctx, raw_item)
    if raw_points is None:
        return raw_points
    alternative = _texture_item(ctx, [0., -.18, ctx.table_z+.006], True)
    if alternative is None:
        return raw_points
    proxy = copy(ctx)
    proxy.memory = deepcopy(ctx.memory)
    alt_points = landmarks(proxy, alternative)
    if alt_points is None or set(raw_points) != set(alt_points):
        return raw_points
    alpha, evidence = _evidence(ctx, raw_item, alternative, True)
    result = {name:(1.-alpha)*np.asarray(point)+alpha*np.asarray(alt_points[name])
              for name,point in raw_points.items()}
    ctx.memory['fold_r06_initial_blend'] = dict(step=int(ctx.steps), evidence=evidence,
        correspondence='matching_semantic_landmark_names',
        raw_points={name:np.asarray(point).tolist() for name,point in raw_points.items()},
        alternative_points={name:np.asarray(point).tolist() for name,point in alt_points.items()},
        blended_points={name:point.tolist() for name,point in result.items()},
        raw_measurement_seed_and_segmentation_log_preserved=True)
    return result


def blend_hem_sources(ctx, raw_item, raw_sources, hints, width, inward,
                      corner_contact, seed_world):
    """Blend the two corresponding pre-hover grasp points, then continue the fixed plan."""
    if raw_item is None:
        return raw_sources
    alternative = _texture_item(ctx, seed_world, False)
    if alternative is None:
        return raw_sources
    contour = contour_world(ctx, alternative)
    alt_sources = [corner_contact(ctx, hint, contour, width, inward, index)
                   for index,hint in enumerate(hints)]
    alpha, evidence = _evidence(ctx, raw_item, alternative, False)
    result = [(1.-alpha)*np.asarray(raw)+alpha*np.asarray(alt)
              for raw,alt in zip(raw_sources,alt_sources)]
    ctx.memory['fold_r06_hem_blend'] = dict(step=int(ctx.steps), evidence=evidence,
        correspondence='same_left_right_hint_index_before_hover',
        raw_sources=[np.asarray(point).tolist() for point in raw_sources],
        alternative_sources=[np.asarray(point).tolist() for point in alt_sources],
        blended_sources=[point.tolist() for point in result],
        raw_measurement_seed_and_segmentation_log_preserved=True)
    return result
