"""Destination support height from calibrated silhouettes, not asset dimensions."""
import cv2
import numpy as np
from . import Camera, GeometryUnavailable
from .supported_outline import fit_supported_outline


def observe_pad_surface(observation, segment_fn, plane, destination_xy):
    """Associate a visible convex pad across views by its support contact point.

    A generic upright extrusion is fitted to the silhouettes. Missing, clipped,
    ambiguous or inconsistent evidence must not fall back to a stored thickness.
    """
    coefficients=np.asarray(plane,dtype=float)
    xy=np.asarray(destination_xy,dtype=float)
    if coefficients.shape!=(3,) or xy.shape!=(2,) or not np.isfinite(np.r_[coefficients,xy]).all():
        raise GeometryUnavailable('Observed plane and destination XY required')
    contact=np.r_[xy,coefficients@np.r_[xy,1.]]
    views=[]
    for name in observation['vision']:
        camera=Camera.from_observation(observation,name)
        pixels,depth=camera.project(contact[None])
        u,v=np.rint(pixels[0]).astype(int)
        if depth[0]<=0 or not (0<=u<camera.width and 0<=v<camera.height):
            continue
        groups=segment_fn(camera.image,[('round wooden mat',.5)])
        associated=[]
        for group in groups:
            for mask,score in zip(group.masks,group.scores):
                if score<.5:
                    continue
                mask=np.asarray(mask,bool)
                if mask.shape!=(camera.height,camera.width):
                    mask=cv2.resize(mask.astype(np.uint8),(camera.width,camera.height),interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask[v,u]:
                    associated.append(mask)
        if len(associated)>1:
            raise GeometryUnavailable('Ambiguous destination pad association')
        if associated:
            views.append((camera,associated[0]))
    if len(views)<2:
        raise GeometryUnavailable('Destination pad requires multiple calibrated silhouettes')
    fit=fit_supported_outline(views,coefficients)
    normal=np.asarray(fit['rotation'])[:,2]
    if normal[2]<=1e-6:
        raise GeometryUnavailable('Pad surface cannot define a vertical placement height')
    top=np.asarray(fit['center'])+normal*float(fit['height'])/2
    z=float((normal@top-normal[:2]@xy)/normal[2])
    return {'surface_z':z,'height':float(fit['height']),'rms_px':float(fit['rms_px']),
            'views':len(views),'source':'current_calibrated_rgb_extrusion_on_observed_support'}
