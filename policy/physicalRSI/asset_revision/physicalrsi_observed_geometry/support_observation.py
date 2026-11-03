"""Visible support patches from current calibrated RGB and semantic masks."""
import cv2
import numpy as np
from . import Camera, GeometryUnavailable
from .rgb_support import initial_plane, refine_plane
from .dense_support import photometric_initial_plane
from .support_collision import support_patches


def observe_support(observation, segment_fn, *, occluders=('robot arm',)):
    return observe_support_geometry(observation, segment_fn, occluders=occluders)['patches']


def observe_support_masks(observation, segment_fn, *, occluders=('robot arm',)):
    names=['cam_head']+[name for name in observation['vision'] if name!='cam_head']
    cameras=[Camera.from_observation(observation,name) for name in names]
    if len(cameras)<2:
        raise GeometryUnavailable('Multiple calibrated RGB views required for support')
    masks=[]
    for camera in cameras:
        results=list(segment_fn(camera.image,[('table',.5),('wooden surface',.5)]+[(name,.3) for name in occluders]))
        def resized(value):
            mask=np.asarray(value,bool)
            if mask.shape!=camera.image.shape[:2]:
                mask=cv2.resize(mask.astype('uint8'),(camera.width,camera.height),interpolation=cv2.INTER_NEAREST).astype(bool)
            return mask
        # Material prompts also match objects resting on the support (for
        # example wooden coasters). Their confidence must not override an
        # explicit table detection, even when the table has a fabric covering.
        surfaces=[]
        for prompt in ('table','wooden surface'):
            surfaces=[(float(score),resized(mask)) for result in results
                      if result.prompt==prompt
                      for mask,score in zip(result.masks,result.scores) if score>=.5]
            if surfaces:
                break
        if not surfaces:
            raise GeometryUnavailable('Visible support surface required in each reference view')
        support=max(surfaces,key=lambda item:item[0])[1].copy()
        for result in results:
            if result.prompt in occluders:
                for mask,score in zip(result.masks,result.scores):
                    if score>=.3:
                        support &= ~cv2.dilate(resized(mask).astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
        if not support.any():
            raise GeometryUnavailable('Detected support is fully excluded by occluder masks')
        masks.append(support)
    return cameras,masks


def observe_support_geometry(observation, segment_fn, *, occluders=('robot arm',)):
    cameras,masks=observe_support_masks(observation,segment_fn,occluders=occluders)
    try:
        start,visible=initial_plane(cameras,masks,unique_head_pixels=True,return_support_mask=True)
    except GeometryUnavailable:
        plane,visible=photometric_initial_plane(cameras,masks)
    else:
        masks[0]=visible
        plane=refine_plane(cameras,masks,start,minimum_reference_views=1)
    return {'plane':plane, 'patches':support_patches(cameras[0],visible,plane)}
