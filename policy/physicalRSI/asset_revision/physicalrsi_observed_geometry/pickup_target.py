"""Select current RGB target pixels within a grounded detection, without object CAD."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import os
import numpy as np
from . import GeometryUnavailable
_LOCK=Lock()

@lru_cache(maxsize=1)
def _segmenter(path):
    from perception_services.perception.sam3_segment_text import Sam3TextSegmenter
    return Sam3TextSegmenter(checkpoint_path=Path(path),device=os.environ.get('PHYSICALRSI_PICKUP_SAM_DEVICE','cuda'))


def select_target_region(results, box, shape):
    box=np.asarray(box,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
        raise GeometryUnavailable('Valid grounded target box required')
    candidates=[]
    for result in results:
        for raw,score in zip(result.masks,result.scores):
            mask=np.asarray(raw,bool)
            if mask.shape!=tuple(shape):raise GeometryUnavailable('Target mask must match current RGB')
            if score<.3 or mask.sum()<8:continue
            yy,xx=np.nonzero(mask);bounds=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
            intersection=np.prod(np.maximum(0,np.minimum(bounds[2:],box[2:])-np.maximum(bounds[:2],box[:2])))
            union=np.prod(bounds[2:]-bounds[:2])+np.prod(box[2:]-box[:2])-intersection
            overlap=float(intersection/union)
            if overlap>=.3:candidates.append((overlap,float(score),mask))
    if not candidates:raise GeometryUnavailable('No segmented instance matches grounded target')
    candidates.sort(key=lambda row:row[:2],reverse=True)
    best=candidates[0]
    for overlap,score,mask in candidates[1:]:
        if overlap<best[0]-.1:continue
        iou=np.logical_and(mask,best[2]).sum()/np.logical_or(mask,best[2]).sum()
        if iou<.8:raise GeometryUnavailable('Ambiguous target instance masks')
    yy,xx=np.indices(shape)
    scoped=best[2] & (xx>=np.floor(box[0])) & (xx<=np.ceil(box[2])) & (yy>=np.floor(box[1])) & (yy<=np.ceil(box[3]))
    if scoped.sum()<8:raise GeometryUnavailable('Insufficient scoped target pixels')
    return scoped


def observe_pickup_target(image, description, box):
    path=os.environ.get('PHYSICALRSI_PICKUP_SAM_CHECKPOINT')
    if not path or not Path(path).is_file():raise GeometryUnavailable('Installed target segmentation checkpoint required')
    with _LOCK:
        results=_segmenter(path).segment_many(image,[(description,.3)])
    return select_target_region(results,box,image.shape[:2])


def observed_pickup_contact(detection, camera, *, preferred_pixel=None):
    """Retain a visual correspondence only when it belongs to today's target."""
    from .camera_projection import pixel_to_world_on_height
    mask=observe_pickup_target(camera.image,detection.description,detection.box_xyxy)
    def inside(pixel):
        uv=np.asarray(pixel,float)
        if uv.shape!=(2,) or not np.isfinite(uv).all():return False
        x,y=np.rint(uv).astype(int)
        return 0<=x<mask.shape[1] and 0<=y<mask.shape[0] and bool(mask[y,x])
    pixel=preferred_pixel if preferred_pixel is not None and inside(preferred_pixel) else detection.solid_xy
    if not inside(pixel):raise GeometryUnavailable('No observed contact pixel inside target mask')
    return pixel_to_world_on_height(pixel,world_z=detection.surface_z,camera=camera)
