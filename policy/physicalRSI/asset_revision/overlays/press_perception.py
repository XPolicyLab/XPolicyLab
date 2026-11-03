"""Candidate RGB-only button/card localization with explicit sensor calibration."""
import time
import numpy as np
from .perception.rgb_grounding import RGBGrounding
from .perception.rgb_depth import RGBDepth
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.rgb_support import initial_plane, refine_plane
from physicalrsi_observed_geometry.dense_support import photometric_initial_plane, photometric_plane_from_proposals
from physicalrsi_observed_geometry.support_observation import observe_support_masks
from physicalrsi_observed_geometry.observed_depth_alignment import ObservedPlaneDepthAlignment
from .observed_cards import ObservedNumberCards

def observed_reference_support(camera, rows):
    tables=sorted([row for row in rows if row['query']=='table'],key=lambda row:-row['score'])
    if not tables or tables[0]['score']<.5 or (len(tables)>1 and tables[0]['score']-tables[1]['score']<.2):
        raise GeometryUnavailable('Reference table detection is ambiguous')
    mask=np.zeros(camera.image.shape[:2],bool)
    x0,y0,x1,y1=np.round(tables[0]['box_xyxy']).astype(int)
    mask[max(0,y0):y1+1,max(0,x0):x1+1]=True
    for row in rows:
        if row['query']=='table':continue
        candidate=np.asarray(row['box_xyxy'],float)
        table=np.asarray(tables[0]['box_xyxy'],float)
        overlap=np.maximum(0,np.minimum(candidate[2:],table[2:])-np.maximum(candidate[:2],table[:2])).prod()
        union=np.maximum(0,candidate[2:]-candidate[:2]).prod()+np.maximum(0,table[2:]-table[:2]).prod()-overlap
        if union>0 and overlap/union>.9:
            if tables[0]['score']-row['score']>.2:
                continue
            raise GeometryUnavailable('Conflicting semantic labels for reference support extent')
        x0,y0,x1,y1=np.round(row['box_xyxy']).astype(int)
        mask[max(0,y0-3):y1+4,max(0,x0-3):x1+4]=False
    if mask.sum()<8:
        raise GeometryUnavailable('No visible reference support remains after occlusion exclusion')
    return mask


def learned_depth_plane_proposals(camera, prediction, mask):
    prediction=np.asarray(prediction)
    mask=np.asarray(mask)
    if prediction.shape!=camera.image.shape[:2] or mask.shape!=prediction.shape or mask.dtype!=bool:
        raise GeometryUnavailable('Depth and visible support must match the current camera')
    yy,xx=np.nonzero(mask & np.isfinite(prediction) & (prediction>0))
    if len(xx)<20:
        raise GeometryUnavailable('Insufficient visible depth predictions for initialization')
    stride=max(1,len(xx)//1000);uv=np.c_[xx[::stride],yy[::stride]]
    values=prediction[yy[::stride],xx[::stride]]
    # Dimensionless initialization search; no proposal is usable without the
    # independent, calibrated, two-view/two-split photometric validation.
    return np.asarray([np.median(camera.unproject(uv,values*factor),axis=0)
                       for factor in np.geomspace(.1,10.,129)])


def measured_support(cameras, mask, detector, *, observation=None, depth_prediction=None):
    masks=[mask]+[np.ones(camera.image.shape[:2],bool) for camera in cameras[1:]]
    try:
        plane=refine_plane(cameras,masks,initial_plane(cameras,masks))
        return plane,mask,'sparse_rgb_and_photometric_refinement'
    except GeometryUnavailable:
        # Preserve successful sparse reconstruction; unresolved cases require
        # independently validated image tiles and observed reference visibility.
        try:
            masks=[mask]+[observed_reference_support(camera,detector.predict(camera.image,deadline=time.monotonic()+120))
                          for camera in cameras[1:]]
            plane,visible=photometric_initial_plane(cameras,masks,proposal_error_px=3.)
            return plane,visible,'heldout_photometric_with_observed_reference_boxes'
        except GeometryUnavailable:
            if observation is None or depth_prediction is None:
                raise
            from PourBallsRGB.sam.sam3_http import segment_many_http
            try:
                cameras,masks=observe_support_masks(observation,segment_many_http,
                    occluders=('robot arm','button','number card'))
            except (RuntimeError,OSError,TimeoutError) as error:
                raise GeometryUnavailable('Current support segmentation unavailable') from error
            try:
                plane,visible=photometric_initial_plane(cameras,masks,proposal_error_px=3.)
                return plane,visible,'heldout_photometric_with_current_semantic_masks'
            except GeometryUnavailable:
                proposals=learned_depth_plane_proposals(cameras[0],depth_prediction,masks[0])
                plane,visible=photometric_plane_from_proposals(cameras,masks,proposals,seed_spacing=.005)
                return plane,visible,'heldout_photometric_with_learned_depth_initialization'

class PressPerception:
    def __init__(self, configuration, classify_digit):
        self.detector = RGBGrounding(model_path=configuration['detector'], queries={'buttons':'button','cards':'number card','table':'table','robot':'robot arm'}, threshold=.3)
        self.depth = RGBDepth(model_path=configuration['depth'])
        self.cards = ObservedNumberCards(classify_digit)

    def observe(self, observation):
        camera = Camera.from_observation(observation)
        rgb = camera.image
        # No layout, object pose, simulator depth, task score or seed enters here.
        rows = self.detector.predict(rgb, deadline=time.monotonic()+120)
        digits = self.cards.read(rgb, rows)
        buttons = [r for r in rows if r['query']=='buttons']
        if len(buttons)!=3: raise GeometryUnavailable('Three visually resolved buttons required')
        red, blue = [], []
        for row in buttons:
            x0,y0,x1,y1=row['box_xyxy'];u,v=(x0+x1)/2,(y0+y1)/2
            dx,dy=(x1-x0)*.15,(y1-y0)*.15
            patch=rgb[int(v-dy):int(v+dy)+1,int(u-dx):int(u+dx)+1]
            color=np.median(patch.reshape(-1,3),axis=0)
            if color[0] > 1.3*color[2]:red.append(row)
            elif color[2] > 1.3*color[0]:blue.append(row)
            else:raise GeometryUnavailable('Button color role is ambiguous')
        if len(red)!=2 or len(blue)!=1:raise GeometryUnavailable('Two red count buttons and one blue confirmation button required')
        # Red count buttons follow card reading order; blue may be anywhere.
        buttons=sorted(red,key=lambda r:r['box_xyxy'][0])+blue
        tables = sorted([r for r in rows if r['query']=='table'], key=lambda r:-r['score'])
        if not tables or tables[0]['score']<.5 or (len(tables)>1 and tables[0]['score']-tables[1]['score']<.2):
            raise GeometryUnavailable('Table detection is ambiguous')
        raw = self.depth.predict(rgb, deadline=time.monotonic()+120)
        mask = np.zeros(raw.shape, dtype=bool)
        x0,y0,x1,y1=np.round(tables[0]['box_xyxy']).astype(int);mask[y0:y1+1,x0:x1+1]=True
        for r in rows:
            if r['query']=='table':continue
            x0,y0,x1,y1=np.round(r['box_xyxy']).astype(int)
            mask[max(0,y0-3):y1+4,max(0,x0-3):x1+4]=False
        cameras = [camera] + [Camera.from_observation(observation, name)
                              for name in observation['vision'] if name != 'cam_head']
        plane, mask, support_method = measured_support(cameras, mask, self.detector,
            observation=observation,depth_prediction=raw)
        alignment = ObservedPlaneDepthAlignment(camera=camera, plane=plane, target='axial')
        try:
            depth, fit = alignment.align(raw, mask, deadline=time.monotonic()+30)
        except ValueError as error:
            raise GeometryUnavailable(str(error)) from error
        fit.update(source='current_calibrated_rgb_support_plane', camera_source='current_observation', support_method=support_method,
                   metric_accuracy_certified=False)
        positions=[]
        for r in buttons:
            x0,y0,x1,y1=r['box_xyxy'];u,v=(x0+x1)/2,(y0+y1)/2;dx,dy=(x1-x0)*.15,(y1-y0)*.15
            yy,xx=np.mgrid[int(v-dy):int(v+dy)+1,int(u-dx):int(u+dx)+1]
            values=depth[yy,xx]
            if np.ptp(values)>.015:raise GeometryUnavailable('Button surface depth is not sufficiently consistent')
            positions.append(camera.unproject(np.array([[u,v]]),np.array([np.median(values)]))[0].tolist())
        return {'digits':digits,'positions':positions,'alignment':fit,'detections':rows}

    def close(self):
        self.detector.close();self.depth.close()
