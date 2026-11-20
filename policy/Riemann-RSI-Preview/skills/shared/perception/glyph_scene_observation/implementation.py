"""Original RGB pad localization, descending digit read and independent reacquisition."""
from __future__ import annotations
import cv2
import json
import numpy as np
from skills.shared.observability.public_action_observation.implementation import plain
from skills.shared.geometry.x5_public_frame_model.implementation import project
from skills.shared.perception.material_glyph_outline.implementation import DigitTools, RecognitionNeedsObservation, components

def current_identity_support(obj):
    """Identity evidence is independent of the camera selected for geometry."""
    digit=obj['detection']['digit']
    views=[view for view in obj['views'] if view['winner']==digit
           and (view['reliable'] or view.get('confirmed_by_current_view_agreement'))
           and not view.get('identity_judgment_superseded_by_current_near_normal_view')]
    return any((view['similarity']>=.925 and view['margin']>=.025) or
               (view['similarity']>=.90 and view['margin']>=.06) or
               (view['view_angle_deg']<=12. and view['similarity']>=.93
                and view['topology']['resolves_close_candidates']) for view in views)


class Reader:
    def __init__(self,bank,material_profile='fixed',rig=None):
        self.bank=bank
        self.tools=DigitTools(bank,rig=rig,material_profile=material_profile)
        self.last_report=None
        self.pad_memory=None

    def pad_ring_centers(self,image,frame,expected_count=None):
        # Fixed public pad spacing and radius are asset constants. Find their
        # row from current RGB annulus contrast when a pale random tabletop
        # merges with the pads in a simple color component.
        angle=np.linspace(0,2*np.pi,72,endpoint=False)
        unit=np.c_[np.cos(angle),np.sin(angle)]
        intensity=image.astype(float).mean(axis=2)
        pale=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,1]<65
        counts=(expected_count,) if expected_count in (4,5) else (4,5)
        fits=[]
        for count in counts:
            for translation in np.arange(-.012,.0121,.002):
                for y in np.arange(-.12,-.024,.001):
                    centers=np.c_[(np.arange(count)-(count-1)/2)*.085+translation,np.full(count,y)]
                    samples=[];colors=[]
                    for radius in (.028,.032,.036,.046,.049):
                        xy=(centers[:,None]+unit*radius).reshape(-1,2)
                        uv=np.rint(project(np.c_[xy,np.full(len(xy),frame.mounting_z+.005)])).astype(int)
                        samples.append(intensity[uv[:,1],uv[:,0]].reshape(count,-1))
                        colors.append(pale[uv[:,1],uv[:,0]].reshape(count,-1))
                    inside=np.concatenate(samples[:3],axis=1)
                    outside=np.concatenate(samples[3:],axis=1)
                    contrast=np.median(inside,axis=1)-np.median(outside,axis=1)
                    if np.mean(np.concatenate(colors[:3],axis=1))<.70:continue
                    polarity=1. if np.median(contrast)>0 else -1.
                    contrast=contrast*polarity
                    score=float(np.median(contrast)+.4*np.min(contrast))
                    fits.append((score,count,centers,contrast))
        if not fits:return self.pad_edge_centers(image,frame,expected_count)
        score,count,centers,contrast=max(fits,key=lambda row:row[0])
        if score<10. or np.min(contrast)<1.5:
            return self.pad_edge_centers(image,frame,expected_count)
        return [dict(index=index,xy=center,annulus_contrast=float(contrast[index]),
                     source='current_RGB_annulus_contrast_fixed_public_pad_geometry')
                for index,center in enumerate(centers)]

    def pad_edge_centers(self,image,frame,expected_count=None):
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        edge=cv2.dilate(cv2.Canny(gray,20,45),np.ones((3,3),np.uint8))>0
        angle=np.linspace(0,2*np.pi,96,endpoint=False)
        unit=np.c_[np.cos(angle),np.sin(angle)]
        counts=(expected_count,) if expected_count in (4,5) else (4,5)
        fits=[]
        for count in counts:
            for dx in np.arange(-.010,.0101,.002):
                for y in np.arange(-.11,-.044,.001):
                    centers=np.c_[(np.arange(count)-(count-1)/2)*.085+dx,np.full(count,y)]
                    samples=[]
                    for radius in (.037,.039,.041):
                        xy=(centers[:,None]+unit*radius).reshape(-1,2)
                        uv=np.rint(project(np.c_[xy,np.full(len(xy),frame.mounting_z+.005)])).astype(int)
                        samples.append(edge[uv[:,1],uv[:,0]].reshape(count,-1))
                    support=np.mean(np.logical_or.reduce(samples),axis=1)
                    fits.append((float(np.median(support)+.5*np.min(support)),centers,support))
        score,centers,support=max(fits,key=lambda row:row[0])
        if score<1.10 or np.min(support)<.60:
            raise RuntimeError('Current RGB does not support the circular pad boundary')
        close=[row for row in fits if row[0]>=score-.015 and len(row[1])==len(centers)]
        centers=np.mean([row[1] for row in close],axis=0)
        return [dict(index=index,xy=center,boundary_support=float(support[index]),
                     source='current_RGB_circle_edges_fixed_public_pad_geometry')
                for index,center in enumerate(centers)]

    def pad_centers(self,image,frame,expected_count=None):
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        # Public cube-cushion appearance is pale cream and spatially separated from saturated digits.
        mask=((hsv[:,:,1]<60)&(hsv[:,:,2]>175)).astype(np.uint8)
        roi=np.zeros(mask.shape,np.uint8)
        corners=project(np.array([[-.26,-.16,frame.mounting_z+.005],[.26,-.16,frame.mounting_z+.005],
                                 [.26,.04,frame.mounting_z+.005],[-.26,.04,frame.mounting_z+.005]]))
        cv2.fillPoly(roi,[np.rint(corners).astype(np.int32)],1)
        mask*=roi
        detected=[]
        for item in components(mask,120):
            xy=frame.pixels(item['pixels'],.005)[:,:2]
            span=np.ptp(xy,axis=0)
            if .050<span[0]<.095 and .035<span[1]<.10:
                rect=cv2.minAreaRect(xy.astype(np.float32))
                center=np.asarray(rect[0],float)
                detected.append(dict(xy=center,area=item['area'],bbox=item['bbox']))
        detected.sort(key=lambda item:item['xy'][0])
        if len(detected)<2:
            return self.pad_ring_centers(image,frame,expected_count)
        xy=np.stack([item['xy'] for item in detected])
        counts=(expected_count,) if expected_count in (4,5) else (4,5)
        fits=[]
        for count in counts:
            nominal=(np.arange(count)-(count-1)/2)*.085
            indices=np.argmin(np.abs(xy[:,0,None]-nominal[None,:]),axis=1)
            offsets=xy[:,0]-nominal[indices]
            if len(set(indices))!=len(indices) or np.max(np.abs(offsets))>.014:continue
            if np.ptp(offsets)>.012 or np.ptp(xy[:,1])>.025:continue
            fits.append((float(np.mean(np.abs(offsets))),count,float(np.mean(offsets))))
        if not fits:return self.pad_ring_centers(image,frame,expected_count)
        _,count,translation=min(fits)
        y0=float(np.median(xy[:,1]))
        centers=[[(index-(count-1)/2)*.085+translation,y0] for index in range(count)]
        # A connected robot highlight can hide a pad boundary. Validate the
        # recovered circles against the current pale outer surface annulus.
        angle=np.linspace(0,2*np.pi,90,endpoint=False)
        ring=np.concatenate([np.c_[np.cos(angle),np.sin(angle)]*radius for radius in (.027,.033,.038)])
        for center in centers:
            uv=np.rint(project(np.c_[ring+center,np.full(len(ring),frame.mounting_z+.005)])).astype(int)
            support=float(np.mean(mask[uv[:,1],uv[:,0]]))
            if support<.35:return self.pad_ring_centers(image,frame,expected_count)
        return [dict(index=index,xy=center,source='current_RGB_circles_annulus_fixed_spacing')
                for index,center in enumerate(centers)]

    def read(self,obs,frame,*,family=None,pads=None,inspection_xy=None):
        report=self.tools.detect_digit_candidates(obs,frame,inspection_xy=inspection_xy)
        self.last_report=report
        provisional=[dict(digit=-index-1,xy=obj['anchor_xy'],height_m=.017,
                          origin_to_bottom_m=.0085,source='current_RGB_unconfirmed_obstacle')
                     for index,obj in enumerate(report['objects'])]
        try:
            # Pad count is independent RGB evidence. Conditioning it on the
            # glyph count can silently validate a scene with one missed digit.
            pads=self.pad_centers(obs.vision['cam_head'],frame) if pads is None else pads
        except RuntimeError as error:
            if self.pad_memory is None:
                report['status']='uncertain'
                report['reobservation']=self.tools.request_reobservation(report,str(error))
                raise RecognitionNeedsObservation(str(error),report,dict(digits=provisional,pads=[])) from error
            # Public pads are fixed geometry. Keep their independently
            # measured RGB positions as text when the looking arm hides the
            # row; no old image is used for current digit recognition.
            pads=self.pad_memory['pads']
            report['pad_count_source']='RGB_text_measurement_of_fixed_public_geometry'
            report['pad_measurement_observation_id']=self.pad_memory['observation_id']
        else:
            if self.pad_memory is not None and (len(pads)!=len(self.pad_memory['pads']) or
                    max(np.linalg.norm(np.asarray(a['xy'])-b['xy']) for a,b in zip(pads,self.pad_memory['pads']))>.015):
                reason='Current RGB pad count or position contradicts earlier independent measurement'
                report['status']='uncertain';report['reobservation']=self.tools.request_reobservation(report,reason)
                raise RecognitionNeedsObservation(reason,report,dict(digits=provisional,pads=pads))
            self.pad_memory=dict(pads=pads,observation_id=obs.observation_id,physical_step=obs.physical_step)
            report['pad_count_source']='independent_current_RGB_public_circle_geometry'
        if self.tools.material_profile=='random':
            pad_features=[]
            weak_features=[]
            retained=[]
            for obj in report['objects']:
                if max(view['similarity'] for view in obj['views'])<.90:
                    weak_features.append(obj)
                    continue
                # A texture patch at a measured circular pad center can fit
                # a partial glyph. Require a clear current silhouette before
                # counting it as an initial scene object. A real glyph there
                # remains an explicit inspection target, never discarded.
                near_center=any(np.linalg.norm(np.asarray(obj['anchor_xy'])-pad['xy'])<.018 for pad in pads)
                clear_shape=any(view['reliable'] and
                    ((view['view_angle_deg']<=12. and view['similarity']>=.925 and view['margin']>=.025) or
                     (view['camera']=='cam_head' and view['similarity']>=.96)) for view in obj['views'])
                if near_center and not clear_shape:
                    pad_features.append(obj)
                    continue
                retained.append(obj)
                if obj['status']!='confirmed':continue
                if not current_identity_support(obj):
                    obj['status']='uncertain'
                    obj['initial_identity_quality']='requires_clearer_current_RGB_before_grasp'
            report['objects']=retained
            report['pad_feature_candidates']=pad_features
            report['weak_feature_candidates']=weak_features
        digits=[obj['detection'] for obj in report['objects'] if obj['status']=='confirmed']
        reasons=[]
        if len(report['objects'])!=len(pads):
            reasons.append(f"RGB expected {len(pads)} digits; found {len(report['objects'])}")
        if len(digits)!=len(report['objects']):
            reasons.append('one or more digits need an independent clear observation')
        labels=[digit['digit'] for digit in digits]
        if len(set(labels))!=len(labels):
            reasons.append('independent recognitions contain duplicate identities; assignment must not invent a label')
        report['status']='ready' if not reasons else 'uncertain'
        report['pads']=pads
        if reasons:
            reason='; '.join(reasons)
            report['reobservation']=self.tools.request_reobservation(report,reason)
            raise RecognitionNeedsObservation(reason,report,dict(digits=provisional,pads=pads))
        digits.sort(key=lambda digit:digit['digit'],reverse=True)
        return dict(digits=digits,pads=pads,family=0,observation_id=obs.observation_id,
                    physical_step=obs.physical_step,recognition_status='independently_confirmed')

    def reacquire(self,obs,frame,digit,expected_xy,*,family=None,placed=False):
        try:
            result,report=self.tools.verify_digit_identity(obs,frame,digit,np.asarray(expected_xy),placed=placed)
        except RecognitionNeedsObservation as error:
            self.last_report=error.report
            raise
        self.last_report=report
        return result


class SceneObservation:
    def read(self,inspection_xy=None):
        self.stage='RGB_scene_read'
        try:
            scene=self.reader.read(self.obs,self.frame,inspection_xy=inspection_xy)
        except RecognitionNeedsObservation as error:
            self.event('digit_tools_report',report=error.report)
            # Unconfirmed component positions are obstacles for the existing
            # active-vision route, never identities used to arrange the row.
            self.scene=error.scene
            raise
        self.event('digit_tools_report',report=self.reader.last_report)
        self.event('perception_judgment',scene=scene)
        (self.directory/'detections.json').write_text(json.dumps(plain(scene),indent=2)+'\n')
        self.scene=scene


def read_scene(context):
    return context.read()
