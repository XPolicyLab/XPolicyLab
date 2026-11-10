"""RGB silhouette refinement using a public extruded T mesh, not scene truth."""
import cv2
import numpy as np
from itertools import product


def silhouette(camera, footprint, xy, yaw, table_z, height, shape):
    points=np.asarray(footprint,float).reshape(-1,2)/1000
    c,s=np.cos(yaw),np.sin(yaw)
    points=points@np.array([[c,s],[-s,c]])+xy
    bottom=camera.project(np.c_[points,np.full(len(points),table_z)])
    top=camera.project(np.c_[points,np.full(len(points),table_z+height)])
    result=np.zeros(shape,np.uint8)
    faces=[bottom,top]
    faces.extend(np.array([bottom[i],bottom[(i+1)%len(points)],
                           top[(i+1)%len(points)],top[i]]) for i in range(len(points)))
    for face in faces:
        cv2.fillPoly(result,[np.rint(face).astype(np.int32)],1)
    return result


def refine_source(rgb,camera,footprint,xy,yaw,table_z):
    """Local bounded RGB-only search; includes projected side faces.

    It does not fit any coefficient to simulator diagnostics. A low silhouette
    agreement rejects the proposed update. Search starts from the 2D fit.
    """
    values=rgb.astype(float)
    red,green,blue=values.transpose(2,0,1)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    observed=(red>130)&(red-green>65)&(red-blue>45)
    observed&=((hsv[:,:,0]<=3)|(hsv[:,:,0]>=168))&(hsv[:,:,1]>100)
    initial=np.r_[xy,yaw].astype(float)
    initial_mask=silhouette(camera,footprint,initial[:2],initial[2],table_z,.015,observed.shape)
    region=cv2.dilate(initial_mask,np.ones((41,41),np.uint8))>0
    observed &= region
    def score(p,top_only=False):
        if np.linalg.norm(p[:2]-initial[:2])>.03 or abs(p[2]-initial[2])>.21:
            return -1.
        predicted=silhouette(camera,footprint,p[:2],p[2],
                             table_z+.015 if top_only else table_z,
                             0 if top_only else .015,observed.shape)>0
        return float(np.count_nonzero(predicted&observed)/max(1,np.count_nonzero(predicted|observed)))
    fits=[]
    # Dark/occluded side faces need not be red. Compare both physically
    # meaningful visibility hypotheses using RGB, not a per-scene offset.
    for top_only in (False,True):
        best=initial.copy();quality=score(best,top_only)
        initial_quality=quality
        for linear,angular in ((.005,np.deg2rad(3)),(.002,np.deg2rad(1)),
                               (.001,np.deg2rad(.5)),(.0005,np.deg2rad(.25))):
            for _ in range(12):
                candidates=[]
                for delta in product((-1,0,1),repeat=3):
                    p=best+np.asarray(delta)*[linear,linear,angular]
                    candidates.append((score(p,top_only),p))
                candidate=max(candidates,key=lambda item:item[0])
                if candidate[0]<=quality+1e-6:break
                quality,best=candidate
        fits.append({'xy':best[:2],'angle':float(best[2]),'iou':quality,
                     'initial_iou':initial_quality,'top_only':top_only})
    return max(fits,key=lambda fit:fit['iou'])
