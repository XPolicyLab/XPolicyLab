"""Local RGB pose checks after retreat; no scene truth or force sensing.

Track an already identified public digit near a predicted position. This does
not relax the complete-group ANS1 initial detector or invent invisible poses.
"""
import cv2
import numpy as np
from .number_perception import rotation


class PlacementFeedback:
    def __init__(self, perception):
        self.p = perception
        self.cache = {}

    @staticmethod
    def foreground(image, family):
        r,g,b=image.astype(np.float32).transpose(2,0,1)
        if family=='blue': raw=(b>55)&(b>1.35*r)&(b>1.20*g)
        elif family=='yellow': raw=(r>100)&(g>100)&(r>1.35*b)&(g>1.35*b)&(g>.80*r)
        elif family=='green': raw=(g>55)&(g>1.30*r)&(g>1.20*b)
        elif family=='purple': raw=(r>55)&(b>55)&(r>1.3*g)&(b>1.3*g)
        elif family=='neutral': raw=(np.minimum(np.minimum(r,g),b)>140)&(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)<42)
        else:
            lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB).astype(np.float32)
            raw=(lab[:,:,1]>np.median(lab[:,:,1])+16)&(r>80)
        return cv2.morphologyEx(np.uint8(raw)*255,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))

    def templates(self,digit):
        if digit not in self.cache:
            templates=[]
            for angle in np.deg2rad(np.arange(-96,97,3)):
                mask=np.zeros((96,96),np.uint8)
                for poly in sorted(self.p.digits[str(digit)]['polygons'],key=lambda p:p['hole']):
                    points=np.rint(np.asarray(poly['xy'])@rotation(angle).T*1000+48).astype(np.int32)
                    cv2.fillPoly(mask,[points],0 if poly['hole'] else 1)
                templates.append((float(angle),mask.astype(np.float32),float(mask.sum())))
            self.cache[digit]=templates
        return self.cache[digit]

    def locate(self,rgb,digit,near,family,height=.019,max_distance=.065):
        image=self.p.rectify(rgb,height)
        mask=self.foreground(image,family)
        count,labels,stats,centres=cv2.connectedComponentsWithStats(mask)
        matches=[]
        origin=np.asarray(self.p.bounds)[[0,2]]
        for k in range(1,count):
            x,y,w,h,area=stats[k]
            centroid=origin+centres[k]/1000
            if not (100<area<3800 and 14<w<95 and 14<h<95 and np.linalg.norm(centroid-near)<max_distance+.02):continue
            cx,cy=np.rint(centres[k]).astype(int);ox,oy=cx-70,cy-70
            crop=np.zeros((140,140),np.float32)
            xa,xb=max(0,ox),min(mask.shape[1],ox+140);ya,yb=max(0,oy),min(mask.shape[0],oy+140)
            crop[ya-oy:yb-oy,xa-ox:xb-ox]=labels[ya:yb,xa:xb]==k
            candidates=[]
            for angle,template,total in self.templates(digit):
                _,intersection,_,loc=cv2.minMaxLoc(cv2.matchTemplate(crop,template,cv2.TM_CCORR))
                score=intersection/max(area+total-intersection,1)
                xy=origin+np.asarray([ox+loc[0]+48,oy+loc[1]+48])/1000
                if np.linalg.norm(xy-near)<=max_distance:
                    candidates.append(dict(digit=digit,xy=xy.tolist(),yaw=angle,confidence=float(score),family=family))
            if not candidates:continue
            best=max(candidates,key=lambda item:item['confidence'])
            # Near-equivalent symmetric fits prefer the intended upright pose.
            ties=[v for v in candidates if v['confidence']>=best['confidence']-.008]
            best=min(ties,key=lambda item:abs(item['yaw']))
            # Reuse ANS1's shape IoU requirement, plus the local spatial window
            # and ambiguity margin; do not demand a stronger unexplained cutoff
            # than the initial recognition of the same visible object.
            if best['confidence']>=.58:matches.append(best)
        if not matches:return None
        matches.sort(key=lambda item:-item['confidence'])
        if len(matches)>1 and matches[0]['confidence']-matches[1]['confidence']<.06:return None
        return matches[0]

    @staticmethod
    def correct(pose,goal):
        return bool(pose is not None and np.linalg.norm(np.asarray(pose['xy'])-goal)<.012 and abs(pose['yaw'])<.16)
