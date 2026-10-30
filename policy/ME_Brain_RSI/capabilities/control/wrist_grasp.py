"""ANS4 wrist contact/temporal evidence; not digit pose or force inference.

Visible fingers define the contact band, not scene coordinates. Known material
comes from ANS1's initial RGB identity. Complete digit shape is unnecessary.
"""
import cv2
import numpy as np
from .placement_feedback import PlacementFeedback


def wrist_features(rgb, family):
    if rgb is None:
        return dict(available=False, contact=False, reason='missing_wrist')
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<64:
        return dict(available=False, contact=False, reason='invalid_wrist')
    h,w=image.shape[:2]
    material=PlacementFeedback.foreground(image.astype(np.uint8),family)>0
    # Scale the dark-finger threshold with image illumination; fixed 85 merged
    # close-range tabletop shadows into the fingers in dim RGB replay.
    intensity=image.max(axis=2)
    threshold=min(85.,.5*float(np.percentile(intensity,70)))
    dark=(intensity<threshold)&~material
    tips=[]
    for half in (0,1):
        mask=dark.copy();mask[:int(.2*h)]=False
        if half:mask[:,:w//2]=False
        else:mask[:,w//2:]=False
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        candidates=[]
        for k in range(1,n):
            if stats[k,4]<.008*h*w:continue
            ys,xs=np.where(labels==k)
            if ys.max()<.92*h:continue
            top=float(np.quantile(ys,.01));x=float(np.median(xs[ys<=top+2]))
            if .25*w<x<.75*w and .3*h<top<.8*h:
                candidates.append((int(stats[k,4]),x/w,top/h))
        if not candidates:return dict(available=False,contact=False,reason='fingers_not_resolved')
        _,x,y=max(candidates);tips.append([x,y])
    left,right=tips;mid=(left[0]+right[0])/2;top=max(left[1],right[1])
    band=np.zeros((h,w),bool)
    band[int(top*h):min(h,int((top+.23)*h)),max(0,int((left[0]-.025)*w)):min(w,int((right[0]+.025)*w))]=True
    n,labels,stats,centres=cv2.connectedComponentsWithStats(material.astype(np.uint8))
    items=[]
    for k in range(1,n):
        x,y,bw,bh,area=stats[k];cx,cy=centres[k]/[w,h]
        if area<.008*h*w or area>.35*h*w or abs(cx-mid)>.18:continue
        overlap=float(np.count_nonzero((labels==k)&band)/(h*w))
        contact=bool(overlap>.008 and (y+bh)/h>top+.06 and cy>top-.16)
        items.append(dict(area=float(area/(h*w)),centre=[float(cx),float(cy)],
            band_fraction=overlap,contact=contact,bottom=float((y+bh)/h)))
    items.sort(key=lambda x:(x['contact'],x['band_fraction'],x['area']),reverse=True)
    obj=items[0] if items else None
    return dict(available=True,contact=bool(obj and obj['contact']),object=obj,
        finger_tips=tips,jaw_gap=float(right[0]-left[0]),reason='wrist_material_geometry')


def decide(reference, observations, lift_metres, source_present):
    """Return held / empty / unknown. Absence of evidence never means held."""
    pair=observations[-2:]
    if len(pair)<2 or not all(x.get('available') for x in pair):return 'unknown'
    if lift_metres<.035:return 'unknown'
    if reference and reference.get('contact') and all(x['contact'] for x in pair):
        ref=reference['object'];objects=[x['object'] for x in pair]
        stable=all(.55<o['area']/ref['area']<1.8 and
                   np.linalg.norm(np.asarray(o['centre'])-ref['centre'])<.13 for o in objects)
        stable=stable and np.linalg.norm(np.asarray(objects[0]['centre'])-objects[1]['centre'])<.035
        if stable:return 'held'
    if source_present and all(not x['contact'] and x['jaw_gap']<.10 for x in pair):
        return 'empty'
    return 'unknown'
