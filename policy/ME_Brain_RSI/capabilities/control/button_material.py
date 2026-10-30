"""Current wrist RGB red-orange cap evidence, separated from warm table wood."""
from __future__ import annotations
import cv2
import numpy as np

def observed_robot_occlusion(rgb,possible_robot):
    """Extend a geometry-supported robot component by its observed RGB only.

    The static wrist prior can miss a finger's image edge. A dark achromatic
    component must connect to the camera's lower border and substantially
    overlap the public finger envelope before its current RGB pixels censor
    an object boundary. No convex fill or object-colored pixel is added.
    """
    hsv=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2HSV)
    # HSV saturation is unstable near black: warm lighting can give a black
    # finger high nominal saturation despite tiny RGB values.
    dark=(hsv[:,:,2]<40)|((hsv[:,:,1]<85)&(hsv[:,:,2]<95))
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    result=possible_robot&((hsv[:,:,1]<65)&(hsv[:,:,2]<210))
    records=[];anchored=np.zeros(result.shape,bool)
    for i in range(1,n):
        m=labels==i;overlap=int(np.count_nonzero(m&possible_robot));bottom=bool(m[-1,:].any())
        if bottom and overlap>=1000:
            result|=m;anchored|=m;records.append(dict(pixels=int(stats[i,4]),projected_finger_overlap=overlap,bottom_border=True))
    # Warm illumination can turn an otherwise black finger high-saturation
    # red. It may grow an already anchored dark body, but cannot seed a body.
    # Growth stays inside the public robot projection and the existing95
    # dark-value envelope, rather than extending the cap search/shape mask.
    warm_envelope=possible_robot&(hsv[:,:,2]<95)
    growth_domain=anchored|warm_envelope
    _,growth_labels,_,_=cv2.connectedComponentsWithStats(growth_domain.astype(np.uint8))
    supported=set(growth_labels[anchored].tolist())-{0}
    grown=np.isin(growth_labels,list(supported))&warm_envelope
    added=grown&~result
    result|=grown
    return result,dict(method='current_dark_body_seed_then_geometry_bounded_connected_low_value_growth',components=records,pixels=int(result.sum()),warm_growth_pixels=int(added.sum()),warm_growth_value_limit=95,warm_growth_requires_projected_robot=True,warm_growth_requires_anchored_dark_body=True)

def cap_material_candidates(rgb,area):
    """Return observed connected material; never a CAD-filled object mask.

    Hue alone labels the reddish table. A public red/orange material also has
    strong normalized red-green contrast and saturation at usable brightness.
    Connected-region geometry and downstream real-CAD boundary fits are still
    required; this material test is not an object identity or press verdict.
    """
    hsv=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2HSV)
    value=rgb.astype(np.float32);r,g,b=value[:,:,0],value[:,:,1],value[:,:,2]
    contrast=(r-g)/(r+g+1.)
    hue=(hsv[:,:,0]<14)|(hsv[:,:,0]>170)
    strong=hue&(hsv[:,:,1]>145)&(hsv[:,:,2]>135)&(contrast>.30)&area
    # Shiny red plastic can mix with white reflection. A brighter, weaker
    # red envelope is allowed only when connected to an observed strong core;
    # dark/similarly-hued wood is not enough to seed or grow this material.
    weak=hue&(hsv[:,:,1]>80)&(hsv[:,:,2]>170)&(contrast>.18)&area
    weak|=strong
    n_weak,l_weak,_,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8))
    supported=set(l_weak[strong].tolist())-{0}
    selected=np.isin(l_weak,list(supported))
    # Closing only bridges <=2px specular interruptions; no hull is filled.
    selected=cv2.morphologyEx(selected.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)&area
    # Remove thin observed wood/specular bridges before component identity.
    # Opening is subtractive: it never fills a cap or invents a crop edge.
    # Remaining components still must not touch the original search boundary.
    raw_material_pixels=int(selected.sum())
    selected=cv2.morphologyEx(selected.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)).astype(bool)&selected
    n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8))
    candidates=[];rejected=[]
    search_boundary=(area.astype(np.uint8)-cv2.erode(area.astype(np.uint8),np.ones((3,3),np.uint8)))>0
    for i in range(1,n):
        count=int(stats[i,4])
        if count<80:continue
        mask=labels==i;yy,xx=np.where(mask)
        hull=cv2.convexHull(np.column_stack([xx,yy]).astype(np.float32))
        solidity=float(count/max(1.,cv2.contourArea(hull)))
        boundary=bool(np.any(mask[0,:]) or np.any(mask[-1,:]) or np.any(mask[:,0]) or np.any(mask[:,-1]))
        search_touch=bool(np.any(mask&search_boundary))
        record=dict(pixels=count,solidity=solidity,image_border_touch=boundary,search_border_touch=search_touch,median_red_green_contrast=float(np.median(contrast[mask])),median_saturation=float(np.median(hsv[:,:,1][mask])),median_value=float(np.median(hsv[:,:,2][mask])))
        if solidity<.65 or count>.95*int(area.sum()) or search_touch or boundary:rejected.append(record);continue
        candidates.append((mask,record))
    return candidates,dict(method='current_rgb_normalized_red_green_and_saturation_then_3px_opened_components',raw_material_pixels=raw_material_pixels,opened_material_pixels=int(selected.sum()),search_pixels=int(area.sum()),rejected=rejected)
