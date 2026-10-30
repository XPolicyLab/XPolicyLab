"""Additional RGB bottle proposals: join labels and coloured body regions.

Estimate a dominant chromatic tabletop hue inside the public table footprint.
This is a segmentation hypothesis, not semantic truth: red/table-coloured or
occluded bottles can still be missed. Existing silhouette/ambiguity gates must
validate every proposal. No scene layout, asset identity or reward is read.
Independent implementation; no fitted PhysicalRSI regression coefficients.
"""
import cv2
import numpy as np


def split_joined_contour(contour,shape):
    """Add marker-separated hypotheses; retain strict downstream shape fitting."""
    mask=np.zeros(shape,np.uint8)
    cv2.drawContours(mask,[contour],-1,255,-1)
    core=cv2.erode(mask,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(core)
    valid=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=100]
    if not 2<=len(valid)<=4:return []
    markers=np.zeros(shape,np.int32);markers[mask==0]=1
    for number,i in enumerate(valid,2):markers[labels==i]=number
    markers=cv2.watershed(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),markers)
    parts=[]
    for number in range(2,len(valid)+2):
        component=((markers==number)&(mask>0)).astype(np.uint8)*255
        for c in cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            _,_,w,h=cv2.boundingRect(c)
            if 150<cv2.contourArea(c)<10000 and 8<w<160 and 8<h<160:parts.append(c)
    return parts


def table_color_bottle_contours(rgb, camera, table_z):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv.transpose(2, 0, 1)
    # Public tabletop footprint, not any episode's object positions.
    uv = camera.project([[-.70, -.4, table_z], [.70, -.4, table_z],
                         [.70, .25, table_z], [-.70, .25, table_z]])
    if not np.isfinite(uv).all():
        return []
    region = np.zeros(h.shape, np.uint8)
    cv2.fillPoly(region, [np.rint(uv).astype(np.int32)], 255)
    region = cv2.erode(region, np.ones((11, 11), np.uint8))
    chromatic = h[(region > 0) & (s > 65) & (v > 60)]
    if len(chromatic) < 500:
        return []
    hist = np.bincount(chromatic // 6, minlength=30)
    if hist.max() / len(chromatic) < .35:
        return []
    mode = int(hist.argmax()) * 6 + 3
    distance = np.abs(h.astype(float) - mode)
    distance = np.minimum(distance, 180 - distance)
    raw = ((s < 65) & (v > 160)) | ((s > 65) & (v > 60) & (distance > 9))
    mask = cv2.morphologyEx(raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    result = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, width, height = cv2.boundingRect(contour)
        if 300<cv2.contourArea(contour)<16000:
            result.extend(split_joined_contour(contour,mask.shape))
        if 150 < cv2.contourArea(contour) < 10000 and 8 < width < 160 and 8 < height < 160:
            result.append(contour)
    return result
