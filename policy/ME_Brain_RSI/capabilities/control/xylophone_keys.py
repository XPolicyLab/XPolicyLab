"""RGB edge fit of eight public xylophone key rectangles; no scene poses.

Prototype: not connected to the policy until real-RGB ambiguity checks pass.
Coordinates are public Geometry/xylophone/00000 functional bounding boxes.
"""
import cv2
import numpy as np

KEY_BOXES=((.135,.161,-.068,.042),(.094,.117,-.063,.037),
           (.049,.074,-.058,.037),(.007,.034,-.053,.037),
           (-.036,-.011,-.048,.037),(-.079,-.053,-.043,.037),
           (-.121,-.096,-.038,.037),(-.164,-.138,-.033,.037))
KEY_HITS=np.array([[.147,-.020],[.104,-.017],[.062,-.016],[.019,-.015],
                   [-.022,-.014],[-.066,-.013],[-.109,-.011],[-.152,-.006]])

def template(angle,scale=600):
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    masks=[]
    for x0,x1,y0,y1 in KEY_BOXES:
        mask=np.zeros((260,260),np.float32)
        v=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])@r.T
        cv2.polylines(mask,[np.int32(v*scale+130)],True,1,1)
        masks.append(mask)
    return np.maximum.reduce(masks),masks,r

def fit_keys(image,valid,bounds,scale=600):
    """Return a supported edge hypothesis, not verified key contacts."""
    edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),40,100)
    if np.count_nonzero(edges)<100 or min(image.shape[:2])<80:return None
    dist=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    dist[valid==0]=20
    candidates=[]
    for deg in range(-180,180,2):
        mask,parts,r=template(np.deg2rad(deg),scale)
        # Crop zero padding, otherwise template size silently excludes keys
        # near the rectified workspace boundary even when all key edges fit.
        bx,by,bw,bh=cv2.boundingRect(np.uint8(mask))
        mask=mask[by:by+bh,bx:bx+bw]
        parts=[m[by:by+bh,bx:bx+bw] for m in parts]
        if bh>dist.shape[0] or bw>dist.shape[1]:continue
        scores=cv2.matchTemplate(dist,mask,cv2.TM_CCORR)/mask.sum()
        y,x=np.unravel_index(scores.argmin(),scores.shape)
        patch=dist[y:y+bh,x:x+bw]
        per_key=[float(patch[m>0].mean()) for m in parts]
        coverage=float(np.mean(patch[mask>0]<3))
        xy=np.array(bounds)[[0,2]]+(np.array([x,y])+130-np.array([bx,by]))/scale
        candidates.append((float(scores[y,x]),coverage,per_key,xy,deg,r))
    if not candidates:return None
    best=min(candidates,key=lambda c:c[0])
    score,coverage,per_key,xy,deg,r=best
    if score>2 or coverage<.8 or sum(v<2.5 for v in per_key)<6:return None
    # Reject unrelated similarly good solutions, including reversed order.
    alternatives=[c for c in candidates if np.linalg.norm(c[3]-xy)>.04
                  or abs((c[4]-deg+180)%360-180)>20]
    margin=min((c[0] for c in alternatives),default=20)-score
    if margin<.15:return None
    return dict(xy=xy,angle=float(np.deg2rad(deg)),key_xy=KEY_HITS@r.T+xy,
                edge_error=score,coverage=coverage,ambiguity_margin=margin)
