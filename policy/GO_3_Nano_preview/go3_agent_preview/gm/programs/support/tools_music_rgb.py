
from pathlib import Path
import cv2
import numpy as np

from ...lib.scene import shape as shape
from ...lib.geometry.transforms import TABLE_Z
from ...lib.perception.colour import blobs, pixels_to_plane
from .bt_percep import occluders

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
MODELS = np.load(DATA_DIR / "tools_music_models.npz")
MALLET_Z = TABLE_Z + .013
KEY_PARTS = tuple(MODELS[f"key_{i}"] for i in range(8))
_a = np.linspace(0, 2*np.pi, 24, endpoint=False)
SPHERE = np.array([[.013*np.sin(t)*np.cos(a), .013*np.sin(t)*np.sin(a),
                    .013*np.cos(t)] for t in np.linspace(0,np.pi,13)
                   for a in _a])

def _component(mask, comp):
    out = np.zeros(mask.shape,bool)
    out[comp["pix"][:,1],comp["pix"][:,0]]=True
    return out

def xylophone_scene(rgb, qs, gs):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    occ=occluders(qs,gs,dilate=2)
    area=shape.region_mask(rgb.shape[:2],((-.34,.26),(-.15,.18)),TABLE_Z)
    paint=(hsv[:,:,0]>18)&(hsv[:,:,1]>90)&(hsv[:,:,2]>115)&area&~occ
    paint=cv2.morphologyEx(paint.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    xylo=shape.fit(paint,KEY_PARTS,TABLE_Z,np.pi+np.deg2rad(np.linspace(-20,20,7)),
                   search=.08,ignore=occ,maxiter=95)
    if xylo is None:
        return {}
    xylo["name"]="xylophone"

    verts=shape.box([-.215,-.125,0],[.213,.096,.028])
    instr=shape.render((verts,),(xylo["xy"][0],xylo["xy"][1],xylo["yaw"],TABLE_Z),
                       (0,0,rgb.shape[1],rgb.shape[0]))>.05
    red=(hsv[:,:,0]>=158)&(hsv[:,:,0]<=177)&(hsv[:,:,1]>115)&(hsv[:,:,2]>110)&~instr&~occ
    comps=blobs(red.astype(np.uint8),min_area=16)
    best=None
    for c in comps[:5]:
        if c["area"]>400: continue
        p=shape.fit(_component(red,c),(SPHERE,),MALLET_Z,[0.],ignore=occ,search=.03,maxiter=75)
        if p is not None and (best is None or p["residual"]<best["residual"]):best=p
    if best is None:return {"xylophone":xylo}

    white=(hsv[:,:,1]<75)&(hsv[:,:,2]>160)&~instr&~occ
    v,u=np.nonzero(white)
    points=pixels_to_plane(np.c_[u+.5,v+.5],MALLET_Z+.0035)[:,:2]
    points=points[(points[:,1]>-.315)&(points[:,1]<.12)&(points[:,0]>-.53)&(points[:,0]<.22)]
    pts=points-best["xy"]
    dist=np.linalg.norm(pts,axis=1)
    pts=pts[(dist>.026)&(dist<.207)&((dist<.065)|(dist>.125))]
    angles=np.linspace(-np.pi,np.pi,360,endpoint=False)
    scores=[]
    for a in angles:
        axis=np.array([np.cos(a),np.sin(a)])
        along=pts@axis;cross=pts@np.array([-axis[1],axis[0]])
        good=(along>.025)&(along<.205)&(np.abs(cross)<.0055)

        scores.append(float(np.sum(good)+np.sum(good&(along>.14))))
    if not len(pts) or max(scores)<9:return {"xylophone":xylo}
    a=angles[int(np.argmax(scores))]
    axis=np.array([np.cos(a),np.sin(a)])
    along=pts@axis;cross=pts@np.array([-axis[1],axis[0]])
    good=(along>.026)&(along<.205)&(np.abs(cross)<.006)
    selected=pts[good]

    if len(selected)>=8:
        cov=selected.T@selected
        eig,vec=np.linalg.eigh(cov);axis2=vec[:,-1]
        if axis2@axis<0:axis2=-axis2
        axis=axis2
    yaw=float(np.arctan2(axis[1],axis[0])-np.pi/2)
    return {"xylophone":xylo,"mallet":{**best,"yaw":yaw,"axis":axis,
            "name":"mallet","shaft_pixels":len(selected),"shaft_vote":max(scores)}}
