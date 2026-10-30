
import cv2
import numpy as np

from ...lib.scene import shape as shape
from ...lib.geometry.transforms import TABLE_Z
from .bt_percep import occluders
from ...lib.perception.colour import blobs

SIDE = .035
PARTS = (shape.box([-.0175]*3,[.0175]*3),)
HUES = {"red":(172,6),"orange":(11,24),"yellow":(25,37),
        "green":(46,76),"cyan":(77,101),"blue":(108,135)}

# The spring button's cap sits inside the red hue window's wrap edge: the red
# range is hue<=6 and the button's own detector uses hue>=4, so their overlap is
# hue 4..6.  Which side a render lands on is not stable across environments -- a
# cap at hue 6.3 passes for a third red cube while the same cap at hue 7.2 does
# not -- so measure() cannot rely on the colour test alone to tell a cube from
# the button.  Exclude the button's footprint explicitly, the way occluders()
# already excludes the arms.
BUTTON_XY = (0., -.3)
BUTTON_HALF = (.04, .04)

def color_mask(rgb,color):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    low,high=HUES[color]
    hue=(h[:,:,0]>=low)&(h[:,:,0]<=high) if low<=high else ((h[:,:,0]>=low)|(h[:,:,0]<=high))


    saturation=55 if color=="cyan" else 115
    value=190 if color=="orange" else 100
    return hue&(h[:,:,1]>=saturation)&(h[:,:,2]>=value)

def measure(rgb,colors,qs,gs=None,*,support=TABLE_Z,near=None,limits=((-0.50,.50),(-.32,.08)),opened_colors=(),exclude_button=True):
    dc=occluders(qs,gs,dilate=2)
    if exclude_button:
        dc|=shape.region_mask(rgb.shape[:2],((BUTTON_XY[0]-BUTTON_HALF[0],BUTTON_XY[0]+BUTTON_HALF[0]),
                                             (BUTTON_XY[1]-BUTTON_HALF[1],BUTTON_XY[1]+BUTTON_HALF[1])),
                              support+.036)
    region=shape.region_mask(rgb.shape[:2],limits,support)
    out=[]
    for color in colors:
        mask=color_mask(rgb,color)
        if color in opened_colors:
            mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,
                                 np.ones((3,3),np.uint8)).astype(bool)
        mask=mask&region&~dc
        comps=blobs(mask.astype(np.uint8),min_area=40)
        for comp in comps[:4]:
            if comp["area"]>2200: continue
            selected=np.zeros(mask.shape,bool)
            selected[comp["pix"][:,1],comp["pix"][:,0]]=True
            p=shape.fit(selected,PARTS,support+SIDE/2,
                        np.linspace(0,np.pi/2,8,endpoint=False),ignore=dc,
                        period=np.pi/2,allowed=limits,search=.045)
            if p is None or p["residual"]>.55: continue
            if near is not None and np.linalg.norm(p["xy"]-near)>.075: continue
            out.append({**p,"color":color,"name":color,"height":SIDE,"z":support,
                        "polys":[_footprint(p["xy"],p["yaw"])]})
    return out

def _footprint(xy,yaw):
    from ..build_tower import rect
    return rect(xy,yaw,SIDE,SIDE)

def grasps(obj):


    return [{"yaw":obj["yaw"]+k*np.pi/2,"width":SIDE,"tip":tip}
            for tip in (.004,.009) for k in range(4)]
