
import cv2
import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.motion import plan as m
from ...lib.geometry.transforms import rot_z
from ...lib.geometry.camera import WRIST
from . import cube_rgb as C
from . import stack_blocks_by_language_see as S

def expected_mask(obj,plan):

    arm=plan["arm"]
    q=np.zeros(6)
    tcp_to_camera=np.linalg.inv(m.tcp(arm,q))@kin.wrist_camera_T(arm,q)
    camera=plan["pick"]@tcp_to_camera
    points=C.PARTS[0]@rot_z(obj["yaw"]).T+np.r_[obj["xy"],obj["z"]+C.SIDE/2]
    pixels=[WRIST.project(camera,p) for p in points]
    mask=np.zeros((480,640),np.uint8)
    if any(p is None for p in pixels):return mask.astype(bool)
    hull=cv2.convexHull(np.rint(pixels).astype(np.int32))
    cv2.fillConvexPoly(mask,hull,1)

    mask=cv2.erode(mask,np.ones((17,17),np.uint8))
    return mask.astype(bool)

def evidence(rgb,color,mask):
    area=int(mask.sum())
    observed=S.wrist_mask(rgb,color)
    coverage=float(np.count_nonzero(observed&mask)/max(1,area))
    return dict(ok=area>=100 and coverage>=.50,coverage=coverage,pixels=area)

def check(ctx,obj,plan):
    result=evidence(ctx.rgb(f"cam_{plan['arm']}_wrist"),obj["color"],
                    expected_mask(obj,plan))
    ctx.log("held_rgb",obj=obj["name"],arm=plan["arm"],**result)
    return result["ok"]
