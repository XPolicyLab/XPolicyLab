
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from ...lib.geometry.transforms import rot_z
from ...lib.geometry.camera import HeadCamera
from ...lib.perception.colour import pixels_to_plane

def box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)

def project(points, camera=None):
    if camera is None:
        c = (np.asarray(points) - HeadCamera.POS) @ HeadCamera.R
        fx, cx, cy = HeadCamera.FX, HeadCamera.CX, HeadCamera.CY
    else:
        c = (np.asarray(points) - camera[:3, 3]) @ camera[:3, :3]
        from ...lib.geometry.camera import WRIST
        fx, cx, cy = WRIST.FX, WRIST.CX, WRIST.CY
    return np.stack([cx + fx * c[:, 0] / -c[:, 2],
                     cy - fx * c[:, 1] / -c[:, 2]], axis=1) - 0.5

def render(parts, pose, roi, camera=None):
    u0, v0, w, h = roi
    out = np.zeros((h, w), np.uint8)
    R = rot_z(pose[2])
    for part in parts:
        pts = np.asarray(part) @ R.T + np.array([pose[0], pose[1], pose[3]])
        uv = project(pts, camera) - np.array([u0, v0])
        if not np.all(np.isfinite(uv)) or np.max(np.abs(uv)) > 1e5:
            continue
        poly = cv2.convexHull(np.rint(uv * 16).astype(np.int32))
        cv2.fillConvexPoly(out, poly, 255, lineType=cv2.LINE_AA, shift=4)
    return out.astype(float) / 255.

def fit(mask, parts, z, yaw_starts, *, xy=None, allowed=None, ignore=None,
        period=2*np.pi, search=.06, camera=None, maxiter=100):
    mask = np.asarray(mask, bool)
    ignore = np.zeros_like(mask) if ignore is None else np.asarray(ignore, bool)
    vs, us = np.nonzero(mask & ~ignore)
    if len(us) < 20:
        return None
    if xy is None:
        if camera is None:
            xy = pixels_to_plane(np.stack([us+.5, vs+.5], axis=1), z)[:, :2].mean(axis=0)
        else:
            from ...lib.geometry.camera import WRIST
            xy = np.mean([WRIST.to_plane(camera, u+.5, v+.5, z)[:2]
                          for u,v in zip(us[::max(1,len(us)//400)],vs[::max(1,len(vs)//400)])],axis=0)
    xy = np.asarray(xy,float)

    pad = 12
    u0,v0 = max(0,int(us.min())-pad), max(0,int(vs.min())-pad)
    u1,v1 = min(mask.shape[1],int(us.max())+pad+1), min(mask.shape[0],int(vs.max())+pad+1)
    roi = (u0,v0,u1-u0,v1-v0)
    target = cv2.GaussianBlur(mask[v0:v1,u0:u1].astype(float),(5,5),.65)
    weight = (~ignore[v0:v1,u0:u1]).astype(float)
    denom = max(1.,float((target*weight).sum()))

    def cost(x):
        p = xy + x[:2]*.01
        if np.linalg.norm(p-xy)>search:
            return 1e3+np.linalg.norm(p-xy)
        if allowed is not None:
            (a,b),(c,d)=allowed
            if p[0]<a or p[0]>b or p[1]<c or p[1]>d:
                return 1e3
        pred=render(parts,(p[0],p[1],x[2],z),roi,camera)
        pred=cv2.GaussianBlur(pred,(5,5),.65)
        return float(np.sum(weight*(target-pred)**2)/denom)

    ranked = sorted(((cost(np.array([0.,0.,y])),float(y)) for y in yaw_starts))
    best=None
    for _,yaw in ranked[:4]:
        start=np.array([0.,0.,yaw])
        simplex=np.array([start,start+np.array([.4,0,0]),
                          start+np.array([0,.4,0]),start+np.array([0,0,.08])])
        r=minimize(cost,start,method="Nelder-Mead",options={"maxiter":maxiter,
                   "xatol":1e-3,"fatol":1e-6,"initial_simplex":simplex})
        if best is None or r.fun<best[0]: best=(float(r.fun),r.x)
    residual,x=best
    return {"xy":xy+x[:2]*.01,"yaw":float((x[2]+period/2)%period-period/2),
            "z":float(z),"residual":residual,"visible":len(us),"roi":roi}

def region_mask(shape, limits, z):
    (x0,x1),(y0,y1)=limits
    uv=project(np.array([[x,y,z] for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]))
    out=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(out,np.rint(uv).astype(np.int32),1)
    return out.astype(bool)
