"""Small-tilt upright silhouette fallback, public static geometry + RGB only.
Existing residual/ambiguity thresholds remain mandatory. This is a hypothesis,
not object identity or grasp success. No scene IDs, depth or oracle state.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def fit_tilted_upright(camera,table_z,contour,models,seed_fits):
    if not seed_fits or seed_fits[0]["rms"]>4:return []
    angles=np.linspace(0,2*np.pi,32,endpoint=False)
    directions=np.c_[np.cos(angles),np.sin(angles)]
    support=(contour.reshape(-1,2).astype(float)@directions.T).max(0)
    by_asset={m["asset"]:m for m in models}
    seeds=[f for f in seed_fits if
           by_asset[f["asset"]]["height"]>.14 and max(by_asset[f["asset"]]["extent"][:2])<.09]
    output=[]
    for seed in seeds[:8]:
        model=by_asset[seed["asset"]];points=np.asarray(model["points"],float)
        def transformed(t):
            tilt=Rotation.from_euler("xy",t[3:]).as_matrix()
            q=points@tilt.T;floor=float(q[:,2].min());q[:,2]-=floor
            return q,tilt,floor
        def residual(t):
            q,_,_=transformed(t)
            yaw=Rotation.from_euler("z",t[2]).as_matrix()
            world=q@yaw.T;world[:,:2]+=t[:2];world[:,2]+=table_z
            return (camera.project(world)@directions.T).max(0)-support
        fit=least_squares(residual,np.r_[seed["xy"],seed["yaw"],0.,0.],
             bounds=([-.6,-.4,-2*np.pi,-.12,-.12],[.6,.25,2*np.pi,.12,.12]),
             max_nfev=60,diff_step=1e-4)
        rms=float(np.sqrt(np.mean(fit.fun**2)));maximum=float(abs(fit.fun).max())
        if rms>=1.5 or maximum>=4 or np.linalg.norm(fit.x[3:])>.12:continue
        q,tilt,floor=transformed(fit.x)
        nominal=np.r_[model["opening_xy"],model["grasp_height"]]
        center=tilt@nominal;center[2]-=floor
        if np.linalg.norm(center[:2]-nominal[:2])>.015:continue
        yaw=Rotation.from_euler("z",fit.x[2]).as_matrix()
        xy=fit.x[:2]+(yaw@center)[:2]
        if not (-.43<xy[0]<.5 and -.34<xy[1]<.12):continue
        extent=np.ptp(q,axis=0)
        output.append(dict(xy=xy,rms=rms,max_error=maximum,yaw=float(fit.x[2]),
            height=float(q[:,2].max()),asset=model["asset"],
            grasp_height=float(center[2]),extent=extent.tolist(),
            tilt=fit.x[3:].tolist(),geometry_source="public_model_small_tilt_rgb_fit"))
    return sorted(output,key=lambda f:f["rms"])
