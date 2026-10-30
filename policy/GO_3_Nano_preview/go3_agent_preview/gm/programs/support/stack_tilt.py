
import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.motion import manip as manip
from ...lib.motion import plan as m
from ...lib.geometry.transforms import make_T, rot_y, rot_z
from ...lib.geometry.transforms import TABLE_Z
from . import cube_rgb as C

def _box_hits_bounds(corners,lower,upper):
    if not (np.all(corners.max(axis=0)>lower) and
            np.all(corners.min(axis=0)<upper)):
        return False
    axes=corners[[4,2,1]]-corners[0]
    lengths=np.linalg.norm(axes,axis=1)
    if np.any(lengths<1e-9):return True
    axes=axes/lengths[:,None]
    projected=corners@axes.T
    centre=(lower+upper)*.5
    radius=np.abs(axes)@((upper-lower)*.5)
    middle=axes@centre
    gaps=np.maximum(projected.min(axis=0)-(middle+radius),
                    (middle-radius)-projected.max(axis=0))
    return not np.any(gaps>.002)

def choose(ctx,arm,obj,destination,z_target,objects):
    other="right" if arm=="left" else "left"
    other_boxes=list(kin.link_box_corners(other,ctx.cmd_q[other],pad=.001).values())
    blockers=[]
    for item in objects:
        footprint=np.concatenate(item["polys"])
        blockers.append((item["name"],np.r_[footprint.min(axis=0),item["z"]],
                         np.r_[footprint.max(axis=0),item["z"]+item["height"]]))
    cube_world=C.PARTS[0]@rot_z(obj["yaw"]).T+np.r_[obj["xy"],obj["z"]+C.SIDE/2]
    opening=manip.gap_command(C.SIDE,.008)

    for angle in (15.,-15.):
        for k in range(4):
            rotation=kin.down_rotation(obj["yaw"]+k*np.pi/2)@rot_y(np.deg2rad(angle))
            pick=make_T(rotation,np.r_[obj["xy"],obj["z"]+.012]-rotation[:,0]*manip.TIP)
            place=make_T(rotation,np.r_[destination,z_target+.0135]-rotation[:,0]*manip.TIP)
            above=pick.copy();above[2,3]+=.040
            lift=pick.copy();lift[2,3]+=.050
            over=place.copy();over[2,3]+=.050
            hover=place.copy();hover[2,3]+=.015
            back=place.copy();back[2,3]+=.060
            try:
                qa=m.solve(arm,above,ctx.cmd_q[arm])
                approach=m.plan_joint(ctx.cmd_q[arm],qa,.09)
                q=qa;paths=[]
                for pose,step in ((pick,.005),(lift,.006),(over,.012),(hover,.010),(place,.003),(back,.006)):
                    path=m.plan_line(arm,q,pose,step_m=step)
                    paths.append(path);q=path[-1]
            except m.PlanError:
                continue
            attached=(cube_world-pick[:3,3])@rotation

            def clear(path,contact=False,carried=False):
                for q in path:
                    boxes=kin.link_box_corners(arm,q,pad=.001)
                    if any(c[:,2].min()<TABLE_Z+.001 for n,c in boxes.items() if n in ("link5","link6","link7","link8")):
                        return False
                    for corners in boxes.values():
                        lo=corners.min(axis=0);hi=corners.max(axis=0)
                        for name,lower,upper in blockers:
                            if contact and name==obj["name"]:continue
                            if _box_hits_bounds(corners,lower,upper):return False
                        for other_corners in other_boxes:
                            if np.all(hi>other_corners.min(axis=0)) and np.all(lo<other_corners.max(axis=0)):return False
                    if carried:
                        tcp=m.tcp(arm,q);corners=attached@tcp[:3,:3].T+tcp[:3,3]
                        lo=corners.min(axis=0);hi=corners.max(axis=0)
                        if lo[2]<TABLE_Z-.0005:return False
                        for name,lower,upper in blockers:
                            if name!=obj["name"] and np.all(hi>lower) and np.all(lo<upper):return False
                return True

            if not clear(approach):continue
            if not clear(paths[0],contact=True):continue
            if not all(clear(p,contact=True,carried=True) for p in paths[1:5]):continue
            if not clear(paths[5],contact=True):continue
            return dict(arm=arm,name=obj["name"],opening=opening,width=C.SIDE,
                        pick=pick,above=above,lift=lift,over=over,hover=hover,place=place,
                        carry=paths[2],down=paths[3],final=paths[4],delta=0.,
                        dest=np.asarray(destination),z_target=z_target,approach_path=approach,
                        tilt_degrees=angle)
    return None
