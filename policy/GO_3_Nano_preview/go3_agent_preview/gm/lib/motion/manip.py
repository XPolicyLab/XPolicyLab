
from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.motion.plan import tcp_now
from ...lib.motion import plan as m
from ...lib.geometry.transforms import wrap
from ...lib.geometry.transforms import TABLE_Z, rot_z

FINGER = ((0.031, 0.0034, 0.010), (0.051, 0.0134, 0.016), (0.071, 0.040, 0.030))
BODY = (0.074, 0.087, 0.030)

SURVEY_Q = np.array([0., 0., .5, 0., 0., 0.])
TIP = kin.FINGER_TIP_X-kin.TCP_X

def gap_command(width, margin):
    return float(np.clip(((width+2*margin-.0008)/2-kin.GRIP_MIN)/
                         (kin.GRIP_MAX-kin.GRIP_MIN),0.,1.))

def expected_reading(width):
    return gap_command(width,0.)-.2

def parked(ctx):
    yield from m.move_joints(ctx,{a:SURVEY_Q for a in ("left","right")},max_dq=.09)
    yield from m.grip(ctx,("left","right"),1.,max_steps=5)

def obstacles_clear(xy,yaw,opening,tip_z,objects,exclude=(),margin=.002):
    for fp,z0,z1 in finger_rects(xy,yaw,opening,tip_z,grow=margin):
        for o in objects:
            if o.get("name") in exclude: continue
            if o["z"]>=z1 or o["z"]+o["height"]<=z0: continue
            if any(overlap(fp,p) for p in o["polys"]): return False
    return True

def _qualified(arm,qs,floor):

    if not m.joint_path_ok(arm,qs,floor): return False
    for q in qs[::3]+qs[-1:]:
        for name,c in kin.link_box_corners(arm,q,pad=.002).items():
            if name in ("link5","link6","link7","link8") and c[:,2].min()<floor:
                return False
    return True

def high_to(ctx,arm,T,*,floor=TABLE_Z+.012):
    q0=ctx.cmd_q[arm]
    for height in (max(T[2,3],.90),max(T[2,3],.86),T[2,3]):
        high=T.copy();high[2,3]=height
        try:
            qh=m.solve(arm,high,q0)
            direct=m.plan_joint(q0,qh,.09)
            if not _qualified(arm,direct,floor):continue
            descent=m.plan_line(arm,qh,T,step_m=.010)
        except m.PlanError:continue
        yield from m.follow(ctx,{arm:direct},max_settle=8)
        yield from m.follow(ctx,{arm:descent},max_settle=8)
        return
    raise m.PlanError(f"{arm} no qualified high approach")

def _carry_line(arm,q_start,over,step_m=.012,max_dq=None):
    raw=m.plan_line(arm,q_start,over,step_m=step_m)
    if max_dq is None:return raw
    limit=float(max_dq)
    if not 0.<limit<=.10:raise m.PlanError("carry joint step must be in (0,.10]")
    path=[];previous=q_start
    for q in raw:
        path.extend(m.plan_joint(previous,q,limit))
        previous=q
    return path

def choose_grasp(ctx,arm,obj,grasps,destination,z_target,yaw_target=None,objects=()):
    xy=np.asarray(obj["xy"])
    for g in grasps:
        jaw=float(g["yaw"])
        width=float(g["width"]);tip=float(g["tip"])
        carry_step=float(g.get("carry_step_m",.012))
        carry_max_dq=g.get("carry_max_dq")
        if not 0.<carry_step<=.012:continue
        offset=np.asarray(g.get("offset",[0.,0.]))
        source=xy+rot_z(obj["yaw"])[:2,:2]@offset
        opening=gap_command(width,.008)
        if not obstacles_clear(source,jaw,opening,obj["z"]+tip,objects,(obj["name"],)):
            continue
        delta=0. if yaw_target is None else float(wrap(yaw_target-obj["yaw"]))
        yplace=jaw+delta
        dest=np.asarray(destination)+rot_z(obj["yaw"]+delta)[:2,:2]@offset
        z=obj["z"]+tip+TIP
        zp=z_target+tip+TIP+.0015
        p=m.down_pose(np.r_[source,z],jaw)
        above=p.copy();above[2,3]=obj["z"]+obj["height"]+TIP+.018


        lift=p.copy();lift[2,3]=max(z+.025,obj["z"]+obj["height"]+TIP+.018)
        over=m.down_pose(np.r_[dest,max(lift[2,3],z_target+obj["height"]+TIP+.025)],yplace)
        place=m.down_pose(np.r_[dest,zp],yplace)
        hover=place.copy();hover[2,3]+=.015
        try:
            qa=m.solve(arm,above,ctx.cmd_q[arm])
            qg=m.solve(arm,p,qa)
            ql=m.solve(arm,lift,qg)
            qt=m.solve(arm,over,ql)
            qp=m.solve(arm,place,qt)
            if any(abs(q[5])>3.02 for q in (qa,qg,ql,qt,qp)): continue
            m.plan_line(arm,qg,lift,step_m=.01)
            planner=g.get("carry_planner",_carry_line)
            carry=planner(arm,ql,over,carry_step,carry_max_dq)
            down=m.plan_line(arm,qt,hover,step_m=.010)
            final=m.plan_line(arm,down[-1],place,step_m=.003)
            if not obstacles_clear(dest,yplace,opening,z_target+tip,objects,
                                   (obj["name"],)):
                continue
        except m.PlanError:
            continue
        plan={"arm":arm,"name":obj["name"],"opening":opening,"width":width,
                "pick":p,"above":above,"lift":lift,"over":over,"hover":hover,
                "place":place,"carry":carry,"down":down,"final":final,
                "carry_step_m":carry_step,
                "carry_max_dq":carry_max_dq,
                "delta":delta,"dest":np.asarray(destination),"z_target":z_target}
        if "carry_planner" in g:


            plan["carry_planner"]=g["carry_planner"]
            plan["carry_start"]=ql.copy()
        return plan
    return None

def transfer(ctx,obj,plan,objects=()):
    arm=plan["arm"]
    claims=ctx.memory.setdefault("workspace_claims",{})
    if claims and arm not in claims:
        raise m.PlanError("transfer requested before releasing another workspace claim")
    claims[arm]={"source":obj["xy"],"destination":plan["dest"],"radius":.10}
    ctx.cmd_g[arm]=plan["opening"]

    if plan.get("approach_path") is not None:
        yield from m.follow(ctx,{arm:plan["approach_path"]},max_settle=8)
    else:
        yield from high_to(ctx,arm,plan["above"])
    yield from m.move(ctx,{arm:plan["pick"]},step_m=.005,max_settle=10)
    yield from settle_tcp(ctx,arm,max_steps=8)
    yield from m.grip(ctx,arm,0.,min_steps=5,max_steps=12)
    g=ctx.g(arm);expected=expected_reading(plan["width"])
    # The gripper reading tracks the first finger only.  With num_envs=10
    # the two fingers can settle asymmetrically on a cube they do hold, and
    # that finger then reads ~0.09 wide of the aperture.  A miss closes past
    # the cube and reads low, so the closing side keeps its tight bound.
    ok=g>.025 and -.09<g-expected<.16
    ctx.log("grasp",obj=obj["name"],arm=arm,g=g,expect=expected,ok=ok,
            at=plan["pick"][:3,3])
    ctx.snap("grasp",cams=("cam_head",f"cam_{arm}_wrist"))
    if not ok:
        yield from m.grip(ctx,arm,plan["opening"],max_steps=6)
        yield from m.move(ctx,{arm:plan["above"]},step_m=.006)
        claims.pop(arm,None)
        return False
    test=plan["pick"].copy();test[2,3]+=.025
    yield from m.move(ctx,{arm:test},step_m=.006)
    g2=ctx.g(arm)
    ctx.log("test_lift",obj=obj["name"],arm=arm,g_before=g,g_after=g2,
            ok=g2>.025 and abs(g2-g)<.06)
    ctx.snap("test_lift",cams=("cam_head",f"cam_{arm}_wrist"))

    visual_ok=plan["held_check"](ctx,obj,plan) if "held_check" in plan else True
    if g2<=.025 or abs(g2-g)>=.06 or not visual_ok:
        yield from m.move(ctx,{arm:plan["pick"]},step_m=.006)
        yield from m.grip(ctx,arm,plan["opening"],max_steps=6)
        yield from m.move(ctx,{arm:plan["above"]},step_m=.006)
        claims.pop(arm,None)
        return False
    yield from m.move(ctx,{arm:plan["lift"]},step_m=.010)

    if "carry_planner" in plan:
        path=plan["carry"]
        if (not path or np.max(np.abs(ctx.cmd_q[arm]-plan["carry_start"]))>.01
                or np.max(np.abs(path[0]-ctx.cmd_q[arm]))>.10000001):
            raise m.PlanError(f"{arm} custom carry start differs from qualified path")
        yield from m.follow(ctx,{arm:path})
    elif plan.get("carry_max_dq") is None:
        yield from m.move(ctx,{arm:plan["over"]},step_m=plan.get("carry_step_m",.012))
    else:
        carry=_carry_line(arm,ctx.cmd_q[arm],plan["over"],
                          plan.get("carry_step_m",.012),plan["carry_max_dq"])
        yield from m.follow(ctx,{arm:carry})
    if ctx.g(arm)<.025: raise m.PlanError(f"{arm} hold lost carrying {obj['name']}")
    yield from m.move(ctx,{arm:plan["hover"]},step_m=.010)
    yield from settle_tcp(ctx,arm,max_steps=10)
    yield from m.move(ctx,{arm:plan["place"]},step_m=.003)
    yield from settle_tcp(ctx,arm,max_steps=10)
    yield from m.grip(ctx,arm,1.,max_steps=8,min_steps=4)
    ctx.log("placed",obj=obj["name"],arm=arm,xy=plan["dest"],z=plan["z_target"])
    back=plan["place"].copy();back[2,3]+=.060
    retreat=plan.get("retreat_path")
    if retreat is not None and np.max(np.abs(ctx.cmd_q[arm]-plan["retreat_start"]))<.01:
        yield from m.follow(ctx,{arm:retreat})
    else:
        yield from m.move(ctx,{arm:back},step_m=.006)
    obj["xy"]=plan["dest"].copy();obj["z"]=plan["z_target"]
    obj["yaw"]+=plan["delta"]

    if "polys" in obj:

        if obj["height"]==.035:
            obj["polys"]=[rect(obj["xy"],obj["yaw"],.035,.035)]
    claims.pop(arm,None)
    return True

def park_arm(ctx,arm):
    if np.max(np.abs(ctx.cmd_q[arm]-SURVEY_Q))<.02:return
    T=m.tcp(arm,ctx.cmd_q[arm])
    up=T.copy();up[2,3]+=.012
    try:
        qs=m.plan_line(arm,ctx.cmd_q[arm],up,step_m=.006)
    except m.PlanError:
        qs=[]
    if qs:yield from m.follow(ctx,{arm:qs},max_settle=5)
    path=m.plan_joint(ctx.cmd_q[arm],SURVEY_Q,.09)
    if not _qualified(arm,path,TABLE_Z+.025):
        raise m.PlanError(f"{arm} no qualified parking path")
    yield from m.follow(ctx,{arm:path},max_settle=6)
    ctx.memory.setdefault("workspace_claims",{}).pop(arm,None)

def finish(ctx):
    for arm in ("left","right"):
        ctx.cmd_g[arm]=1.
        if np.max(np.abs(ctx.cmd_q[arm]-SURVEY_Q))<.02: continue
        try:
            yield from park_arm(ctx,arm)
        except m.PlanError as exc:
            ctx.log("retreat_error",arm=arm,error=str(exc))
    yield from m.home(ctx,max_dq=.09)
    yield from m.grip(ctx,("left","right"),1.,max_steps=8,min_steps=4)
    yield from m.hold(ctx,5)


def rect(c, yaw, lx, ly):
    R = rot_z(yaw)[:2, :2]
    return np.array([np.asarray(c, float)[:2] + R @ np.array([sx * lx / 2, sy * ly / 2])
                     for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def overlap(A, B):
    for Pp in (A, B):
        for i in range(len(Pp)):
            e = Pp[(i + 1) % len(Pp)] - Pp[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False
    return True


def finger_rects(c, yaw, u, tip_z, grow=0.0):
    half = kin.unit_to_width(u) / 2
    a = np.array([np.cos(yaw), np.sin(yaw)])
    out, z_lo = [], tip_z
    for z_top, thick, hw in FINGER:
        for s in (-1.0, 1.0):
            cc = np.asarray(c, float)[:2] + s * a * (half + thick / 2)
            out.append((rect(cc, yaw, thick + 2 * grow, 2 * hw + 2 * grow), z_lo, tip_z + z_top))
        z_lo = tip_z + z_top
    z_b, hl, hw = BODY
    out.append((rect(c, yaw, 2 * hl + 2 * grow, 2 * hw + 2 * grow), tip_z + z_b, tip_z + 0.16))
    return out


def settle_tcp(ctx, arm, tol=0.0015, still=0.0004, max_steps=15):
    prev = m.tcp(arm, ctx.q(arm))[:3, 3]
    target = tcp_now(ctx, arm)[:3, 3]
    # A TCP that has stopped moving short of the target is resting on a contact
    # equilibrium (with num_envs=10 an off-centre grip holds it 1.8-2.5 mm out),
    # and waiting cannot close the gap.  The timeout would carry on from the same
    # state anyway; leaving after three still steps only saves the wait, which
    # matters in a task with little slack in its step budget.
    still_n = 0
    for i in range(max_steps):
        yield
        cur = m.tcp(arm, ctx.q(arm))[:3, 3]
        if np.linalg.norm(cur - target) < tol and np.linalg.norm(cur - prev) < still:
            return i + 1
        still_n = still_n + 1 if np.linalg.norm(cur - prev) < still else 0
        if still_n >= 3 and np.linalg.norm(cur - target) < .004:
            ctx.log("settle_still", arm=arm, err_mm=(cur - target) * 1000)
            return i + 1
        prev = cur
    ctx.log("settle_timeout", arm=arm, err_mm=(m.tcp(arm, ctx.q(arm))[:3, 3] - target) * 1000)
    return max_steps
