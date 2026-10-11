"""Explicit dual-arm commands. Independent cuRobo plans, slow-only synchronization.

This is NOT a coupled collision planner. Endpoint workspace guards are conservative
and do not certify intermediate cross-arm or tabletop-object collision freedom.
"""
import numpy as np


def merge_paths(paths, starts, keys):
    n = max(len(p) for p in paths.values())
    if not n or any(not p for p in paths.values()):
        raise ValueError("empty dual path")
    out = []
    for i in range(1, n + 1):
        action = dict(paths["left"][-1])
        for arm, path in paths.items():
            key = keys[arm]
            samples = np.asarray([starts[arm]] + [a[key] for a in path], dtype=float)
            t = i * len(path) / n
            lo = min(int(t), len(path)); hi = min(lo + 1, len(path))
            action[key] = (samples[lo] + (t-lo)*(samples[hi]-samples[lo])).tolist()
        out.append(action)
    return out


def execute(r, cmd, env, hooks, session, poses, grippers, settle, run_actions,
            report, offset, hold_steps, max_plan_steps, settle_steps):
    children = [r.commands.Command.from_dict(v) for v in cmd.value]
    targets = {c.arm: r.resolve(c, *poses[c.arm], session["home_poses"].get(c.arm), offset)
               for c in children}
    # No crossing arms. Not a path-wise collision certificate.
    for points in ({a:r.tcp_of(*p,offset) for a,p in poses.items()},
                   {a:t["tcp"] for a,t in targets.items()}):
        if points["left"][0] > -.08 or points["right"][0] < .08 or np.linalg.norm(points["left"]-points["right"]) < .30:
            return {"ok":False,"note":"sync endpoint guard: keep arms in separated left/right workspaces","motion_steps":0}, None
    paths = {}; infos = {}
    for c in children:
        t=targets[c.arm]; p,q=poses[c.arm]
        request={"arm":c.arm,"target_pose":[float(v) for v in (*t["pos"],*t["quat"])],
                 "start_pose":[float(v) for v in (*p,*q)],"grippers":dict(grippers),
                 "executed_motion":True,"settle_steps":hold_steps,"duration_s":None,"max_plan_steps":max_plan_steps}
        paths[c.arm], infos[c.arm] = hooks.plan_keypose(env,request)
        if infos[c.arm].get("plan_status") != "Success" or not paths[c.arm] or infos[c.arm].get("truncated"):
            return {"ok":False,"note":f"sync {c.arm} planning failed; neither arm executed", "plans":infos,"motion_steps":0},None
    starts=r._read_joints(env,hooks)
    keys={a:env.robot_manager.process_name(bot.arm_name) for a,bot in r.target_arms(env)}
    actions=merge_paths(paths,starts,keys)
    is_push = any(c.kind == "push3d" for c in children)
    trace=[]; path_abort=None
    if is_push:
        if not all(c.kind == "push3d" and abs(c.value[2]) <= .01 for c in children):
            return {"ok":False,"note":"push requires two planar push3d commands","motion_steps":0},None
        try:
            actions, waypoints = r._sibling("cartesian_push").build(r,env,hooks,poses,targets,grippers,len(actions))
        except Exception as exc:
            return {"ok":False,"note":str(exc),"motion_steps":0},None
        executed=0;stop=None
        for action in actions:
            n,stop=run_actions([action],settle);executed+=n
            live=r.live_poses(env)
            trace.append({a:r.tcp_of(*p,offset).tolist() for a,p in live.items()})
            if stop:break
            if len(trace)>3 and any(abs(trace[-1][a][2]-r.tcp_of(*poses[a],offset)[2])>.012 for a in targets):
                path_abort="actual TCP left the 1.2cm height corridor; stopped push"
                break
    else:
        executed,stop=run_actions(actions,settle)
    extra=0; stable=0; previous=r.live_poses(env)
    for _ in range(settle_steps):
        if stop or path_abort: break
        count,stop=run_actions([actions[-1]],settle);extra+=count
        current=r.live_poses(env); good=True
        for arm,t in targets.items():
            p,q=current[arm]; oldp,oldq=previous[arm]
            good &= (np.linalg.norm(r.tcp_of(p,q,offset)-t["tcp"]) <= .003 and r.quat_angle_deg(q,t["quat"]) <= 2
                     and np.linalg.norm(r.tcp_of(p,q,offset)-r.tcp_of(oldp,oldq,offset)) <= .0005 and r.quat_angle_deg(q,oldq)<=.2)
        stable=stable+1 if good else 0;previous=current
        if stable>=2:break
    after=r.live_poses(env); outcomes=[]
    for c in children:
        ok,note,metrics=r.motion_outcome(c,targets[c.arm],poses[c.arm],after[c.arm],offset)
        outcomes.append(dict(arm=c.arm,ok=ok,note=note,**metrics))
    if path_abort:
        outcomes.append({"arm":"both","ok":False,"note":path_abort})
    report["plan_status"]="Success"
    return dict(ok=all(v["ok"] for v in outcomes),note="; ".join(v["arm"]+": "+(v["note"] or "target reached") for v in outcomes),
                tcp_trace_m=trace,cartesian_push=is_push,children=outcomes,plans=infos,plan_status="Success",motion_steps=executed+extra,
                path_steps={a:len(p) for a,p in paths.items()},synchronized_steps=len(actions),
                settle_executed=extra,settle_saved=settle_steps-extra),stop
