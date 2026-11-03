"""No API: remote controller smoke; optional scripted two-group gate regression."""
import argparse
from pathlib import Path
import time
from ..runtime.io_utils import atomic, read_json
from ..runtime.protocol import validate_plan


def target(state, dx=0., dz=0., contact=False):
    p=[float(x) for x in state['grasp_point']];p[0]+=dx;p[2]+=dz
    j=state['jaw_axis'];axis=state['tool_axis']
    return {'target_points':[[p[i]-.02*j[i] for i in range(3)], [p[i]+.02*j[i] for i in range(3)]],
        'orientation_point':[p[i]-.1*axis[i] for i in range(3)],
        'contact_constraint':{'surface_point':p,'surface_normal':[0.,0.,1.], 'working_point_local':[0.,0.,0.], 'tolerance_mm':2.} if contact else None}


def group(states, dx=0., dz=0., contact=False, active='both'):
    return {'active_arms':active,
        **{a:target(states[a],dx,dz,contact) if active in ('both',a) else None for a in ('left','right')},
        'grippers':{a:{'command':'keep','width_mm':None,'when':'after'} for a in ('left','right')},
        'duration_s':.5 if active=='none' else 0.}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    p.add_argument('--two-group',action='store_true');args=p.parse_args()
    root=Path(args.run_dir);seen=set();plans=0;checks=0;home=None;began=time.monotonic()
    while not (root/'END.json').exists() and not (root/'STOP').exists():
        if time.monotonic()-began>2100: raise TimeoutError('smoke harness startup deadline')
        req=read_json(root/'READY.json')
        if not req or req['request_id'] in seen:time.sleep(.1);continue
        seen.add(req['request_id'])
        obs=read_json(root/'observations'/req['observation_id']/'observation.json')
        if args.two_group:
            if req['kind']=='checkpoint':
                checks+=1
                assert obs['last_execution']['status']=='complete'
                assert all(v['position_error_mm']<=5 for v in obs['last_execution']['arrival'].values())
                # This is deliberately scripted, not a substitute for the real visual reviewer.
                reply={'decision':'CONTINUE' if checks==1 else 'REPLAN',
                       'reason':'Scripted API-free integration gate: '+('allow continuation' if checks==1 else 'exercise pending-target cancellation')}
                atomic(root/f'smoke_checkpoint_{checks}.json', {'request':req, 'verdict':reply,
                    'source_observation_index':obs['source_observation_index'], 'control_steps_used':obs['control_steps_used']})
            elif plans<2:
                home=home or obs['state']
                first=group(home,dz=.01)
                first.update(checkpoint=True,expectation='Both arms reach the 1 cm raised intermediate; return to the known home target remains clear.')
                reply={'groups':[first,group(home)]};validate_plan(reply);plans+=1
            elif plans==2:
                assert checks==2, 'Both review branches must have been exercised'
                reply={'groups':[group(home)]};validate_plan(reply);plans+=1
            else:
                reply={'tool':'harness_stop','reason':'integration_smoke_complete'}
        elif plans==0:
            home=obs['state']
            reply={'groups':[group(home,dz=.01)]}
            validate_plan(reply);plans+=1
        elif plans==1:
            # Keep the passive-physics check independent of any unreachable IK
            # target: rejecting a motion must not skip the diagnostic wait.
            wait=group(obs['state'],active='none')
            # Use a short diagnostic wait within the motion/wait phase limit.
            reply={'groups':[wait]}
            validate_plan(reply);plans+=1
        elif plans==2:
            reply={'groups':[group(home)]}
            validate_plan(reply);plans+=1
        else:
            reply={'tool':'harness_stop','reason':'integration_smoke_complete'}
        atomic(root/'observations'/req['observation_id']/'reply.json',
            {k:req[k] for k in ('run_id','observation_id','request_id')}|{'payload':reply})


if __name__=='__main__':main()
