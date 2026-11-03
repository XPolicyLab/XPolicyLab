"""Policy-host supervisor: standard WS server + isolated Agent, one session per reset.

Uses the active Python environment (XPolicyLab + the benchmark CuRobo required).
Each reset receives an independent agent workspace and archive.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from runtime import POLICY, DEFAULT_PREFIX, config, create_run_directory, save


from processes import stop


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',help='New run directory; relative paths use the XPolicyLab root. '
                   'Defaults to Q0_PREVIEW_RUN_DIR, then Q0_PREVIEW_RUN_ROOT/<uuid>, '
                   'then runs/Q0_Preview/<uuid> inside XPolicyLab.')
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--task',required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--python-prefix',default=DEFAULT_PREFIX);p.add_argument('--config')
    p.add_argument('--plans',type=int,default=100);p.add_argument('--wall-seconds',type=int,default=10800)
    p.add_argument('--turn-seconds',type=int,default=3600);p.add_argument('--smoke',action='store_true')
    p.add_argument('--checkpoint-seconds',type=int,default=60)
    p.add_argument('--smoke-two-group',action='store_true',help='API-free scripted CONTINUE/REPLAN gate regression; implies --smoke')
    args=p.parse_args();args.smoke=args.smoke or args.smoke_two_group
    if min(args.plans,args.wall_seconds,args.turn_seconds,args.checkpoint_seconds)<1:p.error('Limits must be positive')
    root=create_run_directory(args.root)
    print(f'[Q0-Preview] Run directory: {root}',flush=True)
    sessions=root/'sessions';sessions.mkdir()
    cfg=config(args.python_prefix,args.config)
    agent_env=os.environ.copy()
    agent_env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if not args.smoke:
        if not agent_env.get(cfg['api_key_env']):
            raise RuntimeError('Missing API environment variable: '+cfg['api_key_env'])
    save(root/'agent_config.json',cfg)
    internal=str(sessions)
    overrides=[f'port={args.port}',f'host={args.host}','bench_name=RoboDojo',f'task_name={args.task}',
        'ckpt_name=api','env_cfg_type=arx_x5',f'seed={args.seed}',f'policy_name={POLICY.name}',
        'action_type=joint',f'run_root={internal}',f'max_plans={args.plans}',
        f'wall_seconds={args.wall_seconds}',f'turn_seconds={args.turn_seconds}',f'checkpoint_seconds={args.checkpoint_seconds}']
    env=os.environ.copy()
    env.pop(cfg['api_key_env'], None)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES=str(args.gpu),
               PYTHONPATH=os.pathsep.join([str(POLICY.parents[2]),str(POLICY.parents[1])]))
    cmd=[sys.executable,'-u',str(POLICY.parents[1]/'setup_policy_server.py'),
         '--config_path',str(POLICY/'deploy.yml')]
    cmd+=['--overrides']+overrides
    children={};server=None
    def interrupted(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (root/'server.log').open('w') as log:
            server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        until=time.monotonic()+180
        while True:
            if server.poll() is not None:raise RuntimeError('Policy server exited; see server.log')
            listening = False
            for table in ('/proc/net/tcp', '/proc/net/tcp6'):
                path = Path(table)
                if path.exists():
                    listening |= any(row.split()[3] == '0A' and
                        int(row.split()[1].rsplit(':', 1)[1], 16) == args.port
                        for row in path.read_text().splitlines()[1:])
            if listening:
                break
            if time.monotonic()>until:
                raise TimeoutError('Policy server startup')
            time.sleep(.25)
        save(root/'SERVICE_READY.json',{'port':args.port,'backend':'native','pid':server.pid,'smoke':args.smoke})
        while server.poll() is None:
            if (root/'STOP').exists():break
            for ep in sessions.iterdir():
                if not (ep/'episode.json').exists() or ep in children:continue
                acmd=[str(Path(args.python_prefix)/'bin/python'),'-u',str(POLICY/'tools/agent_entry.py'),
                    'smoke_agent' if args.smoke else 'runner','--run-dir',str(ep)]
                if not args.smoke:acmd+=['--config',str(root/'agent_config.json')]
                if not args.smoke and cfg['provider']=='openai':acmd+=['--enable-openai']
                if args.smoke_two_group:acmd+=['--two-group']
                with (ep/'agent.log').open('w') as log:
                    child=subprocess.Popen(acmd,env=agent_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[ep]=(child,None)
            for ep,(child,ended_at) in list(children.items()):
                if (ep/'END.json').exists():
                    ended_at=ended_at or time.monotonic();children[ep]=(child,ended_at)
                    if time.monotonic()-ended_at>15:stop(child)
                elif child.poll() is not None:
                    (ep/'STOP').touch()
                    raise RuntimeError(f'Agent exited before END: {ep.name} (see agent.log)')
            time.sleep(.1)
        if server.poll() not in (None,0):raise RuntimeError('Policy server failed; see server.log')
    except KeyboardInterrupt:
        pass
    finally:
        for ep,(child,_) in children.items():
            (ep/'STOP').touch();stop(child)
        stop(server)

if __name__=='__main__':main()
