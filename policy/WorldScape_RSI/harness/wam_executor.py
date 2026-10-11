"""WorldScape execution in the same native environment and recorder as Astra."""
import fcntl
import json
import time
from pathlib import Path
from .execution_router import settings
from .wam.client import RoboDojoClient
from .wam.control import run_wam_segment

def execute(env, config, capture, output):
    cfg = settings()['wam']
    deadline = time.monotonic() + float(config.get('episode_budget_s',1800))
    lock = Path(cfg['lock_path'])
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        # Cooperates with Ken's existing WAM launcher. No GPU/service restart.
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with RoboDojoClient(cfg['url'],cfg['timeout_s']) as client:
            Path(output,'wam_metadata.json').write_text(json.dumps(client.metadata,default=str,indent=2))
            remaining = int(env.step_lim)-int(env.take_action_cnt[0])
            if remaining <= 0 or env.is_episode_end():
                return {'termination_reason':'native','decisions':0}
            result = run_wam_segment(env,client,max_blocks=(remaining+23)//24,
                task=cfg['task'], seed=cfg['seed'],
                should_yield=lambda:time.monotonic() >= deadline or int(env.take_action_cnt[0]) >= int(env.step_lim))
            capture.decisions = result.blocks_requested
            native = env.is_episode_end()
            reason = 'native' if native else ('wall_budget' if time.monotonic()>=deadline else 'wam_block_limit')
            if not native:
                env.success[0] = False
                env.is_episode_end()
            return {'termination_reason':reason,'decisions':result.blocks_requested,
                    'wam_steps_executed':result.steps_executed}
