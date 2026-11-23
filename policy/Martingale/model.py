# Copyright © 2026 Martingale. All Rights Reserved.
"""Fixed-system adapter: every task runs the SAME inference session."""
from pathlib import Path
import os
import threading
import uuid

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from martingale.hosts.xpolicylab_bridge import Bridge, WAIT
from martingale.inference.config import InferenceConfig
from martingale.inference.session import Session

# deploy.yml keys plus the server keys XPolicyLab's setup_policy_server.py adds (protocol, host, port, relay_url, pings)
ACCEPTED_KEYS=frozenset({'policy_name','protocol','host','port','relay_url','ws_ping_interval_s','ws_ping_timeout_s',
                         'bench_name','task_name','ckpt_name','env_cfg_type','seed','action_type','gpu_id','eval_batch',
                         'poll_s','inference_config','result_dir'})

class Model(ModelTemplate):
    CLEANUP_TIMEOUT_S = 30.0

    def __init__(self,model_cfg):
        super().__init__();cfg=dict(model_cfg or {})
        unknown=sorted(set(cfg)-ACCEPTED_KEYS)
        if unknown:raise ValueError('Unsupported deploy.yml keys: '+', '.join(map(str,unknown))+'. The accepted keys are listed in the adapter README.')
        if cfg.get('action_type','joint')!='joint':raise ValueError('Only joint actions are supported')
        self.config=InferenceConfig.load(cfg.get('inference_config') or os.environ['MARTINGALE_INFERENCE_CONFIG'])
        self.root=Path(cfg.get('result_dir') or os.environ['MARTINGALE_OUTPUT_ROOT']).resolve()
        self.gpu=cfg.get('gpu_id');self.poll_s=float(cfg.get('poll_s',2));self.bridge=Bridge(self.poll_s)
        mode=os.environ.get('MARTINGALE_RUN_GEOMETRY_FITTING','0')
        if mode not in ('0','1'):raise ValueError('MARTINGALE_RUN_GEOMETRY_FITTING must be 0 or 1')
        self.run_geometry_fitting=mode=='1'
        practice=os.environ.get('MARTINGALE_SELF_PLAY','0')
        if practice not in ('0','1'):raise ValueError('MARTINGALE_SELF_PLAY must be 0 or 1')
        self.self_play=practice=='1'
        if self.self_play and self.run_geometry_fitting:
            raise ValueError('Self-play subcalls and geometry fitting are separate modes')
        self.geometry_task=os.environ.get('MARTINGALE_GEOMETRY_TASK') or cfg.get('task_name') or 'unknown'
        self.session=None;self.thread=None
    def _wait_for_cleanup(self):
        if self.thread is None:return
        self.thread.join(timeout=self.CLEANUP_TIMEOUT_S)
        if self.thread.is_alive():
            raise RuntimeError('Previous episode cleanup is still running; retain this session and retry after cleanup completes.')
    def reset(self):
        # A session still held here never received its episode's end: the client left it after a
        # fault. End it (provider process, tool server and cell worker stop; its status file records
        # the client-side fault) before the new episode starts with a fresh session and fresh memory.
        if self.session is not None and self.bridge.over is None:
            self.bridge.abandon()
        # A verdict can arrive before provider termination, credential persistence and worker
        # shutdown finish. Never detach the old worker.
        self._wait_for_cleanup()
        self.bridge=Bridge(self.poll_s);self.session=None;self.thread=None
    def update_obs(self,obs):
        self.bridge.update_obs(obs)
        if self.session is None:
            self.session=Session(self.bridge,self.root/uuid.uuid4().hex,self.config,gpu=self.gpu,
                                 run_geometry_fitting=self.run_geometry_fitting,geometry_task=self.geometry_task,
                                 self_play=self.self_play)
            self.thread=threading.Thread(target=self.session.run,name='martingale-inference',daemon=False)
            self.thread.start()
    def get_action(self):
        actions=self.bridge.get_action()
        if self.run_geometry_fitting and self.session is not None and self.session.geometry_ready:
            return [{'martingale':'geometry_fitting_complete'}]
        if actions is WAIT:
            return actions
        return [{key: np.asarray(value) for key, value in action.items()} for action in actions]
    def on_trial_end(self,result=None):
        self.bridge.episode_end(result)
        self._wait_for_cleanup()
        return True
    # One conversation drives one environment (eval_batch: false). The batch entry points
    # accept exactly one environment and refuse more, so a batched client fails loudly.
    def update_obs_batch(self,obs_list):
        if len(obs_list)!=1:raise ValueError('Martingale runs one environment per process; keep eval_batch: false')
        self.update_obs(obs_list[0])
    def get_action_batch(self,env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list)!=1:raise ValueError('Martingale runs one environment per process; keep eval_batch: false')
        return [self.get_action()]
