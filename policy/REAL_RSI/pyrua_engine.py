import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .real_rsi_assets import model_path

POLICY_DIR = Path(__file__).resolve().parent
ARM_DIM, EE_DIM = [6, 6], [1, 1]
DEFAULTS = dict(robot='robodojo_xpl_rgb', primitives='no-vla', runtime='openai', model='offline-program',
                max_turns=100, budget_s=5400.0, images='on-demand', feedback='pure', poll_s=20.0, wrap_up_s=30.0,
                result_dir='./pyrua_results', openai_options={})

_LOCK = threading.Lock()
_STATE = {'program': None, 'conversation': -1, 'replies': 0}
_SERVER = None

def read_program(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'no PyRUA program at {path}')
    return path.read_text()

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        messages = body.get('messages') or []
        with _LOCK:
            if not any(m.get('role') == 'assistant' for m in messages):
                _STATE['conversation'] += 1
                _STATE['replies'] = 0
            conversation, reply, program = _STATE['conversation'], _STATE['replies'], _STATE['program']
            _STATE['replies'] += 1
        if reply == 0 and program:
            name, arguments = 'python', {'code': read_program(program)}
        else:
            name, arguments = 'finish', {'status': 'success', 'summary': 'offline program finished'}
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': f'call_{conversation}_{reply}', 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        data = json.dumps({'id': f'offline-{conversation}-{reply}', 'object': 'chat.completion',
                           'created': int(time.time()), 'model': body.get('model'),
                           'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls'}],
                           'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def start_endpoint():
    global _SERVER
    with _LOCK:
        if _SERVER is None:
            _SERVER = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
            threading.Thread(target=_SERVER.serve_forever, name='pyrua-programs', daemon=True).start()
        return f'http://127.0.0.1:{_SERVER.server_address[1]}/v1'

class PyruaEngine:
    def __init__(self, model_cfg, task, program):
        from XPolicyLab.utils.process_data import get_robot_action_dim_info
        from pyrualean.hosts.xpolicylab_bridge import Bridge

        dims = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        if list(dims['arm_dim']) != ARM_DIM or list(dims['ee_dim']) != EE_DIM:
            raise ValueError(f"PyRUA programs drive two six-joint arms with one gripper joint each; "
                             f"{model_cfg['env_cfg_type']} has {dims['arm_dim']} / {dims['ee_dim']}")
        read_program(program)
        self.task, self.program = task, Path(program)
        self.cfg = dict(DEFAULTS, **(model_cfg.get('pyrua') or {}))
        self.env_cfg_type = model_cfg['env_cfg_type']
        self.gpu = model_cfg.get('gpu_id')
        self.poll_s, self.wrap_up_s = float(self.cfg['poll_s']), float(self.cfg['wrap_up_s'])
        self.result_dir = Path(os.environ.get('PYRUALEAN_RESULT_DIR') or self.cfg['result_dir']).resolve()
        os.environ.setdefault('SAM3_CHECKPOINT_PATH', model_path('facebook__sam3/sam3.pt'))
        os.environ['OPENAI_BASE_URL'] = start_endpoint()
        os.environ.setdefault('OPENAI_API_KEY', 'offline')
        for var in ('PYRUALEAN_BASE_URL', 'PYRUALEAN_API_KEY_ENV'):
            os.environ.pop(var, None)
        self._bridge_cls = Bridge
        self.bridge = Bridge(poll_s=self.poll_s)
        self.episode = None
        self.episodes = 0

    def reset(self):
        self.bridge.abandon()
        self.bridge = self._bridge_cls(poll_s=self.poll_s)
        self.episode = None
        self.bridge.reset()

    def update_obs(self, obs):
        first = self.episode is None
        self.bridge.update_obs(obs)
        if first:
            self.episodes += 1
            self.episode = threading.Thread(target=self._play, args=(self.bridge, self.episodes), name='pyrua-episode',
                                            daemon=True)
            self.episode.start()

    def get_action(self):
        return self.bridge.get_action()

    def on_trial_end(self, result=None):
        self.bridge.episode_end(result)
        episode = self.episode
        if episode is not None and episode is not threading.current_thread():
            episode.join(self.wrap_up_s)
        return True

    def _play(self, bridge, count):
        from pyrualean.hosts import xpolicylab_robodojo as host
        from pyrualean.play import play_episode

        task = self.task
        cell = host.Cell(suite=self.env_cfg_type, task=task, seed=0,
                         cuda_device=int(self.gpu) if self.gpu not in (None, '') else None)
        with _LOCK:
            _STATE['program'] = self.program
        out = self.result_dir / f"{task}_{time.strftime('%Y%m%d_%H%M%S')}_{count}"
        c = self.cfg
        try:
            result = play_episode(cell, out, arm='cells', model=str(c['model']), robot=str(c['robot']),
                                  runtime=str(c['runtime']), primitives=str(c['primitives']),
                                  images=str(c['images']), feedback=str(c['feedback']),
                                  max_turns=int(c['max_turns']), budget_s=float(c['budget_s']),
                                  base_url=None, api_key_env=None, boot_options={'bridge': bridge},
                                  openai_options=dict(c['openai_options'] or {}))
            print(f"[REAL_RSI] {task} episode {count}: status {result.get('status')}  ({out})", flush=True)
        finally:
            bridge.finish()
