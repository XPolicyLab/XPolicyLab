"""Verify official source bytes without importing a simulator or evaluator."""
import hashlib
import subprocess
from rsi.rule_configuration import settings

def runtime_hashes(runtime, mode):
    names = ['src/eval_client/eval_env.py', 'env/observation_manager/obs_manager.py',
             'env/robot_manager/robot_manager.py', 'env/robot_manager/control_manager.py',
             'env/scene_manager/layout_manager.py', 'env/seed_manager/seed_manager.py',
             'env/reward_manager/reward_manager.py', 'env/reward_manager/func_parser.py',
             'env/camera_manager/capture/render_sync.py', 'env/planner_manager/curobo_planner.py',
             'task/RoboDojo/tasks/' + mode + '.py', 'task/RoboDojo/config/' + mode + '.yml']
    return {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in names}

def verify_official_files(runtime, mode):
    observed = runtime_hashes(runtime, mode)
    for name, digest in observed.items():
        expected = subprocess.check_output(['git', '-c', 'safe.directory=' + str(runtime),
            '-C', str(runtime), 'show', settings()['official_commit'] + ':' + name])
        if hashlib.sha256(expected).hexdigest() != digest:
            raise ValueError('Official protected file differs from the pinned commit: ' + name)
    return observed
