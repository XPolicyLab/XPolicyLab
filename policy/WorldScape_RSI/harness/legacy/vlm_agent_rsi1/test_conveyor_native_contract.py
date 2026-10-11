"""Run the actual native Python reward and terminal code without Isaac/physics.

Predicate responses are fixtures; this verifies update semantics, NOT a rollout.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, List, Tuple

from conveyor_reward_feedback import ConveyorRewardFeedback

ROOT = Path('/ML-vePFS/model_partner/ken/RoboDojo')
manager_path = ROOT / 'env/reward_manager/reward_manager.py'
task_path = ROOT / 'task/RoboDojo/tasks/pick_from_conveyor_by_image.py'
eval_path = ROOT / 'src/eval_client/eval_env.py'


def definition(path, kind, name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, kind) and n.name == name)


class Parser:
    def __init__(self, n):
        self.bottom = False
        self.lift = False
        self.calls = 0

    def _check_env_success(self, i):
        return True

    def is_A_on_B_bottom(self, args):
        self.calls += 1
        return self.bottom

    def is_lift(self, args):
        self.calls += 1
        return self.lift


namespace = dict(Any=Any, List=List, Tuple=Tuple, Func_Parser=Parser,
                 safe_deepcopy_keep_callable=copy.deepcopy, deepcopy=copy.deepcopy)
for path, kind, name in ((manager_path, ast.ClassDef, 'RewardManager'),
                         (task_path, ast.FunctionDef, 'run_reward'),
                         (eval_path, ast.FunctionDef, 'is_episode_end')):
    tree = ast.Module(body=[definition(path, kind, name)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)

manager = namespace['RewardManager'](1)
env = SimpleNamespace(reward_manager=manager, take_action_cnt=[0], step_lim=700, num_envs=1,
                      success=[True], end_flag=[False], get_obs_batch=lambda **kw: None)
namespace['run_reward'](env)
with TemporaryDirectory() as root:
    adapter = ConveyorRewardFeedback(env, root)
    baseline = adapter.snapshot(env, 'baseline', 'actual native registration')
    assert adapter.supported and baseline['current_score'] == 0
    for step, bottom, lift, expected in ((1, True, False, 0), (2, True, True, 1)):
        env.take_action_cnt[0] = step
        manager.func_parser.bottom, manager.func_parser.lift = bottom, lift
        manager.step()
        namespace['is_episode_end'](env)
        before = copy.deepcopy(vars(manager))
        packet = adapter.snapshot(env, str(step), 'native update completed')
        assert packet['current_score'] == expected
        assert before.keys() == vars(manager).keys()
        for key in before:
            if key == 'func_parser':
                assert vars(before[key]) == vars(manager.func_parser)
            else:
                assert before[key] == vars(manager)[key]
    assert packet['delta_score'] == 1
    assert packet['phase_conditions']['native_and_group']['historical'] is True
    assert packet['phase_conditions']['target_lifted']['current'] is None
    assert manager._gated_score_lst is None  # task has NO process scoring
print(json.dumps({'result': 'passed', 'native_source_executed': ['RewardManager', 'run_reward', 'is_episode_end'],
                  'predicate_inputs': 'fixtures, no physics rollout', 'snapshot_mutations': 0,
                  'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (manager_path, task_path, eval_path)}}, indent=2))
