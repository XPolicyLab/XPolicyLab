"""Write terminal-only native check booleans; never used in a live observation."""
import json
import os
from pathlib import Path

def write_terminal_checks(env):
    if not env.is_episode_end() or not hasattr(env, '_cat0_checks'):
        return
    root = os.environ.get('ROBODAWN_DESKTOP_BRIDGE')
    if not root:
        return
    rm = env.reward_manager
    names = ('inside', 'below_height', 'no_other_category_a', 'no_other_category_b', 'grippers_open')
    result = {'terminal_only': True, 'not_policy_input': True, 'categories': {}}
    try:
        for idx in range(3):
            checks = getattr(env, f'_cat{idx}_checks')()
            result['categories'][f'cat{idx}'] = {
                name: bool(rm.check_once(check, 0)) for name, check in zip(names, checks)}
        result['robot_home'] = bool(rm.check_once(rm.all_robot_back_to_origin(), 0))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    path = Path(root).parent / 'posthoc_review'
    path.mkdir(exist_ok=True)
    (path / 'native_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
