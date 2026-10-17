"""Install small templates into the parent XPolicyLab workspace without overwriting files."""
from pathlib import Path
import json
import shutil
from delivery import POLICY_DIR, WORKSPACE


def main():
    templates = POLICY_DIR / 'templates'
    for src in (templates / 'models').rglob('*'):
        if not src.is_file():
            continue
        dest = WORKSPACE / src.relative_to(templates)
        if dest.exists():
            if dest.read_bytes() != src.read_bytes():
                raise FileExistsError(f'Existing file differs: {dest}')
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    env = WORKSPACE / 'env_cfg'
    env.mkdir(exist_ok=True)
    src = templates / 'env_cfg/piper_x.yml'
    dest = env / src.name
    if dest.exists() and dest.read_bytes() != src.read_bytes():
        raise FileExistsError(f'Existing Piper X environment differs: {dest}')
    if not dest.exists():
        shutil.copy2(src, dest)
    robot = env / 'robot/_robot_info.json'
    robot.parent.mkdir(exist_ok=True)
    existing = json.loads(robot.read_text()) if robot.exists() else {}
    piper = {'arm_dim': [6, 6], 'ee_dim': [1, 1]}
    if 'piper_x' in existing and existing['piper_x'] != piper:
        raise ValueError('Existing Piper X robot dimensions differ')
    if existing.get('piper_x') != piper:
        existing['piper_x'] = piper
        robot.write_text(json.dumps(existing, indent=2) + '\n')
    (WORKSPACE / 'assets/base').mkdir(parents=True, exist_ok=True)
    (WORKSPACE / 'assets/weights').mkdir(parents=True, exist_ok=True)
    link = POLICY_DIR / 'runtime/checkpoints'
    if not link.is_symlink() and not link.exists():
        link.symlink_to('../../../../assets/base', target_is_directory=True)
    if link.resolve() != (WORKSPACE / 'assets/base').resolve():
        raise ValueError('Unexpected runtime/checkpoints path; it must resolve to workspace/assets/base')
    print(f'Prepared {WORKSPACE}; next stage and verify assets.')


if __name__ == '__main__':
    main()
