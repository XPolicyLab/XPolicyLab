"""Strict text-only skill loading for the single shared policy."""
import hashlib
import json
import os
from pathlib import Path

def load_task_skill(task, root=None):
    project = Path(__file__).resolve().parent.parent
    catalog = json.loads((project/'configs/tasks.json').read_text())
    if task not in {t['task'] for t in catalog}:
        raise ValueError(f'Unknown benchmark task: {task}')
    if os.environ.get('ROBODAWN_TASK_SKILLS','1') != '1':
        raise ValueError('This experiment requires the frozen text skill loader')
    root = Path(root or os.environ.get('ROBODAWN_TASK_SKILL_ROOT',project/'skills'))
    paths=[root/'common.md',root/task/'SKILL.md']
    content='\n\n'.join(p.read_text(encoding='utf-8-sig') for p in paths)
    if len(content)>100000: raise ValueError('Skill exceeds reviewed 100000-character bound')
    return '\n\nTASK GUIDANCE (current observations override historical advice):\n'+content+'\n\n'+'Current ICL: approved frozen Open compact demonstrations where available; official task video otherwise. Retain every approved decision and terminal camera. Historical coordinates are references, never live observations. This takes precedence over older ICL-source wording in archived skills.', {
        'enabled':True,'task':task,'sha256':hashlib.sha256(content.encode()).hexdigest(),
        'characters':len(content),'files':[str(p) for p in paths],'behavioral_improvement_verified':False}
