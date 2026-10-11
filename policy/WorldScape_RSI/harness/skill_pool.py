"""Frozen, executable bindings and composable text guidance; no task-to-winner table."""
import hashlib
import json
from pathlib import Path

def load():
    raw=(Path(__file__).resolve().parent.parent/'configs/skill_pool.json').read_bytes()
    obj=json.loads(raw)
    if obj.get('schema') != 'rda.skill-pool/v1':
        raise ValueError('Unknown skill pool schema')
    for name,entry in obj['skills'].items():
        expected='text_guidance' if entry['executor']=='astra' else 'wam.run_wam_segment'
        if (entry['executor'] not in ('astra','wam') or entry['binding']!=expected
            or not isinstance(entry['guidance'],str)):
            raise ValueError('Unimplemented skill binding: '+name)
    return obj['skills'],hashlib.sha256(raw).hexdigest()

def catalogue():
    pool,_=load()
    return {name:{k:v for k,v in entry.items() if k!='guidance'} for name,entry in pool.items()}

def validate(executor,skills):
    pool,_=load()
    if (not isinstance(skills,list) or not 1<=len(skills)<=5
        or any(not isinstance(x,str) for x in skills)
        or len(set(skills))!=len(skills)):
        raise ValueError('Skills must be a nonempty bounded list of unique handles')
    if any(name not in pool or pool[name]['executor']!=executor for name in skills):
        raise ValueError('Unknown skill or incompatible executor')
    if executor=='wam' and skills!=['continuous_visual_control']:
        raise ValueError('WAM only supports continuous_visual_control in this version')
    if executor=='astra' and 'verify_recover' not in skills:
        raise ValueError('Astra requires verify_recover')
    return skills

def guidance(plan):
    validate(plan['executor'],plan['skills'])
    pool,revision=load()
    if plan.get('skill_pool_sha256')!=revision:
        raise ValueError('Skill pool changed after selection')
    if plan['executor']=='wam':
        return ''
    return '\n\nMANAGER-SELECTED SHARED SKILLS:\n'+'\n'.join(
        name+': '+pool[name]['guidance'] for name in plan['skills'])
