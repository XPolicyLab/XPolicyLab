"""Use the matching frozen Open controller while retaining shared episode accounting."""
import importlib, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
CATALOG=json.loads((ROOT/'configs/open_restore.json').read_text())

def branch(task):
    return CATALOG['mapping'].get(task)

def activate(task):
    os.environ['ROBODAWN_RSI2_HARNESS_V1']='1'
    os.environ['ROBODAWN_POUR_FLOW']='1'
    os.environ['ROBODAWN_POUR_CONTINUE']='0'
    os.environ['ROBODAWN_POUR_MILESTONE20']='0'
    # These collection-only reward channels are deliberately inactive in this comparison.
    for key in ('ROBODAWN_PRIVILEGED_REWARD_FEEDBACK','ROBODAWN_REWARD_FEEDBACK',
                'RSI5_PRIVILEGED_REWARD_FEEDBACK','RSI6_REWARD_FEEDBACK'):
        os.environ[key]='0'
    os.environ['ROBODAWN_RUN_MODE']='benchmark'
    run=os.environ.get('RDA_RUN_DIR')
    if run:
        os.environ['RSI4_RUN']=run
        os.environ['RSI6_RUN']=run
        os.environ['RSI7_RUN']=run
    return branch(task)

def module(task, name):
    selected=activate(task)
    if selected:
        return importlib.import_module('.legacy.'+selected+'.'+name, __package__)
    return importlib.import_module('.'+name,__package__)

def metadata(task):
    return {'schema':'rda.open-restoration/v1','task':task,'branch':branch(task),
            'execution_mode':'registered_open_code' if branch(task) else 'shared_enhanced_actions',
            'enhanced_actions':True,'privileged_reward_feedback':False,
            'icl':'approved_open_compact_or_official_video','source_manifest':'configs/open_restore.json'}

def execute_sequence(task, *args):
    if not branch(task):
        raise ValueError('No registered Open branch')
    dep=module(task,'deploy')
    # Equivalent to the original episode initialization, once per session.
    session=args[6]
    if not session.get('_open_initialized'):
        for key in ('_ik_check_disabled','_calibration_disabled','_gripper_probe_disabled',
                    '_gripper_floor_disabled','_planner_disabled'):
            setattr(dep,key,False)
        session['_open_initialized']=True
    if task == 'align_blocks':
        module(task,'rsi4_live').start(args[0])
    return dep._run_sequence(*args)
