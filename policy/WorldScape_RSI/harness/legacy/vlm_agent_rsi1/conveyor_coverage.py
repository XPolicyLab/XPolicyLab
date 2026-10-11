"""Persistent task-only coverage scheduler; no GPU or evaluator calls."""
from pathlib import Path
import json,hashlib,os,sys
TASK='pick_from_conveyor_by_image'
KEN=Path('/ML-vePFS/model_partner/ken')
CONTROL=KEN/'RoboDawn/task_control'/TASK
DELIVERY=KEN/'RoboDojo/collected_data/open_sft_v2'/TASK
LAYOUTS=KEN/'RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5'
def load(p,default=None):
    return json.loads(p.read_text()) if p.exists() else default
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def key(seed,layout):return (TASK,seed,layout)
def catalogue():
    rows=[]
    for directory in sorted(LAYOUTS.iterdir(),key=lambda p:int(p.name) if p.name.isdigit() else -1):
        if not directory.is_dir() or not directory.name.isdigit():continue
        files=sorted(directory.glob(TASK+'_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        for actual_id,p in enumerate(files):
            rows.append(dict(task=TASK,seed=int(directory.name),layout_id=actual_id,file=str(p),
                file_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return rows
def rebuild():
    import h5py
    actual=DELIVERY.resolve()
    assert actual==Path('/manifold-obs/ken/open_sft_v2')/TASK,'Unauthorized delivery resolution'
    manifest=load(DELIVERY/'dataset/manifest.json',{'files':[]})
    entries={e['episode']:e for e in manifest['files']}
    success=[]
    for p in sorted((DELIVERY/'dataset'/TASK/'arx_x5/data').glob('*.hdf5')):
        with h5py.File(p,'r') as f:
            a=f.attrs
            if not (a.get('success') and a.get('recording_valid') and a.get('eligible_for_sft') and 0<int(a.get('transitions',0))<=int(a.get('native_step_limit',0))==700):continue
            seed=int(a['collection_seed']) if 'collection_seed' in a else None
            layout=int(a['actual_layout_id']) if 'actual_layout_id' in a else None
            entry=entries.get(p.stem,{})
            preview=DELIVERY/entry.get('preview','__missing__')
            native=load(preview/'native_result.json',{})
            details=list(native.get('details',{}).values())
            # Single-episode artifact only; multi-episode native results need explicit linkage.
            if layout is None and len(details)==1 and details[0].get('success') is True:
                layout=details[0].get('layout_id')
            success.append(dict(task=TASK,episode=p.stem,path=str(p),seed=seed,layout_id=layout,
                identity_status='known' if seed is not None and layout is not None else 'unknown',
                source='delivered HDF5 explicit identity attributes; fallback single-episode native_result layout only'))
    known=[key(r['seed'],r['layout_id']) for r in success if r['identity_status']=='known']
    report={'task':TASK,'successful_files':len(success),'known_unique_combinations':len(set(known)),
        'known_duplicate_files':len(known)-len(set(known)),'unknown_identity_files':len(success)-len(known),
        'successes':success,'catalogue':catalogue(),'geometry_note':'File hashes are provenance, not proof of distinct physical geometry. Same layout across seeds is not claimed as new geometry.'}
    save(CONTROL/'coverage.json',report)
    return report
def choose(catalog,successes,attempts):
    successes=list(successes)+[a for a in attempts if a.get('outcome')=='success_pending_delivery']
    def excluded(c):
        # Missing identities conservatively match every compatible combination.
        return any((s.get('seed') is None or s['seed']==c['seed']) and
            (s.get('layout_id') is None or s['layout_id']==c['layout_id']) for s in successes)
    remaining=[c for c in catalog if not excluded(c)]
    counts={}
    for a in attempts:
        if a.get('seed') is not None and a.get('layout_id') is not None:
            k=key(a['seed'],a['layout_id']);counts[k]=counts.get(k,0)+1
    fresh=[c for c in remaining if key(c['seed'],c['layout_id']) not in counts]
    if fresh:return fresh[0]
    failed={key(a['seed'],a['layout_id']) for a in attempts if a.get('outcome') in ('failure','infrastructure_interrupted_resolved') and a.get('seed') is not None and a.get('layout_id') is not None}
    retry=[c for c in remaining if key(c['seed'],c['layout_id']) in failed]
    return min(retry,key=lambda c:counts[key(c['seed'],c['layout_id'])]) if retry else None
def attempts():return [load(p) for p in sorted((CONTROL/'attempts').glob('*.json'))]
def preflight(seed):
    state=load(CONTROL/'policy.json',{'paused':True})
    if state.get('paused',True):raise RuntimeError('Paused by user; no GPU launch authorized')
    try:from .conveyor_compact_icl import load_messages
    except ImportError:from conveyor_compact_icl import load_messages
    _,proof=load_messages()
    save(CONTROL/'icl_preflight.json',proof)
    report=rebuild()
    if report['successful_files']>=50:raise RuntimeError('Target quota reached; no new claim')
    selected=choose(report['catalogue'],report['successes'],attempts())
    if selected is None:raise RuntimeError('No unresolved legal combination; never wrap to successful scenes')
    if selected['seed']!=seed:raise RuntimeError('Requested seed differs from scheduler next seed: '+str(selected['seed']))
    save(CONTROL/'selection.json',selected)
    return selected
def apply_queue(env):
    selected=load(CONTROL/'selection.json')
    if load(CONTROL/'policy.json',{'paused':True}).get('paused',True):raise RuntimeError('Paused')
    assert selected and int(env.eval_seed)==selected['seed']
    manager=env.seed_manager
    path=Path(manager.seed_info[selected['layout_id']]['scene_layout'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==selected['file_sha256']
    report=rebuild()
    assert choose([selected],report['successes'],[]) is not None,'Already successful/unknown identity collision'
    manager.seed_list=[selected['layout_id']];manager.idx=0;manager.ed_idx=1;manager.st_idx=0
    manager._current_batch_seeds=None
def identity(env):
    seed=getattr(env,'eval_seed',None)
    layout=getattr(env,'current_env_seed_map',{}).get(0)
    return {'collection_seed':int(seed) if seed is not None else None,
            'actual_layout_id':int(layout) if layout is not None else None}
def check_episode(env):
    if load(CONTROL/'policy.json',{'paused':True}).get('paused',True):raise RuntimeError('Paused')
    selected=load(CONTROL/'selection.json')
    actual=identity(env)
    if not selected or (actual['collection_seed'],actual['actual_layout_id'])!=(selected['seed'],selected['layout_id']):
        raise RuntimeError('Actual reset identity does not match selected legal layout')
    report=rebuild()
    if choose([selected],report['successes'],[]) is None:raise RuntimeError('Duplicate successful or unresolved identity')
def record(env,path,attrs,reason):
    i=identity(env)
    success=bool(attrs.get('success')) and bool(attrs.get('eligible_for_sft'))
    outcome='success_pending_delivery' if success else ('interrupted_or_resource_unknown' if reason in ('interrupted','exception') else 'failure')
    save(CONTROL/'attempts'/(Path(path).stem+'.json'),dict(task=TASK,seed=i['collection_seed'],layout_id=i['actual_layout_id'],
        outcome=outcome,reason=reason,native_steps=int(attrs['transitions']),source_file=str(path)))
if __name__=='__main__':
    if sys.argv[1]=='rebuild':print(json.dumps(rebuild(),indent=2))
    elif sys.argv[1]=='preflight':print(json.dumps(preflight(int(sys.argv[2]))))
    elif sys.argv[1]=='publish':
        import subprocess
        if sys.argv[2]!=TASK:raise ValueError('Task mismatch')
        subprocess.run([sys.executable,str(Path(__file__).with_name('conveyor_package_native_delivery.py')),*sys.argv[2:]],check=True)
        print(json.dumps(rebuild(),indent=2))
