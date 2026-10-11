"""Strict whole-decision ICL v2. Never falls back to keyframes or truncates."""
import base64,hashlib,json
from pathlib import Path

REQUIRED=('decision_id','control_step','state','raw_model_response','original_model_fields',
          'submitted_commands_or_flow','parsed_executed_actions','complete_feedback','available_cameras','images')
def checked(root,rel,digest):
    p=(root/rel).resolve()
    try:p.relative_to(root.resolve())
    except ValueError:raise ValueError('ICL source escaped bank')
    blob=p.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('ICL source hash mismatch')
    return blob
def load_messages(root,max_images=None,max_content_bytes=None):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m.get('schema')!='conveyor.full_round_icl.v2' or m.get('status')!='complete':
        raise ValueError('Full-round ICL source incomplete; no keyframe fallback allowed')
    refs=m.get('rounds',[])
    if len(refs)!=m['decision_total'] or m['complete_decisions_exported']!=m['decision_total']:
        raise ValueError('Incomplete decision coverage')
    messages=[];count=0;size=0;coverage=[]
    records=[]
    for ref in refs:
        raw=checked(root,ref['record_path'],ref['record_sha256']);r=json.loads(raw)
        if any(k not in r for k in REQUIRED):raise ValueError('Missing original decision fields')
        if not isinstance(r['original_model_fields'],dict):raise ValueError('Original model fields unavailable')
        records.append((r,raw,False))
    if [r['decision_id'] for r,_,_ in records]!=list(range(1,m['decision_total']+1)):
        raise ValueError('Merged, missing or reordered decision IDs')
    steps=[r['control_step'] for r,_,_ in records]
    if steps!=sorted(steps):raise ValueError('Out-of-order observation boundary')
    terminal=m['terminal_observation'];raw=checked(root,terminal['record_path'],terminal['record_sha256']);r=json.loads(raw)
    if r.get('kind')!='terminal_observation' or r.get('is_decision') is not False:
        raise ValueError('Terminal observation must not masquerade as a decision')
    records.append((r,raw,True))
    for record,raw,is_terminal in records:
        cameras=record['available_cameras'];images=record['images']
        if len(set(cameras))!=len(cameras) or sorted(i['camera'] for i in images)!=sorted(cameras):
            raise ValueError('Not all available cameras included exactly once')
        parts=[{'type':'text','text':('TERMINAL OBSERVATION, NOT AN EXTRA DECISION\n' if is_terminal else 'COMPLETE ORIGINAL DECISION\n')+raw.decode('utf-8')}]
        size+=len(raw)
        for im in images:
            if im.get('original_resolution_preserved') is not True:raise ValueError('Original resolution not preserved')
            data=checked(root,im['path'],im['sha256']);size+=len(data);count+=1
            parts.append({'type':'text','text':im['camera']})
            parts.append({'type':'image_url','image_url':{'url':'data:'+im['mime_type']+';base64,'+base64.b64encode(data).decode(),'detail':'high'}})
        messages.append({'role':'user','content':parts})
        coverage.append({'decision_id':record.get('decision_id'),'control_step':record['control_step'],'terminal':is_terminal,'cameras':cameras})
    if max_images is not None and count>max_images:raise ValueError('Full ICL exceeds configured image limit; report, do not sample')
    if max_content_bytes is not None and size>max_content_bytes:raise ValueError('Full ICL exceeds configured content limit; report, do not truncate')
    proof={'status':'constructed_not_proof_of_model_display','decisions':m['decision_total'],
        'images':count,'bytes_before_base64':size,'coverage':coverage,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'compression_in_loader':False,'downstream_image_or_context_limits':'unknown unless explicitly configured',
        'actual_desktop_display_coverage':'must be recorded separately after reading every round and viewing every image'}
    return messages,proof
