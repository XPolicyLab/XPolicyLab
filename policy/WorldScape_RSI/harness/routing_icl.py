"""Frozen visual references and separately sourced manager examples."""
import hashlib
import json
from pathlib import Path
from .icl_demos import _image

def build(demos, parse_plan):
    content=[]; references=[]
    for demo in demos:
        if demo.kind!='official_video' or demo.source.get('video_only') is not True:
            raise ValueError('Routing visual references require official video-only provenance')
        content.append({'type':'text','text':
            'REFERENCE VIDEO: visual behavior only, from a different scene. It provides no executor choice, '
            'robot actions, measured state or certified success. Current observations are authoritative.'})
        hashes=[]
        for frame in demo.frames:
            if frame.state_line or frame.commands or frame.plan or frame.effect or frame.failed:
                raise ValueError('Video references must not invent execution trace fields')
            content.append({'type':'text','text':frame.label})
            for _,blob in frame.images:
                hashes.append(hashlib.sha256(blob).hexdigest())
                content.append(_image(blob))
        references.append({'source':demo.source,'frame_sha256':hashes})
    path=Path(__file__).resolve().parent.parent/'icl/routing_examples.json'
    raw=path.read_bytes()
    data=json.loads(raw)
    if data.get('schema')!='rda.routing-examples/v1' or not isinstance(data.get('examples'),list):
        raise ValueError('Invalid routing example bank')
    if len(raw)>20000 or len(data['examples'])>4:
        raise ValueError('Routing example bank exceeds fixed context budget')
    examples=[]
    for item in data['examples']:
        if (set(item)!={'id','source_run','selection_source','observed_requirements','plan'}
            or item['selection_source']!='manager'
            or any(not isinstance(item[k],str) or not item[k].strip()
                   for k in ('id','source_run','observed_requirements'))):
            raise ValueError('Only sourced actual manager selections may be route examples')
        plan=parse_plan(json.dumps(item['plan']))
        examples.append({'observed_requirements':item['observed_requirements'],'plan':plan})
    if examples:
        content.append({'type':'text','text':
            'FROZEN DEVELOPMENT MANAGER EXAMPLES. These are observed decisions, not proven optimal '
            'choices or task-to-winner rules. Re-evaluate capability fit in the current scene.\n'+json.dumps(examples)})
    return ([{'role':'user','content':content}] if content else []), {
        'video_references':references,'routing_examples_sha256':hashlib.sha256(raw).hexdigest(),
        'routing_example_ids':[x['id'] for x in data['examples']],
        'routing_example_sources':[x['source_run'] for x in data['examples']]}
