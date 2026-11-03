"""Bounded spatial reads. Images are returned only after an explicit model request."""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from PIL import Image

FIELDS = {'label','description','box','bbox_2d','visibility','motion','velocity_m_s','angular_velocity_rad_s','evidence','instruction_mentioned','interactable','avoid_interaction','lifecycle','current','last_complete','tracking_support','pose_status'}


MAX_IMAGES=32


class QueryInputError(ValueError):
    def __init__(self,field,message):
        self.field=field
        super().__init__(message)


def query_schemas():
    """Operation-specific provider schemas and handler validation share these constraints."""
    integer=lambda low,high=None: {'type':'integer','minimum':low,**({'maximum':high} if high else {})}
    number={'type':['number','null']}
    page={'offset':integer(0),'limit':integer(1,32)}
    fields={'type':'array','items':{'type':'string','enum':sorted(FIELDS)},'maxItems':len(FIELDS)}
    images={'include_images':{'type':'boolean'},'image_kind':{'type':'string','enum':['bbox','segmented','both','last_complete_bbox']},
        'image_times':{'type':'array','items':{'type':'number'},'maxItems':32},'image_offset':integer(0),'max_images':integer(1,MAX_IMAGES)}
    schemas=[]
    for op in ['snapshot','resolve','current','at','trajectory']:
        props={'kind':{'type':'string','enum':['memory_query']},'operation':{'type':'string','enum':[op]}}
        if op=='resolve':props.update(label={'type':'string','minLength':1,'description':'One exact label or alias already published by STMem. Not a natural-language object list.'},**page)
        elif op=='snapshot':props.update(fields=fields,**page,include_images={'type':'boolean'},max_images=integer(1,MAX_IMAGES))
        else:
            props.update(entity_id={'type':'string','minLength':1,'description':'Exact entity_id previously returned by STMem; never substitute a label or free text.'},fields=fields,**images)
            if op=='at':props['start']={'type':'number','description':'Requested simulation time. Last evidence at or before it, never a future image.'}
            if op=='trajectory':props.update(start=number,end=number,**page)
        schemas.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
    schemas.append({'type':'object','additionalProperties':False,'properties':{'kind':{'type':'string','enum':['memory_query']},
        'operation':{'type':'string','enum':['initial']},'entity_id':{'type':'string','minLength':1},'include_images':{'type':'boolean'}},
        'required':['kind','operation','entity_id','include_images']})
    schemas.append({'type':'object','additionalProperties':False,'properties':{
        'kind':{'type':'string','enum':['memory_query']},'operation':{'type':'string','enum':['frames']},
        'native_steps':{'type':'array','items':{'type':'integer','minimum':0},'maxItems':128},
        'cameras':{'type':'array','items':{'type':'string','enum':['cam_head','cam_left_wrist','cam_right_wrist']},'maxItems':3},
        'k':integer(1,16),'offset':integer(0),'limit':integer(1,32),'max_bytes':integer(1024,16777216)},
        'required':['kind','operation','native_steps','cameras','k','offset','limit','max_bytes']})
    schemas.append({'type':'object','additionalProperties':False,'properties':{
        'kind':{'type':'string','enum':['memory_query']},'operation':{'type':'string','enum':['frame_calibration']},
        'frame_id':{'type':'string','minLength':1},
        'cameras':{'type':'array','items':{'type':'string','enum':['cam_head','cam_left_wrist','cam_right_wrist']},'maxItems':3}},
        'required':['kind','operation','frame_id','cameras']})
    return schemas


def validate_query(arguments):
    op=arguments.get('operation');schema=next((x for x in query_schemas() if x['properties']['operation']['enum']==[op]),None)
    if schema is None:raise QueryInputError('operation','Use snapshot, resolve, current, initial, at, trajectory, frames or frame_calibration')
    required=['operation']+(['label'] if op=='resolve' else ['entity_id'] if op not in ['snapshot','frames','frame_calibration'] else [])+(['start'] if op=='at' else [])
    if op=='frame_calibration':required.append('frame_id')
    for key in required:
        if key not in arguments:raise QueryInputError(key,'Missing '+key+('; use exact archived scene frame_id' if op=='frame_calibration' else '; use exact STMem entity_id for entity/image queries'))
    for key,value in arguments.items():
        rule=schema['properties'].get(key)
        if rule is None:raise QueryInputError(key,'Unsupported '+key+' for '+op+'; use entity_id, or resolve one exact published label')
        types=rule['type'] if isinstance(rule['type'],list) else [rule['type']]
        valid=any((t=='null' and value is None) or (t=='string' and isinstance(value,str)) or
            (t=='boolean' and type(value) is bool) or (t=='integer' and type(value) is int) or
            (t=='number' and type(value) in [int,float] and math.isfinite(value)) or (t=='array' and isinstance(value,list)) for t in types)
        if not valid:raise QueryInputError(key,'Invalid type for '+key)
        if 'enum' in rule and value not in rule['enum']:raise QueryInputError(key,'Unsupported '+key)
        if type(value) in [int,float] and (value<rule.get('minimum',-math.inf) or value>rule.get('maximum',math.inf)):raise QueryInputError(key,'Out of range '+key+'; '+str(rule))
        if isinstance(value,str) and len(value)<rule.get('minLength',0):raise QueryInputError(key,key+' must not be empty')
        if isinstance(value,list):
            if len(value)>rule.get('maxItems',len(value)):raise QueryInputError(key,'Too many '+key+'; paginate requests')
            item=rule['items']
            if any((item['type']=='string' and (not isinstance(v,str) or v not in item['enum'])) or (item['type']=='number' and (type(v) not in [int,float] or not math.isfinite(v))) or (item['type']=='integer' and (type(v) is not int or v<item.get('minimum',0))) for v in value):raise QueryInputError(key,'Invalid '+key+' entry')
    if op=='trajectory' and arguments.get('start') is not None and arguments.get('end') is not None and arguments['start']>arguments['end']:raise QueryInputError('end','end must be >= start')
    return arguments


def query_memory(memory, operation, entity_id=None, text=None, start=None, end=None, now=None,
                 fields=None, offset=0, limit=16, include_images=False, image_kind='bbox',
                 image_time=None, max_images=1, label=None, image_times=None, image_offset=0):
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=32:raise QueryInputError('limit','offset >= 0 and limit 1..32 required')
    # Legacy Python arguments remain readable, but arbitrary text can never fetch entities/images.
    if text is not None:raise QueryInputError('text','Free text is not an entity key. Use entity_id or resolve(label=exact published label)')
    arguments={'kind':'memory_query','operation':operation,'offset':offset,'limit':limit}
    if operation=='resolve':arguments['label']=label
    elif operation=='initial':
        arguments.update(entity_id=entity_id,include_images=include_images);arguments.pop('offset');arguments.pop('limit')
    elif operation=='snapshot':arguments.update(fields=fields if fields is not None else [],include_images=include_images,max_images=max_images)
    else:
        arguments.update(entity_id=entity_id,fields=fields if fields is not None else [],include_images=include_images,image_kind=image_kind,image_times=image_times if image_times is not None else [],image_offset=image_offset,max_images=max_images)
        if operation not in ['trajectory']:arguments.pop('offset');arguments.pop('limit')
        if operation=='at':arguments['start']=start
        if operation=='trajectory':arguments.update(start=start,end=end)
    validate_query(arguments)
    for timestamp in (start,end,now,image_time):
        if timestamp is not None and (type(timestamp) not in [int,float] or not math.isfinite(timestamp)):raise QueryInputError('time','timestamps must be finite simulation seconds')
    if operation=='resolve':
        name=label.strip().casefold();candidates=[]
        for eid,e in memory.entities.items():
            names=[*e.get('aliases',[]),e.get('context_label',''),e.get('semantic_label','')]
            if name in {s.casefold() for s in names if isinstance(s,str)}:candidates.append({'entity_id':eid,'label':e.get('context_label',e.get('semantic_label',e['aliases'][0])),'score':1.})
        candidates.sort(key=lambda c:c['entity_id']);page=candidates[offset:offset+limit]
        return {'query_schema':'oeh.stmem.query.v4','operation':'resolve','label':label,'status':'resolved' if len(candidates)==1 else 'ambiguous' if candidates else 'not_observed','best':candidates[0] if candidates else None,'decisive':len(candidates)==1,'scores_are':'exact_alias_match_not_probability','candidates':page,'total_candidates':len(candidates),'next_offset':offset+limit if offset+limit<len(candidates) else None,'unmatched':{'allowed':True},'image_status':'ids_only_use_current_at_or_trajectory_with_entity_id'}
    if operation == 'snapshot':
        ids=list(memory.entities);page_limit=limit
        selected=ids[offset:offset+page_limit]
        result={'query_schema':'oeh.stmem.query.v4','operation':'snapshot','snapshot_time':now,
            'memory_revision':memory.sample_index,'working_set':copy.deepcopy(memory.working_set),'entities':[], 'total_entities':len(ids),
            'next_offset':offset+page_limit if offset+page_limit<len(ids) else None,'images':[]}
        for eid in selected:
            current=query_memory(memory,'current',entity_id=eid,fields=fields,now=now,include_images=False,image_kind='last_complete_bbox',max_images=1)
            result['entities'].append({k:v for k,v in current.items() if k not in {'_image_items','images'}})
            result['images'].extend(current.get('images',[]))
            if current.get('_image_items'):result.setdefault('_image_items',[]).extend(current['_image_items'])
        return result
    if entity_id not in memory.entities:return {'query_schema':'oeh.stmem.query.v4','status':'not_observed','entity_id':entity_id,'hint':'Use an exact entity_id from snapshot/notifications; never invent an ID'}
    if operation=='initial':
        result=memory.query('initial',entity_id=entity_id)
        result['query_schema']='oeh.stmem.query.v4'
        anchor=result.get('initial_anchor') or {}
        if include_images:
            from .stmem_images import asset
            root=memory.directory/'entities'/entity_id
            path=asset(root,anchor.get('assets',{}).get('bbox_crop_path'))
            if path:
                with Image.open(path) as original:
                    image=original.convert('RGB');image.thumbnail((32,32));buffer=io.BytesIO();image.save(buffer,format='PNG')
                data=buffer.getvalue();sha=hashlib.sha256(data).hexdigest()
                sent=memory.directory/'query-images'/(sha+'.png');sent.parent.mkdir(exist_ok=True);sent.write_bytes(data)
                result['images']=[{'entity_id':entity_id,'observed_time':anchor['observed_at'],'size':list(image.size),
                    'camera':anchor.get('camera'),'sha256':sha,'artifact':str(sent.relative_to(memory.directory)),'selection':'initial_anchor'}]
                result['_image_items']=[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii')}]
            else:result['image_status']='initial_image_missing'
        return result
    if operation == 'trajectory' and entity_id in memory.entities:
        entity=memory.entities[entity_id]
        selected=[s for s in entity['history'] if (start is None or s['time']>=start) and (end is None or s['time']<=end)]
        streams={
            'states':selected,
            'field_events':[e for e in entity.get('field_events',[]) if (start is None or e['time']>=start) and (end is None or e['time']<=end)],
            'tracking_states':[s for s in entity.get('tracking_history',[]) if (start is None or s['time']>=start) and (end is None or s['time']<=end)],
            'tracking_observations':[r for r in entity.get('tracking_observations',[]) if (start is None or r['time']>=start) and (end is None or r['time']<=end)],
            'visibility_events':[v for v in entity['visibility_history'] if (start is None or v['time']>=start) and (end is None or v['time']<=end)]}
        has_more=any(len(rows)>offset+limit for rows in streams.values())
        result={'entity_id':entity_id,'next_offset':offset+limit if has_more else None,
                'total_states':len(selected),'total_tracking_states':len(streams['tracking_states']),'truncated':has_more,
                **{key:copy.deepcopy(rows[offset:offset+limit]) for key,rows in streams.items()}}

    else:
        result=memory.query(operation,entity_id=entity_id,text=text,start=start,end=end,now=now)
    if fields:
        keep=set(fields)|{'entity_id','time','frame_id','status','age_s','observed_or_predicted'}
        if operation=='current':result={k:v for k,v in result.items() if k in keep}
        elif operation=='trajectory':result['states']=[{k:v for k,v in s.items() if k in keep} for s in result.get('states',[])]
        elif operation=='at' and result.get('last_observed_state'):
            result['last_observed_state']={k:v for k,v in result['last_observed_state'].items() if k in keep}
    result['query_schema']='oeh.stmem.query.v4'
    if not include_images:return result
    from .stmem_images import samples,asset,combine_capture
    root=memory.directory/'entities'/entity_id
    available=samples(memory,entity_id)
    if image_kind=='last_complete_bbox':available=[r for r in available if r.get('completeness')=='complete']
    times=image_times if image_times else [image_time] if image_time is not None else sorted({s['time'] for s in available if (start is None or s['time']>=start) and (end is None or s['time']<=end)}) if operation=='trajectory' else [start if operation=='at' else now]
    selected=[];missing=[]
    for timestamp in times:
        before=[r for r in available if (timestamp is None or r['time']<=timestamp) and (now is None or r['time']<=now)]
        if not before:
            missing.append({'requested_time':timestamp,'status':'no_complete_image_at_or_before_time' if image_kind=='last_complete_bbox' else 'no_observed_image_at_or_before_time'});continue
        row=max(before,key=lambda r:(r['time'],r.get('sample_index') or -1))
        for kind,key in ([('bbox','bbox_crop_path'),('segmented','crop_path')] if image_kind=='both' else [('bbox','bbox_crop_path')] if image_kind in ['bbox','last_complete_bbox'] else [('segmented','crop_path')]):
            selected.append((timestamp,row,kind,key))
    result.update(images=[],missing_images=missing,total_images=len(selected),next_image_offset=image_offset+max_images if image_offset+max_images<len(selected) else None)
    materialized={}
    for requested,row,kind,key in selected[image_offset:image_offset+max_images]:
        if row.get('_components'):
            capture=(row['time'],row.get('sample_index'),row.get('camera'))
            if capture not in materialized:materialized[capture]=combine_capture(root,row['_components'])
            row=materialized[capture]
        path=asset(root,row.get('assets',{}).get(key))
        if not path:
            result['missing_images'].append({'requested_time':requested,'observed_time':row['time'],'kind':kind,'status':'image_asset_unavailable'});continue
        with Image.open(path) as original:
            image=original.convert('RGBA');image.thumbnail((32,32))
            if image.getextrema()[3]==(255,255):image=image.convert('RGB')
            buffer=io.BytesIO();image.save(buffer,format='PNG')
        data=buffer.getvalue();sha=hashlib.sha256(data).hexdigest();sent=memory.directory/'query-images'/f'{sha}.png';sent.parent.mkdir(exist_ok=True);sent.write_bytes(data)
        result['images'].append({'entity_id':entity_id,'observed_time':row['time'],'sample_index':row.get('sample_index'),'kind':kind,'bbox_2d':row.get('bbox_2d'),'size':list(image.size),'sha256':sha,'artifact':str(sent.relative_to(memory.directory)),'requested_time':requested,'exact_observation':requested is not None and row['time']==requested,'source_observation_ids':row.get('source_observation_ids',[]),'pixel_coverage':row.get('pixel_coverage','original_observed_crop'),'explicit_model_request':True,'selection':'last_complete' if image_kind=='last_complete_bbox' else 'latest_observed_at_or_before'})
        result.setdefault('_image_items',[]).append({'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(data).decode('ascii'),'detail':'high'})
    result['image_status']='ok' if result['images'] else missing[0]['status'] if missing else 'image_asset_unavailable'
    memory.log('explicit_image_query',{'operation':operation,'entity_id':entity_id,'images':result['images'],'next_image_offset':result['next_image_offset']})
    return result
