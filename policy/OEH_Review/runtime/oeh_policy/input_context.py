"""Bounded input projections. Full measurements and archives remain execution authority."""
import copy
import json
import math
import re

DEFAULTS={'profile':'bounded-v1','episode_input_limit':5000000,'main_call_limit':200,
          'main_call_input_limit':15000,'semantic_call_limit':8,'semantic_call_input_limit':18000,
          'output_allowance':150000,'provider_reserve':7500,'tokenizer_multiplier':1.15,'max_entities':6}


def validate(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULTS)-{'enforce_budget'}:raise ValueError('unknown teacher.input_context settings')
    if 'enforce_budget' in value and type(value['enforce_budget']) is not bool:
        raise ValueError('input_context.enforce_budget must be boolean')
    result={**DEFAULTS,**value}
    if result['profile']!='bounded-v1':raise ValueError('unknown input context profile')
    for k in DEFAULTS:
        if k in {'profile','tokenizer_multiplier'}:continue
        if type(result[k]) is not int or result[k]<1:raise ValueError('invalid input context '+k)
    if not 1 <= result['tokenizer_multiplier'] <= 3:raise ValueError('invalid tokenizer multiplier')
    if result['main_call_limit']*result['main_call_input_limit']+result['semantic_call_limit']*result['semantic_call_input_limit']>result['episode_input_limit']:
        raise ValueError('input reservation envelope exceeds episode limit')
    return result


def rounded(value):
    if isinstance(value,float):return round(value,4)
    if isinstance(value,list):return [rounded(v) for v in value]
    if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
    return value


def referenced_entity_ids(value):
    return set(re.findall(r"\bentity_[a-zA-Z0-9]+\b",json.dumps(value,ensure_ascii=False)))


def spatial(snapshot,max_entities=6,entity_ids=()):
    preferred=set(entity_ids)
    es=sorted(snapshot.get('entities',[]),key=lambda e:(e.get('entity_id') not in preferred,e.get('visibility')!='visible',e.get('age_s',0)))
    result={'schema':'oeh.stmem.input.v2','time':snapshot.get('snapshot_time'),'entities':[],
            'initial_review_status':snapshot.get('initial_review_status','unknown'),
            'total':snapshot.get('total_entities',len(es)),'pending':snapshot.get('pending_observations',0)}
    for e in es[:max(max_entities,sum(e.get("entity_id") in preferred for e in es))]:
        item={k:copy.deepcopy(e[k]) for k in ['entity_id','label','description','visibility','motion','time','age_s','evidence','lifecycle','instruction_mentioned','interactable','avoid_interaction'] if k in e}
        item['description']=str(item.get('description',''))[:180]
        b=e.get('box',{});item['box']={k:copy.deepcopy(b[k]) for k in ['center','R','dimensions'] if k in b}
        for k in ['velocity_m_s','angular_velocity_rad_s']:
            if k in e:item[k]=e[k]
        item['uncertainty']={k:e[k] for k in ['position_uncertainty_m','orientation_ambiguous','observed_or_predicted'] if k in e}
        result['entities'].append(item)
    result['omitted']=max(0,result['total']-len(result['entities']))
    return rounded(result)


def packet(source,student_max_steps):
    fields=['request_id','candidate_id','observation_id','decision','native_step','remaining_steps','sim_time_s','task_started','instruction','task_spec','task_context','current_eef','episode_start_eef','home_offset','takeover','guard']
    result={k:copy.deepcopy(source[k]) for k in fields if k in source}
    proposal=copy.deepcopy(source.get('student_proposal',{}))
    if 'table' in proposal:
        proposal['table']=[r for r in proposal['table'] if r[0]<student_max_steps]
        proposal['shown_prefix_steps']=len(proposal['table'])
        proposal['diagnostics_scope']='full_original_proposal'
    result['student_proposal']=proposal
    previous=source.get('previous_result')
    if previous:
        result['previous_result']={k:copy.deepcopy(previous[k]) for k in ['mode','requested_steps','executed_steps','source','holds_before_execution','assessment'] if k in previous}
        result['previous_result']['assessment_source']='model_claim_not_verified'
    if source.get('semantic_context') is not None:result['semantic_context']=copy.deepcopy(source['semantic_context'])
    return rounded(result)


def text_tokens(value):
    import tiktoken
    text=value if isinstance(value,str) else json.dumps(value,separators=(',',':'),ensure_ascii=False)
    return len(tiktoken.get_encoding('o200k_base').encode(text,disallowed_special=()))


def input_estimate(body,prompt,tool,cfg):
    # Exclude base64 from text counts; preserve image positions and estimate patches separately.
    image_tokens=0
    def clean(x):
        nonlocal image_tokens
        if isinstance(x,dict):
            if x.get('type') in {'image','input_image'}:
                import base64,io
                from PIL import Image
                url=x.get('url',x.get('image_url',''))
                if isinstance(url,str) and url.startswith('data:'):
                    with Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))) as im:
                        image_tokens+=math.ceil(math.ceil(im.width/32)*math.ceil(im.height/32)*1.2)
                return {'type':'image'}
            if x.get('type')=='reasoning':return {'type':'reasoning','retained':True}
            return {k:clean(v) for k,v in x.items() if k not in {'encrypted_content'}}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    content=clean(body.get('input',[]))
    parts={'prompt':text_tokens(prompt),'tool_schema':text_tokens(tool),'current_input':text_tokens(content),'images_estimate':image_tokens}
    estimate=math.ceil(sum(parts.values())*cfg['tokenizer_multiplier'])+cfg['provider_reserve']
    return {'estimated_input':estimate,'parts':parts,'method':'o200k_base proxy + multiplier + calibrated provider reserve; not exact billing'}


PROMPT='''You control a dual-arm robot through robodojo_execute. Follow the original instruction.
Every response must match the supplied action schema. Review the observed result and proposed trajectory;
accept an aligned student prefix or use a bounded edit/eef correction when evidence warrants it.
student uses original candidate rows unchanged. edit/eef corrections and home must obey supplied limits.
Positions are meters in env-relative world; quaternion order wxyz. Proposal rows contain index then each
arm's xyz, roll/pitch/yaw degrees and opening (1 open). Diagnostics describe the full proposal; only the
shown prefix can be selected. Never infer registration, grasp, arrival or completion from a command.
Maintain task_progress using evidence; its verified_completed entries are YOUR claims, not benchmark truth.
Keep this agent conversation persistent; the framework does not reset it each decision.
Return one execution action after bounded queries. All assessment fields required by the schema remain required.
'''

# Shared with the normal runtime prompt; never duplicate the module protocol here.
from .llm_contract import interaction_contract
PROMPT += "\n" + interaction_contract() + "\n"
