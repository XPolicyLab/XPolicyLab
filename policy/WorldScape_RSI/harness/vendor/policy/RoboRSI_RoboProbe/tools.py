"""Reviewed Skill tools produce native actions; WAM Skills reuse the WSP adapter."""
import io,json,math
from collections.abc import Sequence
from dataclasses import replace
import numpy as np
from PIL import Image
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import Action,ActionChunk
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import MotionOutcome,encode_jpeg_data_uri
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF import pose
from .visual_alignment import correction

WAM_IDS={'worldscape-wam-memory','worldscape-wam-nomemory'}
CODE_IDS={'primitive-control','robot-action-timing-tool','other-arm-observer'}|WAM_IDS
BUILTINS={'move_tcp','move_both_tcp','set_gripper','set_grippers'}

class HeldActions(Sequence):
    """A constant-size hold plan; materialize only actions the evaluator consumes."""
    def __init__(self,data,count):
        self.data={key:np.array(value,copy=True) for key,value in data.items()}
        self.count=count
    def __len__(self):return self.count
    def __getitem__(self,index):
        if isinstance(index,slice):return HeldActions(self.data,len(range(self.count)[index]))
        if index<0:index+=self.count
        if not 0<=index<self.count:raise IndexError(index)
        return Action(data={key:value.copy() for key,value in self.data.items()},meta={})

def function(name,description,props=None,required=()):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':props or {},'required':list(required),'additionalProperties':False}}}

def specs(library):
    ids=lambda category:[m['id'] for m in library.catalog(category)]
    paging={'offset':{'type':'integer','minimum':0},'stage':{'type':'string'}}
    result=[function('view_env_state','Read the current official observation without advancing the environment.'),
      function('list_skills','List current-task and generic Skills.'),function('list_tools','List available reviewed Tools, including the two WAM Skills.'),
      function('read_skill','Read complete Skill text; follow next_offset.',{'id':{'type':'string','enum':ids('skill')},**paging},['id']),
      function('read_tool','Read the Tool interface before using it.',{'id':{'type':'string','enum':ids('tool')+sorted(BUILTINS)},**paging},['id']),
      function('call_tool','Run a reviewed Tool. WAM Skill ids take empty arguments and propose a checkpoint-backed action chunk. Only the official evaluator executes native plans.',{'id':{'type':'string','enum':sorted(CODE_IDS)},'arguments':{'type':'object','description':'For robot-action-timing-tool and primitive-control, include op as a string in this object together with arm and other parameters. Example: {op: move, arm: left, position: [x,y,z]}. Do not nest under move and do not use action instead of op.','properties':{'op':{'type':'string','enum':['move','grip','grip_both','move_both','visual_align','move_tcp','move_ee','shift_ee','rotate_ee','wait']},'arm':{'type':'string','enum':['left','right']}}},'stage':{'type':'string'}},['id','arguments']),
      function('inspect_rgb_region','Enlarge current RGB only. region is normalized [left,top,right,bottom] in 0..1; mapping returns full-image pixel coordinates.',{'camera':{'type':'string'},'region':{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':4,'maxItems':4}},['camera','region'])]
    return result

def vector(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Expected '+str(n)+' finite values')
    return a

def tcp_targets(arm,position,q=None):
    if arm not in pose.ARMS:raise ValueError('arm must be left or right')
    out={arm+'_'+k:float(v) for k,v in zip(pose.POSITION_AXES,vector(position,3))}
    if q is not None:out.update({arm+'_'+k:float(v) for k,v in zip(pose.ANGLE_AXES,pose.quat_to_angles(vector(q,4)))})
    return out

def execute(policy,name,args,obs):
    lib=policy.library
    if not isinstance(args,dict):raise ValueError('Arguments must be an object')
    if name=='list_skills':return {'skills':lib.catalog('skill')}
    if name=='list_tools':return {'tools':lib.catalog('tool')+[{'id':k,'kind':'builtin'} for k in sorted(BUILTINS)]}
    if name in ('read_skill','read_tool'):
        key=args['id']
        if name=='read_tool' and key in BUILTINS:return {'id':key,'interface':lib.read('robot-action-timing-tool','tool')['text'],'note':'move_tcp accepts arm, position, optional quaternion_wxyz; set_gripper accepts arm, opening; set_grippers accepts left and right; move_both_tcp accepts left/right pose objects. No steps or timing overrides.'}
        return lib.read(key,'skill' if name=='read_skill' else 'tool',args.get('offset',0),args.get('stage'))
    if name=='view_env_state':
        points={a:pose.pose_to_values(obs.state[a+'_ee_pose']) for a in pose.ARMS}
        return {'state':{k:v.tolist() for k,v in obs.state.items()},'grasp_points':points,
                'tool_poses':{a:{'position':[points[a][k] for k in pose.POSITION_AXES],
                                'quaternion_wxyz':obs.state[a+'_ee_pose'][3:7].tolist()} for a in pose.ARMS},
                'instruction':obs.instruction,'cameras':list(obs.images)}
    if name=='inspect_rgb_region':
        image=Image.fromarray(obs.images[args['camera']]);left,top,right,bottom=vector(args['region'],4)
        if not 0<=left<right<=1 or not 0<=top<bottom<=1:raise ValueError('Use normalized left/top/right/bottom in 0..1')
        box=(math.floor(left*image.width),math.floor(top*image.height),math.ceil(right*image.width),math.ceil(bottom*image.height))
        crop=image.crop(box);scale=min(640/crop.width,480/crop.height);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
        policy.extra_images.append({'type':'image_url','image_url':{'url':encode_jpeg_data_uri(np.asarray(crop))}})
        return {'camera':args['camera'],'source_size':[image.width,image.height],'crop_size':list(crop.size),'full_pixel_mapping':{'offset_xy':list(box[:2]),'scale_xy':[(box[2]-box[0])/crop.width,(box[3]-box[1])/crop.height]},'executed_steps':0}
    if name=='move_both_tcp':
        targets={}
        for arm in pose.ARMS:targets.update(tcp_targets(arm,args[arm]['position'],args[arm].get('quaternion_wxyz')))
        return policy.native_move(targets,obs)
    if name!='call_tool':raise ValueError('Unavailable tool: '+name)
    key=args['id'];a=args.get('arguments',{})
    if not isinstance(a,dict):raise ValueError('Tool arguments must be an object')
    if any(k in a for k in ('motion_stage','mode','timeout','duration_s','reset','finish')):raise ValueError('Legacy controller/timing overrides are not supported; RoboProbe derives native actions')
    if key not in CODE_IDS|BUILTINS:raise ValueError('Unavailable tool: '+key)
    if key in CODE_IDS:
        entry=lib.entries[key]
        if args.get('stage') is not None and args['stage'] not in entry['stages']:raise ValueError('Invalid stage')
    if key in WAM_IDS:
        if a:raise ValueError('WAM Skills take empty arguments; observation and instruction come from the official environment')
        if not hasattr(policy,'wam_call'):raise ValueError('WAM Skill is unavailable in this adapter')
        return policy.wam_call(key,obs)
    if key=='move_both_tcp':return execute(policy,key,a,obs)
    if key=='set_grippers':return policy.native_move({arm+'_gripper':a[arm] for arm in pose.ARMS},obs)
    if key in ('set_gripper','move_tcp'):
        a={**a,'op':'grip' if key=='set_gripper' else 'move'};key='robot-action-timing-tool'
    if key=='other-arm-observer':
        working=a['working_arm']
        if working not in pose.ARMS:raise ValueError('working_arm must be left or right')
        arm='right' if working=='left' else 'left'
        if not a.get('move'):return {'observer_arm':arm,'flange_pose':obs.state[arm+'_ee_pose'].tolist(),'cameras':list(obs.images),'executed_steps':0}
        if 'degrees' in a:raise ValueError('Use a separate primitive-control rotate_ee call after observing the translation')
        return primitive(policy,{'op':'shift_ee','arm':arm,'delta':a.get('delta',[0,0,.02])},obs)
    if key=='primitive-control':return primitive(policy,a,obs)
    if any(k in a for k in ('steps','gripper')):raise ValueError('Timing is native; omit steps/gripper, use a separate grip call')
    op=a.get('op');arm=a.get('arm')
    if op not in ('move','grip','grip_both','move_both','visual_align'):raise ValueError('arguments.op must be move, grip, grip_both, move_both or visual_align; put arm and position/opening directly in arguments')
    if op=='move_both':return execute(policy,'move_both_tcp',a,obs)
    if op=='grip_both':return policy.native_move({s+'_gripper':a[s] for s in pose.ARMS},obs)
    if arm not in pose.ARMS:raise ValueError('arm must be left or right')
    if op=='grip':return policy.native_move({arm+'_gripper':a['opening']},obs)
    if op=='move':return policy.native_move(tcp_targets(arm,a['position'],a.get('quaternion_wxyz')),obs)
    if op=='visual_align':
        values=pose.pose_to_values(obs.state[arm+'_ee_pose']);tcp=[values[k] for k in pose.POSITION_AXES]
        if 'target_track' in a or 'lookahead_steps' in a:raise ValueError('Use current RGB target_pixel; native step counters are not observations')
        target=a['target_pixel']
        fit=correction(a['samples'],tcp,a['gripper_pixel'],target,a.get('max_correction_m',.06),a.get('gain',.8))
        result=policy.native_move(tcp_targets(arm,fit['position']),obs)
        return replace(result,tool_result=result.tool_result+'\nRGB correction: '+json.dumps(fit))
    raise ValueError('Unknown timing operation')

def primitive(policy,a,obs):
    op=a['op']
    if op=='wait':
        steps=a.get('steps',1)
        if type(steps) is not int or steps<1:raise ValueError('wait.steps must be a positive integer')
        # The evaluator stops iteration at native termination; no budget is observed.
        actions=HeldActions(policy._held_action_data(obs),steps)
        result='Planned native hold; evaluator owns execution and termination.'
        return MotionOutcome(chunk=ActionChunk(actions=actions,control_hz=policy.action_spec.control_hz),tool_result=result)
    if 'steps' in a:raise ValueError('Move/grip duration is derived by RoboProbe; omit steps')
    arm=a.get('arm')
    if arm not in pose.ARMS:raise ValueError('arm must be left or right')
    if op=='grip':return policy.native_move({arm+'_gripper':a['value']},obs)
    flange=obs.state[arm+'_ee_pose'].copy()
    if op=='move_tcp':return policy.native_move(tcp_targets(arm,vector(a['pose'],7)[:3],a['pose'][3:]),obs)
    if op=='move_ee':flange=vector(a['pose'],7)
    elif op=='shift_ee':flange[:3]+=vector(a['delta'],3)
    elif op=='rotate_ee':
        axis=a['axis'];degrees=float(a['degrees']);frame=a.get('frame','tool')
        if axis not in ('x','y','z') or frame not in ('world','tool') or not math.isfinite(degrees):raise ValueError('Use x/y/z axis, finite degrees and world/tool frame')
        q=np.zeros(4);q[0]=math.cos(math.radians(degrees)/2);q[1+('x','y','z').index(axis)]=math.sin(math.radians(degrees)/2)
        flange[3:]=pose.quat_multiply(q,flange[3:]) if frame=='world' else pose.quat_multiply(flange[3:],q)
    else:raise ValueError('Unknown primitive')
    values=pose.pose_to_values(flange)
    return policy.native_move({arm+'_'+k:v for k,v in values.items()},obs)
