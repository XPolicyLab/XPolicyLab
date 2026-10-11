"""Toolbox-only staged procedure. Inputs are RGB-derived anchors, never scene state."""
from copy import deepcopy
import math
from . import commands
ORDER=('wrench','hammer','pliers','tape_measure')
ARMS={'tape_measure':'right','wrench':'left','hammer':'left','pliers':'right'}
PRESETS={k:(('down60','down75','down','down5in') if k in ('wrench','hammer') else ('down60','down75','down')) for k in ORDER}
class ToolboxFlow:
 def __init__(self):
  self.pregrasp_opening=None;self.placement_attempts=0;self.carry_height=94.;self.order=list(ORDER);self.index=0;self.phase='select';self.source=None;self.target=None;self.preset=None;self.pending=None;self.retries=0;self.events=[];self.rejection=None;self.settle_checked=False;self.release_executed=False
 @property
 def tool(self):return self.order[self.index] if self.index<len(self.order) else None
 def snapshot(self):return {'version':'toolbox_flow_v31_open_approach','tool':self.tool,'phase':self.phase,'recovery_batches':self.retries,'placement_retries':self.placement_attempts,'completed_visual_checks':list(self.order[:self.index]),'native_success_claimed':False,'rejection':self.rejection,'carry_height_cm':self.carry_height,'release_executed':self.release_executed}
 def prompt(self):
  return '\nTOOLBOX PROCEDURE (overrides free command planning): '+str(self.snapshot())+'''\nReturn ordinary scene/progress/memory plus "commands":[] and "workflow":{...}. Coordinates are absolute WORLD centimetres, derived from current RGB, not prior-layout replay.
Latest user correction: grasp the MIDDLE of each object for stability, not the rear end. Latest Ken instruction supersedes DOWN-only: approach nearer parallel to table to engage finger-pad MIDDLE, not tips. Ken accepted down60 (30deg below horizontal), upper coated shaft with opposed middle pads. Preserve that strategy and match accepted relative neck-to-pad spacing before contact; keep TOOL itself table-parallel. Contact area is a hypothesis to verify visually, not guaranteed by angle. locate requires grasp_region=center. At locate AND grasp require contact_mode=pad_middle and contact_evidence describing overlap along inner pads, opposed surfaces, and table clearance. source remains TCP at wrist+14.5cm, not actual tip: mesh tip reaches15.757cm, distal-pad longitudinal midpoint about14cm. Do not blindly equate TCP with tip or object balance point. For down75 source z>=78.3cm, down60 z>=77.9cm, conservative static table-clearance guard; actual tool/rim contact still needs live inspection. Ken correction: center means WHOLE TOOL balance region, never rubber-handle midpoint. Latest Ken: shift rearward off slippery neck onto broad straight black coating while preserving whole-tool balance. At locate AND grasp provide grasp_reference=whole_tool, mass_bias=headward|central, balance_evidence explaining current visible mass distribution and selected contact. Never claim exact center of mass from RGB. Short lift must show no pitch/yaw/slip before transport; jaw width alone is insufficient.\nselect: op=locate, source=[x,y,z] fingertip grasp point, target=[x,y,z] placement fingertip point, preset from tool allowed list. Code approaches 6cm above source.
approach: op=grasp descends ONLY, source refinement<=3cmXY/2cmZ. For larger visual correction use op=relocalize, source, localization_evidence, empty=true; requires observed open hand, move<=10cm, keeps hover and updates stored source. contact_check: inspect all cameras before op=close; require isolated_target=true, pad_overlap=true, contact_mode=pad_middle, contact_evidence describing both pads and neighbor exclusion. op=adjust_grasp permits <=1.5cm open-hand correction with source, clear_of_contact=true and contact_evidence; then recheck. closed_check: op=lift requires correct_tool=true, contact_evidence and observed opening>.20; lifts3cm ONLY, then fresh inspection. Never bypass these checks with recovery closure.
 grasp_check: op=raise, retained=true AND orientation_ok=true after short-lift images; code lifts long tools in <=7cm segments to94cm with a fresh retention check between segments, without rotating a held object, then waits for another image. Select a carry-reachable orientation before grasping; down15 inner carry may fail IK.
high_check: op=carry, retained=true AND orientation_ok=true after HIGH lift images; optional target refinement. Ken requires far/near transport: default far segment12cm, stopping at10cm distance from goal; within10cm use at most3cm. max_step_cm optional1..12 may reduce these limits. Fresh images after every segment. Long moves change distance only, never native speed or physics. Preserve orientation.
place_check: op=align permits horizontal-only0.05..1cm correction at unchanged z with retained=true, orientation_ok=true, clear_of_contact=true, alignment_evidence and target. If touching, use reapproach to lift before lateral correction. It keeps place_check and requires a new image. op=lower is vertical ONLY: target XY within0.2cm of actual pose, max_step_cm optional0.2..1 default1, retained=true, orientation_ok=true, target optional; code approaches support in at most1cm vertical chunks; every lower requires alignment_evidence PLUS head_in_recess=true, handle_in_channel=true, recess_floor_verified=true and placement_evidence={head:...,handle:...,floor:...}. Identify both tool endpoints relative to actual cavity edges, including tool-to-TCP offset and height/parallax uncertainty. Raised divider top is NOT recess floor. Unknown/occluded footprint blocks lowering, with fresh images between chunks, requesting a fresh image between chunks, never releases. Long tools default94cm. On raise only, optional clearance_height=83..94 requires object_parallel=true, clearance_confirmed=true and concrete clearance_evidence from current whole-tool/rim images. No descent. Recheck those fields at every reduced-height raise and carry; never infer flatness from jaw width. If hanging/pivoted/uncertain, retain94 or recover. This is a provisional lower-height route, not validated success.
release_check: op=release, supported=true AND aligned=true; opens ONLY, then fresh images. withdrawal: op=retreat, target absolute XYZ, empty=true, clearance_evidence required. Each move <=10cm; any horizontal component requires clear_of_tool=true. Choose vertical clearance first, then outward; no fixed tall lift. Once outside tray, phase released. Never repeat release after successful opening. A failed retreat can retry via op=retreat at most2 times with fresh evidence.
released: op=observe allows ONE native wait settle/trigger recheck. Score miss must be repaired or aborted, never confirmed. op=confirm, inside=true AND aligned=true; code advances to next tool. For a settled failed wrench placement, op=repair with empty=true,supported=true,repair_evidence,source,target,preset and normal contact/balance fields reapproaches open-handed; close/lift checks remain mandatory. Available while steps_used<700; preserve tool index and no native success claim. Use local current tray coordinates, not original table source. If repair infeasible, op=reject records failure; never assert alignment to advance. These are visual claims, never native success.
Empty approach: optional preshape with opening0.6..0.85, empty=true, clear_of_contact=true, clearance_evidence; only >=3cm above source, starts observed open>=0.8. Inspect actual width before descent. source_reapproach in recovery requires observed open empty hand, reason, lift_cm0.5..3; vertical unload restores selected preset, no general2 cutoff, before650. Repair hover now stays4..7cm above source, retaining final descent gate.
Unexpected dropped wrench: op=recover_drop in recovery with observed opening<=0.20, empty=true and drop_evidence; opens ONLY and requires fresh images. Then op=repair requires settled=true and all normal supported/contact/balance fields. No general-retry cutoff, before650 steps; no release/success claim. Never use this while held.
Any uncertainty/failure: op=recover, reason=concrete visual diagnosis, commands=[up to3 active-arm primitives], resume=approach|grasp_check|high_check|place_check|release_check. Opening requires safe_support=true. Maximum2 general recovery batches per tool. Placement retries use separate op=reapproach, retained=true, reason, lift_cm=0.5..3: lifts vertically ONLY, then fresh observation before align/lower. No two-attempt cutoff; continue evidence-guided placement retries while steps_used<800. Preserve grasp and native900 budget. Never repeat an identical failed offset without fresh evidence. Every recovery is followed by fresh observation.
finish: op=finish plans both arms home; finish_check: op=finish asks native checker. op=abort ends failed episode when recovery is exhausted or budget cannot cover completion+reset.
Tape constraint: casing must remain parallel to the table throughout grasp, lift, carry and lowering. Latest user correction permits more horizontal/oblique wrist with middle-pad contact while keeping casing parallel to table. Every tape raise/carry/lower/release requires object_parallel=true from current live images. Tilt/uncertainty blocks advancement. Tape recovery permits translation/gripper only; never rotate/point/pose3d/push a held casing. Solve placement with XY alignment and descent height, not tilt.\nArm assignments fixed: tape right, wrench left, hammer left, pliers right. Initial locate may provide order derived from current visual left-to-right positions; all4 tools exactly once and tape last. Order then fixed. Tool presets: '''+str(PRESETS)+'\n'
 @staticmethod
 def point(v):
  if not isinstance(v,list) or len(v)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v):raise ValueError('point must contain3 finite numbers')
  x,y,z=map(float,v)
  if not (-50<=x<=50 and -30<=y<=30 and 76.5<=z<=100):raise ValueError('point outside task workspace')
  return [x,y,z]
 def compile(self,payload,state):
  clone=deepcopy(self);out=clone._compile(payload,state)
  parsed,errors=commands.parse_command_list(out)
  if errors or len(parsed)>4:raise ValueError('invalid generated batch: '+str(errors))
  self.__dict__.update(clone.__dict__);return out
 def _compile(self,p,state):
  if not isinstance(p,dict):raise ValueError('workflow object required')
  op=p.get('op');used=state.get('steps_used',0) or 0
  if self.pending:raise ValueError('execution feedback pending')
  if op=='observe':
   if self.phase!='released' or self.settle_checked:raise ValueError('Only one post-release settle observation allowed')
   self.settle_checked=True
   return self.queue(['wait'],'released')
  if op=='reject':
   if self.phase!='released' or not p.get('reason'):raise ValueError('reject needs released phase and concrete reason')
   self.rejection={'tool':self.tool,'reason':p['reason']};self.phase='aborted';return ['done']
  if op=='abort':self.phase='aborted';return ['done']
  if self.phase=='aborted':return ['done']
  if self.phase in ('finish','finish_check'):
   if op!='finish':raise ValueError('finish required')
   if self.phase=='finish_check':return ['done']
   return self.queue(['left homeplan','right homeplan'],'finish_check')
  if 'order' in p:
   if op!='locate' or self.phase!='select' or self.index!=0 or used!=0:raise ValueError('Order may be selected once before any physical action')
   order=p['order']
   if not isinstance(order,list) or len(order)!=4 or set(order)!=set(ORDER) or order[-1]!='tape_measure':raise ValueError('Visual left-to-right order requires all four tools, tape last')
   self.order=list(order)
  arm=ARMS[self.tool];pos=list(state[arm]['position_cm'])
  if self.tool=='tape_measure' and op in ('raise','carry','lower','release') and p.get('object_parallel') is not True:raise ValueError('current visual evidence of table-parallel tape required')
  def move(target):
   d=[b-a for a,b in zip(pos,target)]
   if math.dist(pos,target)>39:raise ValueError('move exceeds39cm; split recovery')
   pos[:]=target
   return f'{arm} move3d '+ ' '.join(f'{v:.3f}' for v in d)
  if op=='preshape':
   if self.phase!='approach' or self.source is None or self.release_executed:raise ValueError('preshape requires empty approach')
   opening=state[arm].get('opening')
   if p.get('empty') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or opening<.8:raise ValueError('preshape starts from observed open empty hand')
   if pos[2]<self.source[2]+3 or p.get('clear_of_contact') is not True or not p.get('clearance_evidence'):raise ValueError('preshape only above source with visual clearance')
   width=p.get('opening')
   if type(width) not in (int,float) or not math.isfinite(width) or not .6<=width<=.85:raise ValueError('preshape width must be0.6..0.85')
   self.pregrasp_opening=width
   return self.queue([f'{arm} gripper {width:.3f}'],'approach')
  if op=='source_reapproach':
   if self.phase!='recovery' or self.release_executed or self.source is None:raise ValueError('source reapproach requires failed unreleased approach')
   if used>=650:raise ValueError('source reapproach needs native budget')
   opening=state[arm].get('opening');minimum=.8 if self.pregrasp_opening is None else max(.5,self.pregrasp_opening-.08)
   if p.get('empty') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or opening<minimum:raise ValueError('source reapproach requires observed open empty hand')
   if not isinstance(p.get('reason'),str) or not p['reason'].strip():raise ValueError('fresh source diagnosis required')
   dz=p.get('lift_cm',2)
   if type(dz) not in (int,float) or not math.isfinite(dz) or not .5<=dz<=3:raise ValueError('empty unload must lift0.5..3cm')
   self.point([pos[0],pos[1],pos[2]+dz])
   return self.queue([f'{arm} pose3d 0 0 {dz:.3f} '+self.preset],'approach')
  if op=='recover_drop':
   if self.tool!='wrench' or self.phase!='recovery' or self.release_executed:raise ValueError('drop recovery requires unreleased wrench recovery phase')
   if used>=650:raise ValueError('drop recovery needs native budget')
   opening=state[arm].get('opening')
   if p.get('empty') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or not 0<=opening<=.20:raise ValueError('observed empty closed hand required')
   if not isinstance(p.get('drop_evidence'),str) or not p['drop_evidence'].strip():raise ValueError('fresh visual drop evidence required')
   return self.queue([f'{arm} gripper open'],'drop_open_check')
  if op=='repair':
   released_ready=self.phase=='released' and self.release_executed and self.settle_checked
   dropped_ready=self.phase=='drop_open_check' and not self.release_executed and p.get('settled') is True
   if self.tool!='wrench' or not (released_ready or dropped_ready):raise ValueError('repair requires settled released or observed settled dropped wrench')
   if used>=700:raise ValueError('repair needs remaining native budget')
   opening=state[arm].get('opening')
   if p.get('empty') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or opening<.8:raise ValueError('repair needs observed open empty hand')
   if p.get('supported') is not True or not isinstance(p.get('repair_evidence'),str) or not p['repair_evidence'].strip():raise ValueError('fresh supported failed-placement evidence required')
   self._grasp_balance(p)
   source=self.point(p.get('source'));target=self.point(p.get('target'))
   preset=p.get('preset',self.preset)
   if preset not in PRESETS[self.tool]:raise ValueError('invalid repair preset')
   self._pad_contact(p,source,preset)
   hover=[source[0],source[1],max(source[2]+4,min(pos[2],source[2]+7))]
   self.point(hover)
   if math.dist(pos,hover)>25:raise ValueError('repair approach exceeds25cm')
   d=[b-a for a,b in zip(pos,hover)]
   self.pregrasp_opening=None;self.source=source;self.target=target;self.preset=preset
   self.release_executed=False;self.settle_checked=False
   return self.queue([f'{arm} pose3d '+' '.join(f'{v:.3f}' for v in d)+' '+preset],'approach')
  if op=='reapproach':
   if self.phase not in ('place_check','release_check','recovery') or self.target is None or self.release_executed:raise ValueError('reapproach requires held tool at placement before release')
   if used>=800:raise ValueError('native budget reserves100 steps for withdrawal/reset')
   opening=state[arm].get('opening')
   if p.get('retained') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or not .20<opening<.8:raise ValueError('fresh held-tool evidence required')
   if not isinstance(p.get('reason'),str) or not p['reason'].strip():raise ValueError('contact diagnosis required')
   dz=p.get('lift_cm',2)
   if type(dz) not in (int,float) or not math.isfinite(dz) or not .5<=dz<=3:raise ValueError('unload vertically0.5..3cm')
   target=self.point([pos[0],pos[1],pos[2]+dz])
   self.placement_attempts+=1
   return self.queue([move(target)],'place_check')
  if op=='retreat':
   if not self.release_executed or self.phase not in ('withdrawal','recovery'):raise ValueError('retreat requires verified completed release')
   opening=state[arm].get('opening')
   if p.get('empty') is not True or type(opening) not in (int,float) or not math.isfinite(opening) or opening<.8:raise ValueError('retreat requires observed open empty hand')
   if not isinstance(p.get('clearance_evidence'),str) or not p['clearance_evidence'].strip():raise ValueError('fresh withdrawal clearance evidence required')
   target=self.point(p.get('target'))
   if not .05<=math.dist(pos,target)<=10:raise ValueError('withdrawal step must be0.05..10cm')
   if math.dist(pos[:2],target[:2])>.05 and p.get('clear_of_tool') is not True:raise ValueError('lift clear before lateral withdrawal')
   if target[2]<pos[2]-.05:raise ValueError('withdrawal cannot descend')
   if self.phase=='recovery':
    if self.retries>=2:raise ValueError('recovery limit reached; abort')
    self.retries+=1
   outside=target[0]<=-20 if arm=='left' else target[0]>=22
   return self.queue([move(target)],'released' if outside and p.get('clear_of_tool') is True else 'withdrawal')
  if op=='align':
   if self.phase!='place_check':raise ValueError('align only at placement inspection')
   if p.get('retained') is not True or p.get('orientation_ok') is not True:raise ValueError('retention and orientation evidence required')
   if not isinstance(p.get('alignment_evidence'),str) or not p['alignment_evidence'].strip():raise ValueError('current matching-recess evidence required')
   if p.get('clear_of_contact') is not True:raise ValueError('lateral alignment requires clear of contact; recover by lifting if touching')
   target=self.point(p.get('target'))
   if abs(target[2]-pos[2])>.01 or not 0.05<=math.dist(pos[:2],target[:2])<=1:raise ValueError('align is horizontal only,0.05..1cm')
   return self.queue([move([target[0],target[1],pos[2]])],'place_check')
  if op in ('relocalize','adjust_grasp'):
   needed='approach' if op=='relocalize' else 'contact_check'
   if self.phase!=needed:raise ValueError('wrong phase for open-hand correction')
   opening=state[arm].get('opening',0)
   if type(opening) not in (int,float) or not math.isfinite(opening) or opening<(.8 if self.pregrasp_opening is None else max(.5,self.pregrasp_opening-.08)):raise ValueError('correction requires observed open hand')
   source=self.point(p.get('source'));self._pad_contact(p,source,self.preset)
   if op=='relocalize':
    if p.get('empty') is not True or not p.get('localization_evidence'):raise ValueError('fresh empty-hand localization evidence required')
    dest=[source[0],source[1],max(pos[2],source[2]+3)]
    if math.dist(pos,dest)>10:raise ValueError('relocalization exceeds10cm')
   else:
    if p.get('clear_of_contact') is not True or math.dist(pos,source)>1.5:raise ValueError('adjustment needs contact clearance and <=1.5cm')
    dest=source
   self.source=source
   return self.queue([move(dest)],needed)
  if self.phase=='contact_check' and op=='close':
   if p.get('isolated_target') is not True or p.get('pad_overlap') is not True:raise ValueError('verify target alone between middle pads before close')
   self._pad_contact(p,self.source,self.preset)
   opening=state[arm].get('opening');minimum=.8 if self.pregrasp_opening is None else max(.5,self.pregrasp_opening-.08)
   if type(opening) not in (int,float) or not math.isfinite(opening) or opening<minimum:raise ValueError('close inspection requires observed open or validated preshaped hand')
   return self.queue([f'{arm} gripper close'],'closed_check')
  if self.phase=='closed_check' and op=='lift':
   opening=state[arm].get('opening',0)
   if p.get('correct_tool') is not True or not p.get('contact_evidence'):raise ValueError('verify correct tool before lifting')
   if type(opening) not in (int,float) or not math.isfinite(opening) or opening<=.20:raise ValueError('no object contact for lift')
   return self.queue([f'{arm} move z 3'],'grasp_check')
  if op=='recover':
   if self.release_executed:raise ValueError('after release use retreat; never replay opening or grasp')
   if self.retries>=2:raise ValueError('recovery limit reached; abort')
   if not p.get('reason'):raise ValueError('recovery reason required')
   lines=p.get('commands');items,errors=commands.parse_command_list(lines)
   if errors or not 1<=len(items)<=3:raise ValueError('recovery needs1-3 valid commands')
   if any(c.arm!=arm or c.kind in ('home','homeplan','sync','done','wait') for c in items):raise ValueError('recovery controls active arm only')
   if self.tool=='tape_measure' and any(c.kind not in ('move','move3d','gripper') for c in items):raise ValueError('tape recovery preserves orientation: translation/gripper only')
   if any(c.kind=='gripper' and c.value>=.5 for c in items) and p.get('safe_support') is not True:raise ValueError('opening needs support confirmation')
   if any(c.kind=='gripper' and c.value<.5 for c in items):raise ValueError('recovery cannot bypass pre-close inspection')
   resume=p.get('resume')
   if resume not in ('approach','contact_check','grasp_check','high_check','place_check','release_check'):raise ValueError('invalid recovery resume phase')
   self.retries+=1
   return self.queue(lines,resume)
  expected={'select':'locate','approach':'grasp','grasp_check':'raise','high_check':'carry','place_check':'lower','release_check':'release','released':'confirm'}
  if op!=expected.get(self.phase):raise ValueError('wrong operation for '+self.phase+'; inspect or recover')
  if self.phase=='select':
   if used>710:raise ValueError('insufficient budget to start tool plus final reset')
   if p.get('empty_axial_flip'):raise ValueError('Empty axial flip retired after unsuccessful168step trial')
   if p.get('grasp_region')!='center':raise ValueError('Latest user requires middle-of-object grasp')
   self._grasp_balance(p)
   source=self.point(p.get('source'));target=self.point(p.get('target'));preset=p.get('preset')
   if preset not in PRESETS[self.tool]:raise ValueError('preset not allowed for tool')
   if arm=='left' and source[0]>-12 or arm=='right' and source[0]<15:raise ValueError('source must start in own separated workspace')
   self._pad_contact(p,source,preset)
   self.pregrasp_opening=None;self.source=source;self.target=target;self.preset=preset;self.carry_height=94.;self.retries=0;self.settle_checked=False;self.release_executed=False
   hover=[source[0],source[1],min(90,max(83,source[2]+6))]
   d=[b-a for a,b in zip(pos,hover)]
   if math.dist(pos,hover)>39:raise ValueError('approach outside single move range')
   lines=[f'{arm} pose3d '+ ' '.join(f'{v:.3f}' for v in d)+' '+preset]
   return self.queue(lines,'approach')
  if self.phase=='approach':
   opening=state[arm].get('opening')
   if self.pregrasp_opening is not None and (type(opening) not in (int,float) or not math.isfinite(opening) or abs(opening-self.pregrasp_opening)>.08):raise ValueError('preshape width must be observed before descent')
   self._grasp_balance(p)
   source=self.point(p.get('source',self.source))
   if math.dist(source[:2],self.source[:2])>3 or abs(source[2]-self.source[2])>2:raise ValueError('large refinement needs recovery relocalization')
   if abs(source[2]-pos[2])>8:raise ValueError('descent exceeds8cm')
   self._pad_contact(p,source,self.preset)
   self.source=source
   return self.queue([move(source)],'contact_check')
  if self.phase in ('grasp_check','high_check','place_check'):
   if p.get('retained') is not True or p.get('orientation_ok') is not True:raise ValueError('verified retention AND orientation required; recover if uncertain')
   target=self.point(p.get('target',self.target));self.target=target
   if self.phase=='place_check':
    if not isinstance(p.get('alignment_evidence'),str) or not p['alignment_evidence'].strip():raise ValueError('fresh matching-recess alignment evidence required')
    if math.dist(pos[:2],target[:2])>.2:raise ValueError('lower is vertical only; use align and inspect first')
    self._placement_check(p)
    step=p.get('max_step_cm',1.)
    if type(step) not in (int,float) or not math.isfinite(step) or not .2<=step<=1:raise ValueError('vertical precision step must be0.2..1cm')
    if target[2]>=pos[2]-.01:raise ValueError('lower needs a lower support target')
    z=max(target[2],pos[2]-step)
    return self.queue([move([pos[0],pos[1],z])],'place_check' if z>target[2]+.01 else 'release_check')
   if self.phase=='grasp_check':
    long_tool=self.tool in ('wrench','hammer')
    if 'clearance_height' in p:
     h=p['clearance_height']
     if not long_tool or type(h) not in (int,float) or not math.isfinite(h) or not 83<=h<=94 or h<pos[2]-.05:raise ValueError('long-tool clearance height must be83..94 and not descend')
     if h<94:
      self._flat_clearance(p)
     self.carry_height=float(h)
    if long_tool and self.carry_height<94:self._flat_clearance(p)
    z=min(self.carry_height,pos[2]+7) if long_tool else min(92,max(87,pos[2]+8))
    lines=[move([pos[0],pos[1],z])]
    return self.queue(lines,'grasp_check' if long_tool and z<self.carry_height-.05 else 'high_check')
   lines=[]
   if self.tool in ('wrench','hammer') and self.carry_height<94:
    self._flat_clearance(p)
    if pos[2]<self.carry_height-.3:raise ValueError('actual lift has not reached verified clearance')
   hover=[target[0],target[1],max(self.carry_height,pos[2])] if self.tool in ('wrench','hammer') else [target[0],target[1],max(84,target[2]+5)]
   requested=p.get('max_step_cm',12.)
   if type(requested) not in (int,float) or not math.isfinite(requested) or not 1<=requested<=12:raise ValueError('carry segment must be1..12cm')
   dist=math.dist(pos,hover)
   step=min(requested,12.,max(.05,dist-10.)) if dist>10.05 else min(requested,3.)
   if dist>step:
    dest=[a+(b-a)*step/dist for a,b in zip(pos,hover)]
    return self.queue([move(dest)],'high_check')
   return self.queue([move(hover)],'place_check')
  if self.phase=='release_check':
   if p.get('supported') is not True or p.get('aligned') is not True:raise ValueError('support and alignment required before release')
   if self.release_executed:raise ValueError('release already completed')
   return self.queue([f'{arm} gripper open'],'withdrawal')
  if self.phase=='released':
   if p.get('inside') is not True or p.get('aligned') is not True:raise ValueError('post-release visual check required')
   self.index+=1;self.phase='select' if self.tool else 'finish';return []
 @staticmethod
 def _placement_check(p):
  if any(p.get(k) is not True for k in ('head_in_recess','handle_in_channel','recess_floor_verified')):raise ValueError('both tool endpoints and actual recessed floor must be verified before descent')
  e=p.get('placement_evidence')
  if not isinstance(e,dict) or any(not isinstance(e.get(k),str) or not e[k].strip() for k in ('head','handle','floor')):raise ValueError('separate head, handle and recessed-floor evidence required')
 @staticmethod
 def _pad_contact(p,source,preset):
  if p.get('contact_mode')!='pad_middle':raise ValueError('latest user requires middle-pad overlap, not tip pinch')
  if not isinstance(p.get('contact_evidence'),str) or not p['contact_evidence'].strip():raise ValueError('fresh pad overlap and clearance evidence required')
  floor={'down75':78.3,'down60':77.9}.get(preset,76.5)
  if source[2]<floor:raise ValueError('oblique source below conservative finger/table clearance; inspect geometry')
 def _grasp_balance(self,p):
  if self.tool not in ('wrench','hammer'):return
  if p.get('grasp_reference')!='whole_tool':raise ValueError('grasp must reference whole-tool balance, not handle midpoint')
  if p.get('mass_bias') not in ('headward','central'):raise ValueError('infer visible mass bias toward head or central balance')
  if not isinstance(p.get('balance_evidence'),str) or not p['balance_evidence'].strip():raise ValueError('explain current mass distribution and secure balance contact')
 @staticmethod
 def _flat_clearance(p):
  if p.get('object_parallel') is not True or p.get('clearance_confirmed') is not True or not isinstance(p.get('clearance_evidence'),str) or not p['clearance_evidence'].strip():raise ValueError('lower clearance requires fresh whole-object parallel and rim/path clearance evidence')
 def queue(self,lines,next_phase):
  self.pending={'count':len(lines),'next':next_phase};return lines
 def note_execution(self,report):
  if not self.pending:return
  results=report.get('command_results') or []
  ok=len(results)==self.pending['count'] and all(r.get('ok') and not r.get('retention_warning') for r in results)
  old=self.phase
  if old=='release_check' and self.pending['next']=='withdrawal' and ok:
   arm=ARMS[self.tool];opening=((results[0].get('after') or {}).get(arm) or {}).get('opening')
   ok=type(opening) in (int,float) and math.isfinite(opening) and opening>=.8
   self.release_executed=ok
  self.phase=self.pending['next'] if ok else 'recovery'
  self.events.append({'from':old,'to':self.phase,'execution_ok':ok,'steps':report.get('steps_used')});self.pending=None
