"""Approved compact conveyor ICL, full ordered rounds, no legacy fallback."""
import base64, hashlib, json
from pathlib import Path
TASK='pick_from_conveyor_by_image'
VERSION='ken-unified-conveyor-20261005-v1'
COMPACT_SHA='5922fc2d9d6bbaacca64f44683b5fe690c4b605249ca09c6a3a69a546f23a22b'
BANK=Path('/ML-vePFS/model_partner/ken/RoboDawn/demos/ken_unified_conveyor_20261005_v1')/TASK

def load_messages(root=BANK,max_images=None,max_content_bytes=None):
 root=Path(root).resolve()
 manifest_bytes=(root/'release_manifest.json').read_bytes();m=json.loads(manifest_bytes)
 if not (m.get('approved') is True and m.get('task')==TASK and m.get('version')==VERSION and m.get('decision_count')==14 and m.get('terminal_images')==3 and m.get('compact_sha256')==COMPACT_SHA):raise ValueError('Unapproved or mismatched compact release')
 blobs={}
 for rel,digest in m['files'].items():
  f=(root/rel).resolve()
  try:f.relative_to(root)
  except ValueError:raise ValueError('Release path escaped bank')
  b=f.read_bytes()
  if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Release hash mismatch: '+rel)
  if rel=='compact_demo.json' or rel.startswith('decision_') or rel.startswith('terminal_'):blobs[rel]=b
 raw=blobs['compact_demo.json']
 if hashlib.sha256(raw).hexdigest()!=COMPACT_SHA:raise ValueError('Pinned compact digest mismatch')
 d=json.loads(raw)
 if not (d.get('schema')=='ken.compact_icl.preview.v1' and d.get('status')=='approved' and d.get('task')==TASK and d.get('decision_count')==14):raise ValueError('Compact schema mismatch')
 rounds=d['rounds']
 if len(rounds)!=15 or [x.get('decision') for x in rounds]!=list(range(1,15))+[None]:raise ValueError('Missing or reordered rounds')
 header='APPROVED TASK DEMONSTRATION. Coordinates and target refer only to this example; re-localize the live scene.\nINSTRUCTION: '+d['instruction']+'\nUSAGE NOTES: '+json.dumps(d.get('notes',[]),ensure_ascii=False)
 msgs=[{'role':'user','content':[{'type':'text','text':header}]}];total=0;size=len(header.encode());coverage=[];seen=set()
 for i,r in enumerate(rounds,1):
  terminal=i==15
  if r.get('terminal') is not terminal:raise ValueError('Terminal boundary invalid')
  expected=[(('terminal_'+cam+'.png') if terminal else ('decision_%04d_'%i+cam+'.jpg')) for cam in ('cam_head','cam_left_wrist','cam_right_wrist')]
  if r['images']!=expected:raise ValueError('Camera count/order invalid')
  if terminal:body='TERMINAL OBSERVATION OF THE DEMONSTRATION (not an extra decision). Native success at501/700steps.'
  else:
   if any(k not in r for k in ('robot_state','plan','commands','effect')) or not isinstance(r['commands'],list):raise ValueError('Missing current round fields')
   body='DEMONSTRATION DECISION '+str(i)+'\nCURRENT ROBOT STATE:\n'+r['robot_state']+'\nPLAN:\n'+r['plan']+'\nCOMMANDS:\n'+json.dumps(r['commands'],ensure_ascii=False)+'\nOBSERVED EFFECT:\n'+r['effect']
   if 'YOUR NOTES FROM PREVIOUS TURNS' in body or 'HISTORY OF YOUR COMMANDS' in body:raise ValueError('Recursive prompt history detected')
  parts=[{'type':'text','text':body}];size+=len(body.encode())
  for rel in expected:
   if rel in seen:raise ValueError('Duplicate current image')
   seen.add(rel);b=blobs[rel];size+=len(b);total+=1
   parts.append({'type':'text','text':rel})
   parts.append({'type':'image_url','image_url':{'url':'data:'+('image/png' if rel.endswith('.png') else 'image/jpeg')+';base64,'+base64.b64encode(b).decode(),'detail':'high'}})
  msgs.append({'role':'user','content':parts});coverage.append({'decision':None if terminal else i,'terminal':terminal,'images':expected})
 if total!=45:raise ValueError('Full45image coverage required')
 if max_images is not None and total>max_images:raise ValueError('Configured image limit cannot fit complete ICL; do not truncate')
 if max_content_bytes is not None and size>max_content_bytes:raise ValueError('Configured byte limit cannot fit complete ICL; do not truncate')
 proof={'status':'constructed_not_proof_of_model_display','version':VERSION,'task':TASK,'bank':str(root),'compact_sha256':COMPACT_SHA,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'verified_files':len(m['files']),'decisions':14,'terminal_observations':1,'images':total,'bytes_before_base64':size,'coverage':coverage,'recursive_history_included':False,'source_provenance_loaded_into_prompt':False,'loader_truncation':False,'legacy_fallback':False,'actual_desktop_display_coverage':'record separately after displaying every image'}
 return msgs,proof
