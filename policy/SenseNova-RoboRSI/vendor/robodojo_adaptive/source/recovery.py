"""Recover infrastructure interruptions of the SAME native episode, never best-of-N."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent

def failure_message(run):
 messages=[]
 for p in sorted((run/'segments').rglob('model.jsonl'),key=lambda p:p.stat().st_mtime):
  for line in p.open():
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('type')=='turn.failed':messages.append(r.get('error',{}).get('message',''))
 return messages[-1] if messages else ''

def prepare_retry(task,row):
 source=Path(row['run']) if row.get('run') else None
 if source is None or not (source/'config.json').exists():return False
 message=failure_message(source)
 if not message and row.get('infrastructure_recovery'):message=row['infrastructure_recovery']['reason']
 retry_source=source
 if not (source/'observations/latest.json').exists():
  previous_cfg=json.loads((source/'config.json').read_text())
  replay=previous_cfg.get('options',{}).get('replay')
  if replay and any(x in (row.get('summary') or {}).get('error','') for x in ['Simulator failed','Simulator exited','BrokenPipe']):
   message='Infrastructure recovery: replay reset failed before any new native action'
   retry_source=Path(replay['source_record']).parent.parent
 marker=ROOT/'COMPACTION_RECOVERY.json'
 if marker.exists() and row.get('pid') in [pid for _,pid in json.loads(marker.read_text())['targets']]:
  row['configuration_recovery']=True
  message='Configuration recovery: repeated context compression; preserve this same episode'

 image_rejection='content policy violation' in message
 # Inspection confirmed ordinary simulated robot/toy sorting images. One unchanged normal retry; never mask images or disable filters.
 if image_rejection and row.get('unchanged_image_retry_count',0)>=1:
  row['recovery_block']='Repeated provider image rejection on unchanged public robot images; manual provider review needed'
  return False
 if 'insufficient_user_quota' in message or '额度不足' in message:
  (ROOT/'MODEL_QUOTA_BLOCK.json').write_text(json.dumps(dict(reason=message),ensure_ascii=False,indent=2)+'\n')
  return False
 recoverable=bool(row.get('configuration_recovery') or row.get('infrastructure_recovery')) or image_rejection or any(x in message for x in ['429','maximum number of images','stream disconnected','timed out','Timeout','502','503','504','Infrastructure recovery: replay reset failed'])
 if not recoverable:return False
 if row.get('recovery_count',0)>=4:
  row['recovery_block']='Four infrastructure recovery attempts exhausted; retained evidence'
  return False
 failed_source=source
 source=retry_source
 records=list((source/'episodes').glob('*/result.json'))
 previous_cfg=json.loads((source/'config.json').read_text())
 if len(records)==1 and (source/'observations/latest.json').exists():
  result=json.loads(records[0].read_text())
  if result.get('native_done'):return False
  obs=json.loads((source/'observations/latest.json').read_text())
  step=obs['observation']['control_step'];limit=obs['observation']['step_limit']
 elif (row.get('configuration_recovery') or row.get('infrastructure_recovery')) and not previous_cfg['options'].get('replay') and not any(p.stat().st_size for p in (source/'episodes').glob('*/actions.jsonl')):
  step=0;limit=previous_cfg['options']['review_window_steps']
 else:return False
 if not 0<=step<limit:return False
 # Deduct the failed reset run too; never grant extra cumulative wall budget.
 if failed_source != source:
  from datetime import datetime,timezone
  failed_cfg=json.loads((failed_source/'config.json').read_text())
  began=datetime.strptime(failed_source.name.split('-')[0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
  reset_wall=max(0,(failed_source/'summary.json').stat().st_mtime-began)
  previous_cfg=dict(previous_cfg,episode_seconds=failed_cfg['episode_seconds']-reset_wall)
 segment_path=source/'segments/evaluation-0001/result.json'
 if segment_path.exists():
  segment=json.loads(segment_path.read_text())
 else:
  # SIGINT can close a segment before its final metadata write; conservatively include startup wall time.
  from datetime import datetime,timezone
  started=datetime.strptime(source.name.split('-')[0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
  ended=[p.stat().st_mtime for p in [source/'summary.json',*records] if p.exists()]
  if not ended:
   row['recovery_block']='No end-time evidence for cumulative wall budget';return False
  segment={'wall_seconds':max(0,max(ended)-started)}
 remaining=previous_cfg['episode_seconds'] if failed_source != source else previous_cfg['episode_seconds']-segment.get('wall_seconds_total',segment['wall_seconds'])
 if remaining<300:
  row['recovery_block']='Original cumulative 24h wall budget exhausted'
  return False
 cfg=json.loads((ROOT/'config'/f'{task}.json').read_text())
 cfg.update(continue_from_run=str(source),agent_workspace=str(ROOT/'workspaces'/f'{task}-recover-{time.time_ns()}'),episode_seconds=remaining)
 cfg['options'].update(codex_compact_tokens=60000,replay_fresh_session=True)
 if step:
  cfg['options']['replay']=dict(source_record=str(records[0].parent),episode_id=cfg['test_episodes'][0]['id'],control_step=step)
 else:cfg['options'].pop('replay',None)
 cfg['local_integration']['recovery']=dict(same_episode=True,source_run=str(source),replayed_control_steps=step,reason=message,fresh_model_session=True,preserve_same_episode_workspace=True,cumulative_remaining_wall_seconds=remaining)
 (ROOT/'config'/f'{task}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 history={k:v for k,v in row.items() if k!='attempt_history'}
 row.setdefault('attempt_history',[]).append(history)
 row['recovery_count']=row.get('recovery_count',0)+1
 if image_rejection:row['unchanged_image_retry_count']=row.get('unchanged_image_retry_count',0)+1
 row.pop('recovery_block',None)
 row.update(state='pending',retry_from_step=step,recovery_reason=message)
 for field in ['summary','returncode','pid','gpu','run','native_done','control_step']:row.pop(field,None)
 print('RECOVER',task,'from step',step,'attempt',row['recovery_count'],flush=True)
 return True
