from pathlib import Path
import json,hashlib,h5py,datetime
k=Path('/ML-vePFS/model_partner/ken');p=k/'RoboDawn/evaluation/policies/vlm_agent_rsi1';c=k/'RoboDawn/task_control/pick_from_conveyor_by_image'
assert json.loads((c/'policy.json').read_text())['paused'] is True
f=p/'conveyor_eval_client.py';s=f.read_text();start=s.index('def _delete_resume_manifest(env):');end=s.index('\ndef _close_model_client',start)
old=s[start:end]
backup=c/'resume_cleanup_before_retention_20261003.txt'
if not backup.exists():backup.write_text(old)
s=s[:start]+'def _delete_resume_manifest(env):\n    """Preserve lightweight resume/success evidence under retention policy."""\n    return\n\n'+s[end:];f.write_text(s)
rule={'updated_date_asia_shanghai':'2026-10-03','paused':True,'automatic_media_cleanup_enabled':False,'protect_if':['native_success true','any native success evidence','score100 signal','unknown status','conflicting evidence','invalid recording pending investigation'],'score100_semantics':'deletion protection only; never substitutes native success admission','admission':'native success AND original native budget AND recording and loader validation','quality_tags':'never override success, trigger deletion, or add final-pose/aesthetic admission gates before target count','delete_failure_only_if':'confirmed native failure AND recording/process closed AND no success evidence AND own media AND existing authorization','redundant_success_copy':'retain unique source until final copy fully validated; deletion requires applicable authorization','history_cleanup_script':'old cleanup_old_conveyor.py terminates a specifically verified old process; not a media deletion filter; do not rerun'}
(c/'retention_policy.json').write_text(json.dumps(rule,indent=2))
root=Path('/manifold-obs/ken/open_sft_v2/pick_from_conveyor_by_image');manifest=json.loads((root/'dataset/manifest.json').read_text());rows=[]
for e in manifest['files']:
 h=root/'dataset'/e['path'];digest=hashlib.file_digest(h.open('rb'),'sha256').hexdigest()
 with h5py.File(h,'r') as a:
  attrs={v:a.attrs.get(v).item() if hasattr(a.attrs.get(v),'item') else a.attrs.get(v) for v in ('success','recording_valid','eligible_for_sft','transitions','native_step_limit')}
 rows.append({'episode':e['episode'],'path':str(h),'sha256':digest,'manifest_sha256_match':digest==e['sha256'],'attrs':attrs})
report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'pick_from_conveyor_by_image','paused':True,'gpu_started':False,'files_deleted':False,'quality_based_media_delete_filter_found':False,'private_resume_manifest_auto_unlink':'disabled; preserve lightweight evidence','success_copies':rows,'retention_policy_path':str(c/'retention_policy.json')}
assert all(r['manifest_sha256_match'] for r in rows)
(c/'retention_audit_20261003.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
