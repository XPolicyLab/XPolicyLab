import hashlib
import json
from pathlib import Path
import h5py

base=Path('/ML-vePFS/model_partner/ken')
names=['135053','141108','142234','143257','144655']
root=base/'RoboDojo/collected_data/open_sft_v2/general_pickup'
manifest_path=root/'dataset/manifest.json'
manifest=json.loads(manifest_path.read_text())
rows=[]
for i,n in enumerate(names,1):
    run=base/'run_logs'/('rsi5-general_pickup-20261001-'+n)
    s=json.loads((run/'summary.json').read_text())
    feed=[json.loads(x) for x in (run/'reward_feedback.jsonl').read_text().splitlines()]
    assert feed[0]['native_step']==0 and feed[0]['current_score']==0 and feed[0]['delta_score']==0
    assert all(a['native_step']<=b['native_step'] for a,b in zip(feed,feed[1:]))
    assert feed[-1]['native_step']==s['steps']<=200
    assert all(r['remaining_steps']==200-r['native_step'] for r in feed)
    assert all(r['stage_conditions']['lift_10cm']['current'] is None for r in feed)
    assert s['capture']['success']==feed[-1]['native_success']
    assert s['layout']['layout_id']==i+2
    raw=next((run/'trajectories').glob('*.hdf5'))
    with h5py.File(raw) as f:
        assert int(f.attrs['transitions'])==s['steps']
        assert int(f.attrs['native_step_limit'])==200
        assert not any('reward' in k or 'privileged' in k for k in f)
    entries=[r for r in manifest['files'] if r['source_archive']==str(run)]
    if s['capture']['success']:
        assert len(entries)==1
        entry=entries[0]
        dest=root/'dataset'/entry['path']
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest==entry['sha256']
        loader=json.loads((run/'export/loader_validation.json').read_text())
        assert loader['tasks'][0]['first_sample_read']
        entry.update(uses_privileged_reward_feedback=True,quality=s['quality'],layout=s['layout'],
                     task_skill=s['task_skill'],diagnostics_in_sft_observation=False)
        s['delivered_episode']=entry['episode']
        s['delivered_sha256']=digest
    else:
        assert not entries and not s['capture']['eligible_for_sft']
        s['delivered_episode']=None
    s.update(attempt=i,feedback_records=len(feed),input_skill_version='v'+str(5+i),
             resulting_skill_version='v'+str(6+i),feedback_native_agreement=True,
             launcher_pid=int((run/'launcher.pid').read_text()),
             client_pid=int((run/'client.pid').read_text()),server_pid=int((run/'server.pid').read_text()),
             gpu_preflight=(run/'gpu_before.txt').read_text().strip())
    input_skill=(run/'task_skill_snapshot.md').read_bytes()
    result_source=(base/'run_logs'/('rsi5-general_pickup-20261001-'+names[i])/'task_skill_snapshot.md'
                   if i<len(names) else base/'RoboDojo/collected_data/task_skills/general_pickup/SKILL.md')
    result_skill=result_source.read_bytes()
    (run/'resulting_skill.md').write_bytes(result_skill)
    s.update(input_task_skill_file_sha256=hashlib.sha256(input_skill).hexdigest(),
             resulting_task_skill_file_sha256=hashlib.sha256(result_skill).hexdigest(),
             resulting_skill_snapshot=str(run/'resulting_skill.md'))
    rows.append(s)
assert len({r['layout']['layout_sha256'] for r in rows})==5
manifest_path.write_text(json.dumps(manifest,indent=2))
report={'batch':'rsi5_general_pickup_expansion_20261001','attempts':5,'native_successes':3,
        'new_deliveries':3,'new_episodes':['episode_0000002','episode_0000003','episode_0000004'],
        'dataset_total_episodes':len(manifest['files']),'dataset_root':str(root/'dataset'),
        'strict_native_limit':200,'all_within_native_limit':True,
        'skill_final_version':'v11','evaluation_type':'sequential_adaptation_not_held_out_benchmark',
        'uses_privileged_reward_feedback':True,'input_tokens':None,'output_tokens':None,
        'rows':rows}
out=base/'run_logs/rsi5_general_pickup_batch_20261001.json'
out.write_text(json.dumps(report,indent=2))
print(out)
