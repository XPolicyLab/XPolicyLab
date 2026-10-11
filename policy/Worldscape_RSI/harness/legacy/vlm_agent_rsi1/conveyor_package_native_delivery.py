"""Publish budget-compliant captures into separate preview and training folders."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import re
import h5py
import fcntl

base = Path('/ML-vePFS/model_partner/ken/RoboDojo/collected_data')
task = sys.argv[1]
episode = sys.argv[2]
capture_episode = episode
archive = Path(sys.argv[3]).resolve()
assert task.replace('_','').isalnum() and episode.startswith('episode_') and episode.replace('_','').isalnum()
assert archive.parent == Path('/ML-vePFS/model_partner/ken/RoboDawn/task_control/pick_from_conveyor_by_image/runs').resolve()
source_root = Path(sys.argv[4]) if len(sys.argv)>4 else base/'robodawn_open_sft'
source = source_root/task/'arx_x5/data'/f'{episode}.hdf5'
with h5py.File(source,'r') as f:
    assert f.attrs['success'] and f.attrs['recording_valid'] and f.attrs['eligible_for_sft']
    assert f.attrs['within_native_budget'] and 0 < f.attrs['transitions'] <= f.attrs['native_step_limit'] < 10**9
    steps = int(f.attrs['transitions'])
    limit = int(f.attrs['native_step_limit'])

root = base/'open_sft_v2'
task_root = root/task
assert task == 'pick_from_conveyor_by_image'
assert task_root.resolve() == Path('/manifold-obs/ken/open_sft_v2')/task
dataset = task_root/'dataset'
naming_lock = (archive.parent.parent/'delivery_naming.lock').open('a')
fcntl.flock(naming_lock, fcntl.LOCK_EX)
manifest_path = dataset/'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}
rows = manifest['files']
assert all(re.fullmatch(r'episode_[0-9]{7}', row['episode']) for row in rows), 'Migrate existing delivery names before publishing'
existing = [row for row in rows if row.get('source_episode', row['episode']) == capture_episode]
assert len(existing) <= 1
if existing:
    assert existing[0]['source_archive'] == str(archive)
    episode = existing[0]['episode']
else:
    used = {int(row['episode'][8:]) for row in rows}
    assert used == set(range(len(rows))), 'Non-contiguous delivery index; inspect before allocating'
    assert len(rows) < 50, 'Delivery quota reached'
    episode = f'episode_{len(rows):07d}'
preview = task_root/'preview'/episode
target = dataset/task/'arx_x5/data'/f'{episode}.hdf5'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def copy_checked(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        assert sha(src) == sha(dst), f'Refusing overwrite: {dst}'
    else:
        shutil.copy2(src,dst)
    assert sha(src) == sha(dst)

copy_checked(source,target)
evaluation = archive/'evaluation'
if not evaluation.is_dir():
    match=re.search(r'Video is saved to `([^`]+)`',(archive/'console.log').read_text(errors='replace'))
    assert match
    evaluation=(base.parent/match[1]).parent
for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
    source_episode = sys.argv[5] if len(sys.argv)>5 else episode
    assert source_episode.startswith('episode_') and source_episode[8:].isdigit()
    src=evaluation/f'{source_episode}_{camera}_success.mp4'
    copy_checked(src,preview/f'{camera}.mp4')
for src,name in ((archive/'capture_validation.json','capture_validation.json'),
                 (evaluation/'_result.json','native_result.json')):
    copy_checked(src,preview/name)
result = json.loads((archive/'summary.json').read_text())
result.update(episode=episode, source_episode=capture_episode)
result_path = preview/'result.json'
if result_path.exists():
    assert json.loads(result_path.read_text()) == result, 'Refusing inconsistent result overwrite'
else:
    result_path.write_text(json.dumps(result, indent=2))
validation = json.loads((source_root/'loader_validation.json').read_text())
selected = [row for row in validation['tasks'] if row['task']==task]
assert selected and selected[0]['first_sample_read']
(dataset/'source_loader_validation.json').write_text(json.dumps({'tasks':selected,
    'note':'Source dataset was loader-tested; delivered HDF5 is byte-identical.'},indent=2))
manifest_path=dataset/'manifest.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files':[]}
entry={'task':task,'episode':episode,'source_episode':capture_episode,'path':str(target.relative_to(dataset)),
       'steps':steps,'native_step_limit':limit,'native_success':True,'sha256':sha(target),
       'source_archive':str(archive),'preview':str(preview.relative_to(task_root)),
       'selection':'native-budget gate passed; single-layout candidate, not a reliability claim'}
manifest['files']=[r for r in manifest['files'] if r['episode']!=episode]+[entry]
manifest['episodes']=len(manifest['files'])
manifest_temp=manifest_path.with_suffix('.json.tmp')
manifest_temp.write_text(json.dumps(manifest,indent=2))
manifest_temp.replace(manifest_path)
(preview/'说明.txt').write_text(
    f'任务：{task}\n结果：环境原生成功，{steps}/{limit} 步。\n'
    f'仿真执行：{steps/25:.2f} 秒；视频含初始帧，约 {(steps+1)/25:.2f} 秒。\n'
    'cam_head.mp4：主视角；cam_left_wrist.mp4：左腕；cam_right_wrist.mp4：右腕。\n'
    'result.json：决策次数、耗时和来源；native_result.json：环境评分。\n'
    '模型等待时间不计入视频。通过步数门槛不代表跨场景成功率已验证。\n',encoding='utf-8')
print(json.dumps({'root':str(root),'task':str(task_root),'dataset':str(dataset),
                  'preview':str(preview),'episode':episode,'source_episode':capture_episode,'verified_sha256':sha(target)},indent=2))
