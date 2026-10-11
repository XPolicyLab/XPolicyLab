"""Read-only RSI-1 approved release adapter; never rewrites demonstration content."""
from pathlib import Path
from dataclasses import dataclass
import json,hashlib,base64
RELEASE=Path(__file__).resolve().parent.parent/'icl/approved-open'
def sha(blob):return hashlib.sha256(blob).hexdigest()
@dataclass
class ApprovedDemo:
    data:dict
    raw:bytes
    blobs:list
    proof:dict
    kind:str='rsi1_approved_release'
    @property
    def task(self):return self.data['task']
    @property
    def frames(self):return [x for x in self.data['rounds'] if not x['terminal']]
    def n_images(self):return len(self.blobs)
def load_demos(cfg,task):
    if task=='stack_blocks_by_language':
        return load_stack_v2(cfg,task)
    if (RELEASE/task/'release_manifest.json').is_file():
        return load_task_release(cfg,task)
    assert int(cfg.get('demo_count',1))==1
    assert Path(cfg['demo_bank']).resolve()==RELEASE.resolve()
    manifest_bytes=(RELEASE/'release_manifest.json').read_bytes();m=json.loads(manifest_bytes)
    assert m['approved'] and m['version']=='ken-unified-icl-20261003-v1' and m['owner']=='RSI-1'
    verified={}
    for name,expected in m['files'].items():
        if not name.startswith(task+'/'):continue
        p=(RELEASE/name).resolve();assert p.is_relative_to((RELEASE/task).resolve())
        assert sha(p.read_bytes())==expected,name
        verified[name]=expected
    raw=(RELEASE/task/'compact_demo.json').read_bytes();d=json.loads(raw)
    assert d['status']=='approved' and d['task']==task and d['decision_count']==m['tasks'][task]['decisions']
    assert [x['decision'] for x in d['rounds'] if not x['terminal']]==list(range(1,d['decision_count']+1))
    assert len([x for x in d['rounds'] if x['terminal']])==1 and d['rounds'][-1]['terminal']
    blobs=[]
    for row in d['rounds']:
        assert len(row['images'])==3
        for name in row['images']:
            assert task+'/'+name in verified
            blobs.append((name,(RELEASE/task/name).read_bytes()))
    assert len(blobs)==m['tasks'][task]['images']
    if cfg.get('max_images') is not None and len(blobs)>cfg['max_images']:raise ValueError('Approved ICL exceeds image limit; no truncation allowed')
    return [ApprovedDemo(d,raw,blobs,{'version':m['version'],'owner':m['owner'],'release_manifest_sha256':sha(manifest_bytes),'compact_demo_sha256':sha(raw),'task_files_verified':verified,'decisions':d['decision_count'],'images':len(blobs)})]
def messages(demos):
    assert len(demos)==1;demo=demos[0]
    content=[{'type':'text','text':demo.raw.decode('utf-8')}]
    for name,blob in demo.blobs:
        mime='image/png' if blob.startswith(b'\x89PNG') else 'image/jpeg'
        content.extend([{'type':'text','text':name},{'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(blob).decode()}}])
    return [{'role':'user','content':content},{'role':'assistant','content':'Reference context supplied. The next message is the current observation.'}]
def save_demo(demo,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'compact_demo.json').write_bytes(demo.raw)
    for name,blob in demo.blobs:
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    proof=dict(demo.proof,model_body_byte_identical=True,constructed_input=True,actual_live_model_consumption_verified=False)
    (out/'approved_input_proof.json').write_text(json.dumps(proof,indent=2))

def load_task_release(cfg,task):
    root=RELEASE/task
    raw=(root/'compact_demo.json').read_bytes()
    manifest_raw=(root/'release_manifest.json').read_bytes()
    m=json.loads(manifest_raw);d=json.loads(raw)
    assert int(cfg.get('demo_count',1))==1 and m['approved'] and d['status']=='approved' and d['task']==task
    assert sha(raw)==m['compact_sha256'] and d['version']==m['version']
    for name,expected in m['files'].items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p.read_bytes())==expected,name
    rows=d['rounds'];n=d['decision_count']
    assert [r['decision'] for r in rows if not r.get('terminal')]==list(range(1,n+1))
    assert sum(bool(r.get('terminal')) for r in rows)==1 and rows[-1]['terminal']
    blobs=[]
    for r in rows:
        assert len(r['images'])==3
        for name in r['images']:
            assert name in m['files'];blobs.append((name,(root/name).read_bytes()))
    return [ApprovedDemo(d,raw,blobs,{'version':m['version'],'compact_demo_sha256':sha(raw),
        'release_manifest_sha256':sha(manifest_raw),'decisions':n,'images':len(blobs),'task_files_verified':m['files']})]

def load_stack_v2(cfg,task):
    bank=RELEASE.parent/'approved-stack-v2';root=bank/task
    mr=(bank/'release_manifest.json').read_bytes();m=json.loads(mr)
    assert m['version']=='ken-unified-icl-stack-371-v2'
    for name,expected in m['files'].items():
        p=(bank/name).resolve();assert p.is_relative_to(bank.resolve()) and sha(p.read_bytes())==expected
    raw=(root/'compact_demo.json').read_bytes();d=json.loads(raw)
    assert sha(raw)=='fb9e069a5116cf0402ec3ae1ada42b93b02a8b2e2a003581ecb087e09ad61896'
    assert d['status']=='released_under_ken_authorized_update' and d['task']==task and d['decision_count']==9
    assert [r['decision'] for r in d['rounds'] if not r.get('terminal')]==list(range(1,10))
    assert len(d['rounds'])==10 and d['rounds'][-1]['terminal']
    blobs=[]
    for r in d['rounds']:
        assert len(r['images'])==3
        for name in r['images']:
            assert task+'/'+name in m['files'];blobs.append((name,(root/name).read_bytes()))
    return [ApprovedDemo(d,raw,blobs,{'version':m['version'],'compact_demo_sha256':sha(raw),
        'release_manifest_sha256':sha(mr),'decisions':9,'images':30,'task_files_verified':m['files']})]
