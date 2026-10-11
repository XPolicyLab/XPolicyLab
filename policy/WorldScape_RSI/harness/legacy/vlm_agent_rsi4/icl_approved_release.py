"""Read-only RSI-1 approved release adapter; never rewrites demonstration content."""
from pathlib import Path
from dataclasses import dataclass
import json,hashlib,base64
RELEASE=Path('/ML-vePFS/model_partner/ken/RoboDawn/rsi_shared/icl_releases/ken-unified-icl-20261003-v1')
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
    assert task=='align_blocks' and int(cfg.get('demo_count',1))==1
    assert Path(cfg['demo_bank']).resolve()==RELEASE.resolve()
    manifest_bytes=(RELEASE/'release_manifest.json').read_bytes();m=json.loads(manifest_bytes)
    assert m['approved'] and m['version']==RELEASE.name and m['owner']=='RSI-1'
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
