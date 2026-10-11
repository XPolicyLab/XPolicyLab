"""Private ken.decision_icl.v1 adapter. Original archive remains immutable."""
from pathlib import Path
from dataclasses import dataclass
import json,hashlib,base64,shutil
@dataclass
class SchemaDemo:
    path:Path
    data:dict
    kind:str='ken.decision_icl.v1'
    @property
    def task(self):return self.data['task']
    @property
    def frames(self):return self.data['decisions']
    def n_images(self):return sum(len(x['observations'])+len(x.get('historical_model_input_views',[])) for x in self.frames)+len(self.data['terminal_observation']['images'])
def load_demos(cfg,task):
    assert task=='align_blocks' and int(cfg.get('demo_count',1))==1
    path=Path(cfg['demo_bank'])/task;d=json.loads((path/'demo.json').read_text())
    assert d['schema_version']=='ken.decision_icl.v1' and d['task']==task
    assert d['coverage']['expected_decisions']==d['coverage']['exported_decisions']==len(d['decisions'])
    if d['completeness']['status']!='complete' and not cfg.get('offline_review_incomplete',False):
        raise ValueError('ICL candidate is explicitly incomplete at terminal observation; retained for offline review, not silently enabled')
    for name,info in d['asset_hashes'].items():
        p=(path/name).resolve();assert p.is_relative_to(path.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
    demo=SchemaDemo(path,d)
    if cfg.get('max_images') is not None and demo.n_images()>cfg['max_images']:raise ValueError('Full ICL image limit exceeded; no sampling permitted')
    return [demo]
def text(value):return {'type':'text','text':json.dumps(value,ensure_ascii=False)}
def messages(demos):
    content=[]
    for demo in demos:
        content.append(text({k:v for k,v in demo.data.items() if k not in ('decisions','terminal_observation','asset_hashes')}))
        for decision in demo.frames:
            content.append(text(decision))
            for view in decision['observations']+decision.get('historical_model_input_views',[]):content.extend(image_parts(demo,view))
        content.append(text({'terminal_observation':demo.data['terminal_observation'],'is_decision':False}))
        for view in demo.data['terminal_observation']['images']:content.extend(image_parts(demo,view))
    return [{'role':'user','content':content},{'role':'assistant','content':'Reference context supplied. The next message is the current observation.'}]
def image_parts(demo,view):
    blob=(demo.path/view['path']).read_bytes();mime='image/png' if blob.startswith(b'\x89PNG') else 'image/jpeg'
    return [text(view),{'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(blob).decode()}}]
def save_demo(demo,out_dir):
    shutil.copytree(demo.path,out_dir,dirs_exist_ok=True)
    (Path(out_dir)/'input_construction.json').write_text(json.dumps({'schema_version':demo.kind,'decisions':len(demo.frames),'images':demo.n_images(),'actual_model_consumption':False,'reason':'construction is not proof of viewing'},indent=2))
