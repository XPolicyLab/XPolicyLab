"""Private full-decision ICL. No sampling, resizing or silent truncation."""
from pathlib import Path
from dataclasses import dataclass
import json,hashlib,base64,shutil
@dataclass
class FullDemo:
    path:Path
    data:dict
    kind:str='rsi4_full_decision_v2'
    @property
    def frames(self):return self.data['rounds']
    @property
    def task(self):return self.data['task']
    def n_images(self):return sum(len(x['images']) for x in self.frames)+len(self.data['terminal']['images'])
def load_demos(cfg,task):
    assert task=='align_blocks' and int(cfg.get('demo_count',1))==1
    path=Path(cfg['demo_bank'])/task;data=json.loads((path/'demo.json').read_text())
    assert data['kind']=='rsi4_full_decision_v2' and data['schema_version']==2 and data['native_success']
    assert data['decision_count']==data['exported_decision_count']==len(data['rounds'])
    assert [x['decision_id'] for x in data['rounds']]==list(range(1,data['decision_count']+1))
    assert not data['coverage']['missing_required_decision_fields']
    for name,meta in data['assets'].items():
        p=(path/name).resolve();assert p.is_relative_to(path.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],f'ICL asset corrupted: {name}'
    demo=FullDemo(path,data)
    if cfg.get('max_images') is not None and demo.n_images()>cfg['max_images']:raise ValueError('Full ICL exceeds configured image limit; report limitation, do not sample')
    print(json.dumps({'icl_full_v2':True,'decision_count':len(demo.frames),'image_count':demo.n_images(),'resized':False,'truncated':False,'actual_model_consumption':'unverified until desktop display receipts'}),flush=True)
    return [demo]
def text(value):return {'type':'text','text':value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)}
def messages(demos):
    content=[]
    for demo in demos:
        content.append(text({'kind':demo.kind,'source_run':demo.data['source_run'],'instruction':demo.data['instruction'],'coverage':demo.data['coverage'],'warning':'All real decisions, not selected phases. Copy strategy, never coordinates. A terminal observation is not a new decision. Original RGB and marked model-input image are two renderings of each same camera.'}))
        for entry in demo.frames:
            content.append(text({'decision_id':entry['decision_id'],'control_step':entry['control_step'],'end_control_step':entry['end_control_step'],'request_id':entry['request_id']}))
            content.append(text((demo.path/entry['record']).read_text()))
            content.append(text((demo.path/entry['control_trace']).read_text()))
            source_messages=json.loads((demo.path/entry['transport']/'messages.json').read_text())
            content.append(text({'original_additional_diagnostic_message':source_messages[-1]}))
            for frame in entry['images']:content.extend(image_parts(demo,frame))
        content.append(text((demo.path/'terminal/record.json').read_text()))
        for frame in demo.data['terminal']['images']:content.extend(image_parts(demo,frame))
    content.append(text('End of full-decision reference. The next live observation is a different scene.'))
    return [{'role':'user','content':content}, {'role':'assistant','content':'Reference context supplied. The next message is the current observation.'}]
def image_parts(demo,frame):
    blob=(demo.path/frame['path']).read_bytes();mime='image/png' if blob.startswith(b'\x89PNG') else 'image/jpeg'
    return [text(frame),{'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(blob).decode()}}]
def save_demo(demo,out_dir):
    shutil.copytree(demo.path,out_dir,dirs_exist_ok=True)
    proof={'kind':demo.kind,'constructed_decisions':len(demo.frames),'constructed_images':demo.n_images(),'all_assets_verified_before_load':True,'actual_model_consumption_verified':False,'reason':'Saving/constructing messages is not evidence that a desktop model viewed the content'}
    (Path(out_dir)/'construction_coverage.json').write_text(json.dumps(proof,indent=2))
