from pathlib import Path
import json,hashlib
from . import icl_demos,approved_open_icl
ROOT=Path(__file__).resolve().parent
def load_messages(task,max_images=None):
    if not isinstance(task,str) or task in ('.','..') or '/' in task or '\\' in task:raise ValueError('Invalid task')
    m=json.loads((ROOT/task/'manifest.json').read_text())
    if max_images is not None and m['images']>max_images:raise ValueError('ICL image budget exceeded; no silent truncation')
    for name,digest in m['files'].items():
        p=(ROOT/task/name).resolve()
        if (ROOT/task).resolve() not in p.parents:raise ValueError('Invalid path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('ICL hash mismatch: '+name)
    if m['kind']=='official_video':
        return icl_demos.demo_messages(icl_demos.load_demos({'demo_bank':str(ROOT),'demo_count':1},task))
    raw=(ROOT/task/'compact_demo.json').read_bytes();d=json.loads(raw)
    blobs=[(i,(ROOT/task/i).read_bytes()) for r in d['rounds'] for i in r['images']]
    return approved_open_icl.messages([approved_open_icl.ApprovedDemo(d,raw,blobs,{'bank':str(ROOT),'verified':True})])
