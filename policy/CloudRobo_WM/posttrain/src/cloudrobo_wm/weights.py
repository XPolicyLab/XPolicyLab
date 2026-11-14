from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .platform.io import atomic_json,sha256,digest

def modules(model):
    names=tuple(dict.fromkeys(('mot','proprio_encoder',*model._dino_ckpt_keys(),*model._mode_ckpt_keys())))
    return {name:getattr(model,name) for name in names if getattr(model,name,None) is not None}

def load_tensors(model,path):
    counts={}
    with safe_open(str(path),framework='pt',device='cpu') as f:
        expected={root+'.'+name for root,module in modules(model).items() for name in module.state_dict()}
        if set(f.keys())!=expected:raise ValueError('Complete tensor key set mismatch')
        if f.metadata():raise ValueError('Tensor bundle must not carry free-text metadata')
        for root,module in modules(model).items():
            state={name:f.get_tensor(root+'.'+name) for name in module.state_dict()}
            for name,value in state.items():
                target=module.state_dict()[name]
                if value.shape!=target.shape or value.dtype!=torch.bfloat16 or target.dtype!=value.dtype:raise ValueError('Tensor identity mismatch: '+root+'.'+name)
                if not torch.isfinite(value).all():raise ValueError('Nonfinite tensor: '+root+'.'+name)
            module.load_state_dict(state,strict=True)
            for name,value in module.state_dict().items():
                if not torch.equal(value.detach().cpu(),state[name]):raise ValueError('Loaded tensor value differs: '+root+'.'+name)
            counts[root]=len(state)
    return dict(passed=True,tensors=sum(counts.values()),modules=counts,strict_values=True)

def export_tensors(model,path,contract,step):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    state={root+'.'+name:t.detach().cpu().to(torch.bfloat16).contiguous().clone() for root,module in modules(model).items() for name,t in module.state_dict().items()}
    if not all(torch.isfinite(t).all() for t in state.values()):raise FloatingPointError('Nonfinite export')
    temp=path.with_name(path.name+'.tmp');save_file(state,str(temp));temp.replace(path);path.chmod(0o777)
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if set(f.keys())!=set(state) or f.metadata():raise ValueError('Export structure changed')
        for name,value in state.items():
            if not torch.equal(f.get_tensor(name),value):raise ValueError('Export value differs: '+name)
    report=dict(version='cloudrobo_wm.tensor_export.v1',complete=True,step=step,tensors=len(state),dtype='bfloat16',bytes=path.stat().st_size,sha256=sha256(path),contract_sha256=digest(contract))
    atomic_json(str(path)+'.manifest.json',report)
    return report
