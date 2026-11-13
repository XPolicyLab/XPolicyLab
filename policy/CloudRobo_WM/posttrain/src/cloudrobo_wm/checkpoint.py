from pathlib import Path
import os,uuid,json,hashlib
import torch
import torch.distributed as dist
from .platform.io import atomic_json,digest,sha256
from .weights import export_tensors

PROTOCOL='cloudrobo_wm.as2_full_resume.v1'

def tensor_hash(value):
    h=hashlib.sha256()
    flat=value.detach().reshape(-1)
    for part in flat.split(1048576):h.update(part.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def state_fingerprints(adapter,scheduler,sampler,rng):
    def visit(value):
        if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_hash(value))
        if isinstance(value,dict):return {str(k):visit(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [visit(v) for v in value]
        try:
            import numpy as np
            if isinstance(value,np.ndarray):return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        except ImportError:pass
        return value
    return {name:digest(visit(value)) for name,value in dict(model=adapter.model.state_dict(),master=adapter.engine.optimizer.single_partition_of_fp32_groups,optimizer=adapter.optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng,sampler=sampler.state_dict()).items()}

def save(path,adapter,scheduler,sampler,backend,contract,step,nonce,*,audit=False):
    rank=dist.get_rank();world=dist.get_world_size();path=Path(path)
    if rank==0:
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True);path.chmod(0o777)
    dist.barrier()
    rng=backend.rng()
    meta=dict(protocol=PROTOCOL,rank=rank,world=world,step=step,process_nonce=nonce,contract_sha256=digest(contract),scheduler=scheduler.state_dict(),sampler=sampler.state_dict())
    if audit:meta['fingerprints']=state_fingerprints(adapter,scheduler,sampler,rng)
    state=path/f'rank{rank:06d}.rng.pt';torch.save(rng,state);state.chmod(0o777)
    # Native ZeRO state includes the FP32 master partitions and Adam moments.
    ok=adapter.engine.save_checkpoint(str(path/'native'),tag='state',client_state={'contract_sha256':digest(contract),'step':step},save_latest=False)
    if not ok:raise RuntimeError('Native full checkpoint save failed')
    atomic_json(path/f'rank{rank:06d}.json',meta)
    dist.barrier()
    if rank==0:
        records=[json.loads((path/f'rank{i:06d}.json').read_text()) for i in range(world)]
        if any(r['rank']!=i or r['world']!=world or r['step']!=step or r['contract_sha256']!=digest(contract) for i,r in enumerate(records)):raise ValueError('Checkpoint rank binding mismatch')
        # The native helper is unnecessary for native restoration. Keep only state.
        helper=path/'native/zero_to_fp32.py'
        if helper.exists():helper.unlink()
        files=[dict(path=str(p.relative_to(path)),bytes=p.stat().st_size) for p in path.rglob('*') if p.is_file()]
        names={r['path'] for r in files}
        expected={f'native/state/bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt' for i in range(world)}
        expected|={f'rank{i:06d}.rng.pt' for i in range(world)}
        expected|={f'rank{i:06d}.json' for i in range(world)}
        expected.add('native/state/mp_rank_00_model_states.pt')
        if not expected<=names or any(r['bytes']<=0 for r in files):raise ValueError('Native checkpoint lacks complete rank/model/optimizer/RNG state')
        receipt=dict(protocol=PROTOCOL,complete=True,step=step,world=world,contract_sha256=digest(contract),ranks=list(range(world)),files=files)
        atomic_json(path/'RECEIPT.json',receipt)
        export_tensors(adapter.model,path.parent/f'step{step:06d}.weights.safetensors',contract,step)
        for p in path.rglob('*'):p.chmod(0o777)
    dist.barrier()
    backend.set_rng(rng)
    return meta

@torch.no_grad()
def restore(path,adapter,scheduler,sampler,backend,contract,nonce,*,audit=False):
    path=Path(path);rank=dist.get_rank()
    receipt=json.loads((path/'RECEIPT.json').read_text());meta=json.loads((path/f'rank{rank:06d}.json').read_text())
    if receipt['protocol']!=PROTOCOL or not receipt['complete'] or receipt['world']!=dist.get_world_size() or receipt['contract_sha256']!=digest(contract):raise ValueError('Full resume contract mismatch')
    if meta['rank']!=rank or meta['world']!=dist.get_world_size() or meta['step']!=receipt['step'] or meta['contract_sha256']!=digest(contract):raise ValueError('Rank restore identity mismatch')
    for row in receipt['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid checkpoint inventory path')
        target=path/relative
        if not target.is_file() or target.stat().st_size!=row['bytes']:raise ValueError('Checkpoint file missing/truncated: '+str(relative))
    if meta['process_nonce']==nonce:raise ValueError('Resume verification requires fresh processes')
    rng=torch.load(path/f'rank{rank:06d}.rng.pt',map_location='cpu',weights_only=False);backend.validate_rng_state(rng)
    loaded,client=adapter.engine.load_checkpoint(str(path/'native'),tag='state',load_module_strict=True,load_optimizer_states=True,load_lr_scheduler_states=False)
    # Native load_checkpoint also returns framework metadata. Bind our fields
    # explicitly; strict module loading and all six fingerprints remain required.
    expected_client={'contract_sha256':digest(contract),'step':receipt['step']}
    observed_client={key:client.get(key) for key in expected_client} if isinstance(client,dict) else None
    if not loaded or observed_client!=expected_client:raise ValueError('Native resume client binding differs: '+repr(observed_client))
    if client.get('ds_version')!=adapter.contract['deepspeed']:raise ValueError('Native checkpoint framework version differs')
    scheduler.load_state_dict(meta['scheduler']);sampler.load_state_dict(meta['sampler']);backend.set_rng(rng)
    for master in adapter.engine.optimizer.single_partition_of_fp32_groups:
        if master.dtype!=torch.float32:raise ValueError('Master partition is not FP32 after resume')
    for state in adapter.optimizer.state.values():
        for key in ('exp_avg','exp_avg_sq'):
            if key in state and state[key].dtype!=torch.float32:raise ValueError('Adam moment is not FP32')
    result=dict(passed=True,rank=rank,step=receipt['step'],fresh_process=True,client_binding=observed_client,previous_process_nonce=meta['process_nonce'],process_nonce=nonce)
    if audit:
        after=state_fingerprints(adapter,scheduler,sampler,backend.rng())
        if after!=meta['fingerprints']:raise ValueError('Model/master/Adam/scheduler/RNG/cursor fingerprint mismatch')
        result['fingerprints']=after
    return receipt['step'],result
