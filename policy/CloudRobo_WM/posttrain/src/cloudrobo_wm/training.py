import argparse,datetime,json,math,os,socket,sys,time,traceback,uuid
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Sampler
from .platform.io import atomic_json,digest,sha256
from .platform.backend import Backend
from .platform.ascend_runtime import initialize_runtime
from .platform.zero2 import create_engine,make_optimizer
from .model_factory import create_model,trainable_mode
from .weights import load_tensors
from .checkpoint import save,restore
from .training_metrics import METRICS_CONTRACT,reduce_training_metrics,metric_records
from .platform.initialization import InitializationFencedModule,progress

class RankBatchSampler(Sampler):
    def __init__(self,sampler,rank,world,microbatch):
        self.sampler,self.rank,self.world,self.microbatch=sampler,rank,world,microbatch
        if sampler.quantum!=world*microbatch:raise ValueError('AS2 accumulation must remain one')
    def __iter__(self):
        pending=[]
        for pair in self.sampler:
            pending.append(pair)
            if len(pending)==self.sampler.quantum:
                yield pending[self.rank*self.microbatch:(self.rank+1)*self.microbatch]
                pending=[]
        if pending:raise ValueError('Epoch padding must produce complete global batches')
    def __len__(self):
        end=(self.sampler.epoch+1)*self.sampler.samples_per_epoch
        return max(0,(end-self.sampler.committed['position'])//self.sampler.quantum)

class TrainingStep(InitializationFencedModule):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,sample):return self.model(sample)

def validate_config(cfg):
    t,o,s,r=cfg['topology'],cfg['optimizer'],cfg['schedule'],cfg['runtime']
    if t!={'nodes':18,'local_world_size':16,'world_size':288,'microbatch':12,'gradient_accumulation_steps':1,'global_batch':3456}:raise ValueError('This delivery is fixed to the reviewed 18-node topology')
    if not r['deterministic'] or r['seed']!=42 or r['loss_skip'] or r['data_replacement']:raise ValueError('Determinism and strict data/error behavior are fixed')
    expected=(563328,163,5542,277)
    if s['epochs']!=34 or tuple(s[k] for k in ('epoch_samples','updates_per_epoch','total_updates','warmup_updates'))!=expected:raise ValueError('34-epoch schedule contract changed')
    if o['stage']!=2 or o['communication_dtype'] not in ('fp32','bf16') or any(o[k]!='fp32' for k in ('gradient_accumulation_dtype','master_weight_dtype','state_dtype')):raise ValueError('Unexpected optimizer precision')
    return cfg

def build_dataset(config,root):
    from .datasets.multisource.history import HistoryRichPromptDataset
    v=json.loads(Path(config).read_text());root=Path(root)
    v['manifests']=[str(root/p) for p in v['manifests']];v['stats_path']=str(root/v['stats_path'])
    for key in ('instruction_overlay','prompt_augmentation'):v[key]['root']=str(root/v[key]['root'])
    return HistoryRichPromptDataset(**v)

def worker_init(worker):
    from .utils.pytorch_utils import worker_init_function
    worker_init_function(worker)
    torch.set_num_threads(1)

def build_scheduler(optimizer,total=5542,warmup=277,learning_rate=3e-4):
    main=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(total-warmup,1),eta_min=learning_rate*0.01)
    if warmup<=0:return main
    ramp=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0/warmup,end_factor=1.0,total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[ramp,main],milestones=[warmup])

def run(args):
    cfg=validate_config(json.loads(Path(args.config).read_text()));root=Path(__file__).resolve().parents[2]
    output=Path(args.output);rank=int(os.environ['RANK']);nonce=str(uuid.uuid4())
    os.umask(0)
    os.environ['CLOUDROBO_WM_DATA_ROOT']=str(Path(args.data).resolve())
    os.environ['DS_ACCELERATOR']='npu';os.environ['CLOUDROBO_WM_REAL_ROPE']='1';os.environ['HCCL_DETERMINISTIC']='true'
    os.environ.setdefault('HCCL_INTER_HCCS_DISABLE','TRUE')
    runtime=initialize_runtime(dict(backend='npu',distributed_backend='zero2',storage='shared',output_root=str(output),runtime_timeout_seconds=7200,runtime_control_timeout_seconds=1800,as2=cfg))
    backend=Backend('npu',int(os.environ['LOCAL_RANK']));backend.cpu_group=runtime.cpu_group
    torch.use_deterministic_algorithms(True,warn_only=False)
    backend.seed(42+rank);torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    output.mkdir(parents=True,exist_ok=True);output.chmod(0o777)
    logs=output/'ranks';logs.mkdir(exist_ok=True);logs.chmod(0o777)
    atomic_json(logs/f'rank{rank:06d}.STARTED.json',dict(rank=rank,world=dist.get_world_size(),host=socket.gethostname(),nonce=nonce,deterministic=torch.are_deterministic_algorithms_enabled(),communication=cfg['optimizer']['communication_dtype']))
    progress(output,'building_dataset')
    dataset=build_dataset(root/'configs/data.json',args.data)
    if len(dataset)!=560900 or len(dataset.manifests)!=1 or len(dataset.manifests[0]['episodes'])!=2500:raise ValueError('Selected Clean training population differs')
    sampler=dataset.make_sampler(seed=42,batch_size=12,num_processes=288,accumulation_steps=1)
    if sampler.samples_per_epoch!=563328:raise ValueError('Epoch padding changed')
    model_config=json.loads((root/'configs/model.json').read_text())
    progress(output,'building_model')
    model=create_model(model_config,args.assets,device=backend.device)
    progress(output,'loading_initialization_tensors')
    model.bind_multisource_contract(dataset.checkpoint_contract)
    initialization=load_tensors(model,args.weights)
    model=trainable_mode(model)
    progress(output,'initialization_tensors_loaded')
    options=dict(learning_rate=3e-4,betas=(0.9,0.95),weight_decay=0.01,max_grad_norm=1.0,communication_dtype=cfg['optimizer']['communication_dtype'],output_root=str(output),gradient_order_path=str(root/'configs/gradient_order.json'),trace_gradient_order=bool(args.audit))
    optimizer=make_optimizer(model,options,backend)
    adapter=create_engine(TrainingStep(model),optimizer,options,backend,12,1,288)
    atomic_json(logs/f'rank{rank:06d}.STARTUP.json',dict(rank=rank,world=288,process_nonce=nonce,
        platform=runtime.kind,communication_backend=runtime.environment['communication'],
        framework=runtime.environment['framework'],optimizer=adapter.contract,seed=42+rank,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        data_loader=dict(workers=cfg['runtime']['num_workers_per_rank'],prefetch_factor=cfg['runtime']['prefetch_factor'],start_method='spawn')))
    scheduler=build_scheduler(optimizer)
    # Paths can differ between source and target mounts. The contract binds semantics.
    data_contract=dataset.checkpoint_contract.copy()
    for k in ('training_instruction_overlay','training_prompt_augmentation'):
        if k in data_contract:data_contract[k]={key:v for key,v in data_contract[k].items() if key not in ('root','text_cache')}
    contract=dict(protocol='cloudrobo_wm.as2_training.v1',configuration=cfg,model_config=model_config,data=data_contract,optimizer=adapter.contract,initialization_manifest=json.loads(Path(str(args.weights)+'.manifest.json').read_text()),code_manifest=sha256(root/'FILES_SHA256.json'),metrics=METRICS_CONTRACT)
    records=runtime.gather_objects(digest(contract))
    if len(set(records))!=1:raise ValueError('Cross-rank code/model/data/precision contracts differ')
    if rank==0:atomic_json(output/'RUN_CONTRACT.json',contract)
    step=0
    if args.resume:
        step,result=restore(args.resume,adapter,scheduler,sampler,backend,contract,nonce,audit=args.audit)
        atomic_json(logs/f'rank{rank:06d}.RESTORED.json',result)
    import gc
    if cfg['runtime'].get('gc_freeze',False):
        gc.collect();gc.freeze();gc.enable()
    atomic_json(logs/f'rank{rank:06d}.GC.json',dict(enabled=gc.isenabled(),freeze_count=gc.get_freeze_count(),requested=cfg['runtime'].get('gc_freeze',False)))
    sampler.set_epoch_offset(0)
    epoch=step//163
    sampler.set_epoch(epoch)
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(dataset,batch_sampler=RankBatchSampler(sampler,rank,288,12),num_workers=cfg['runtime']['num_workers_per_rank'],worker_init_fn=worker_init,persistent_workers=True,prefetch_factor=cfg["runtime"]["prefetch_factor"],pin_memory=True,generator=generator,multiprocessing_context="spawn")
    target=min(args.stop_at or 5542,5542)
    metrics=(output/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    first=True
    while step<target:
        sampler.set_epoch(step//163)
        for sample in loader:
            started=time.monotonic()
            indices=torch.stack((sample['window_index'],sample['sample_position']),dim=1).to(backend.device)
            actual=[torch.empty_like(indices) for _ in range(288)];dist.all_gather(actual,indices)
            observed=[tuple(pair) for values in actual for pair in values.cpu().tolist()]
            if first:
                atomic_json(logs/f'rank{rank:06d}.NEXT_INPUT.json',dict(step=step+1,global_cursor=sampler.committed['position'],pairs=indices.cpu().tolist(),process_nonce=nonce));first=False
            adapter.begin_update(1)
            with backend.autocast():loss,values=adapter(sample)
            finite=torch.isfinite(loss.detach()).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise FloatingPointError('Nonfinite training loss; no optimizer/cursor advance')
            adapter.backward(loss,is_last=True);norm=adapter.step()
            scheduler.step();sampler.commit(3456,observed=observed);step+=1
            backend.synchronize();seconds=time.monotonic()-started
            global_loss,global_values,global_norm=reduce_training_metrics(loss,values,norm)
            metadata=dict(step=step,epoch=step/163,rank=rank,world=288,process_nonce=nonce,seconds=seconds,learning_rate=optimizer.param_groups[0]['lr'],consumed_windows=sampler.committed['position'],skipped_updates=0,replacements=0)
            scalar,global_scalar=metric_records(loss,values,norm,global_loss,global_values,global_norm,metadata)
            atomic_json(logs/f'rank{rank:06d}.STATUS.json',scalar)
            if metrics:
                atomic_json(output/'GLOBAL_STATUS.json',global_scalar)
                metrics.write(json.dumps(global_scalar,allow_nan=False)+'\n')
            if step%1000==0 or step==target:
                meta=save(output/'checkpoints'/f'step{step:06d}.zero2',adapter,scheduler,sampler,backend,contract,step,nonce,audit=args.audit)
                atomic_json(logs/f'rank{rank:06d}.SAVED.json',meta)
            if step>=target:break
    dist.barrier()
    if rank==0:atomic_json(output/'FINISHED.json',dict(complete=True,step=step,target=target,full_training=step==5542,world=288,epochs=34,consumed_windows=sampler.committed['position']))
    if metrics:metrics.close()
    runtime.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--assets',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--stop-at',type=int);p.add_argument('--audit',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        out=Path(a.output)/'ranks';rank=int(os.environ.get('RANK','0'))
        atomic_json(out/f'rank{rank:06d}.FAILED.json',dict(error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
if __name__=='__main__':main()
