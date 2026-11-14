"""Fixed-topology launch with distinct save/resume process groups."""
import argparse,json,os,socket,subprocess,sys,time,uuid
from pathlib import Path
from topology import resolve

def wait_records(folder,suffix,validate,timeout=7200):
    deadline=time.monotonic()+timeout
    while True:
        records=[]
        for node in range(18):
            path=folder/f'node{node:03d}.{suffix}.json'
            if path.is_file():
                try:record=json.loads(path.read_text())
                except json.JSONDecodeError:continue
                validate(node,record);records.append(record)
        if len(records)==18:return records
        if time.monotonic()>=deadline:raise TimeoutError('Waiting for all 18 nodes: '+str(folder)+'/'+suffix)
        time.sleep(2)

def exclusive(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    path.chmod(0o777)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','smoke','dry-run'],nargs='?',default='train');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    from cloudrobo_wm.platform.io import atomic_json,sha256
    from cloudrobo_wm.acceptance import verify_smoke
    env,topology=resolve(os.environ,root,expected_world=288)
    comm=env.get('CLOUDROBO_WM_COMM_DTYPE','fp32');gc=env.get('CLOUDROBO_WM_GC_FREEZE','0')
    if comm not in ('fp32','bf16') or gc not in ('0','1'):raise ValueError('Use communication fp32|bf16 and GC freeze 0|1 explicitly')
    required=['CLOUDROBO_WM_DATA_ROOT','CLOUDROBO_WM_ASSET_ROOT','CLOUDROBO_WM_INIT_WEIGHTS','CLOUDROBO_WM_OUTPUT_ROOT']
    if any(not env.get(k) for k in required):raise ValueError('Set all data/asset/weight/output paths explicitly')
    env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',DS_ACCELERATOR='npu',CLOUDROBO_WM_REAL_ROPE='1',HCCL_DETERMINISTIC='true',HCCL_INTER_HCCS_DISABLE='TRUE',OMP_NUM_THREADS=env.get('OMP_NUM_THREADS','4'))
    env.setdefault('CLOUDROBO_WM_NPU_DEVICES',','.join(map(str,range(16))))
    config=root/f'configs/train_{comm}{"_gc" if gc=="1" else ""}.json'
    output=Path(env['CLOUDROBO_WM_OUTPUT_ROOT']);node=topology['node_rank'];port=topology['master_port']
    if port>65514:raise ValueError('Reserve ports base, base+1, base+20 and base+21')
    base=[sys.executable,'-m','torch.distributed.run','--nnodes','18','--nproc-per-node','16','--node-rank',str(node),'--master-addr',topology['master_addr'],'--max-restarts','0']
    training=['--module','cloudrobo_wm.training','--config',str(config),'--data',env['CLOUDROBO_WM_DATA_ROOT'],'--assets',env['CLOUDROBO_WM_ASSET_ROOT'],'--weights',env['CLOUDROBO_WM_INIT_WEIGHTS']]
    if args.mode=='dry-run':
        print(json.dumps(dict(topology=topology,communication=comm,gc_freeze=gc=='1',workers_per_rank=16,prefetch_factor=2,epochs=34,total_updates=5542,warmup=277,command=base+['--master-port',str(port)]+training+['--output',str(output)]),indent=2));return
    manifest=root/'FILES_SHA256.json'
    if not manifest.is_file():raise FileNotFoundError('Delivery has not been sealed: FILES_SHA256.json')
    for row in json.loads(manifest.read_text())['files']:
        if sha256(root/row['path'])!=row['sha256']:raise ValueError('Delivered code changed: '+row['path'])
    ready=Path(env['CLOUDROBO_WM_DATA_ROOT']).parent/'READY.json';ready_record=json.loads(ready.read_text())
    if not ready_record['complete'] or not ready_record['target_sha256_verified'] or not ready_record['permissions_0777_verified']:raise ValueError('Verified target data/assets are required')
    weights_manifest=json.loads(Path(env['CLOUDROBO_WM_INIT_WEIGHTS']+'.manifest.json').read_text())
    if not weights_manifest['complete']:raise ValueError('Initialization export is incomplete')
    binding=dict(code_manifest_sha256=sha256(manifest),configuration_sha256=sha256(config),initialization_sha256=weights_manifest['sha256'],data_manifest_sha256=ready_record['data_manifest_sha256'],communication=comm,gc_freeze=gc=='1',world=288)
    if args.mode=='train':
        acceptance=json.loads(Path(env['CLOUDROBO_WM_SMOKE_REPORT']).read_text())
        if not acceptance['passed'] or acceptance['binding']!=binding:raise ValueError('Matching complete 18-node smoke acceptance is required')
    output.mkdir(parents=True,exist_ok=True);output.chmod(0o777)
    def phase(name,phase_port,extra):
        folder=output/'launch'/name;nonce=str(uuid.uuid4())
        claim=dict(node=node,host=socket.gethostname(),nonce=nonce,binding=binding,master_port=phase_port,control_port=phase_port+1)
        exclusive(folder/f'node{node:03d}.CLAIM.json',claim)
        def check_claim(index,record):
            if record['node']!=index or record['binding']!=binding or record['master_port']!=phase_port:raise ValueError('Cross-node launch binding differs')
        wait_records(folder,'CLAIM',check_claim)
        phase_env=dict(env,MASTER_PORT=str(phase_port))
        try:code=subprocess.run(base+['--master-port',str(phase_port)]+training+extra,env=phase_env).returncode
        except BaseException as error:
            atomic_json(folder/f'node{node:03d}.DONE.json',dict(node=node,claim_nonce=nonce,returncode=-1,error=repr(error)));raise
        atomic_json(folder/f'node{node:03d}.DONE.json',dict(node=node,claim_nonce=nonce,returncode=code))
        def check_done(index,record):
            peer=json.loads((folder/f'node{index:03d}.CLAIM.json').read_text())
            if record['node']!=index or record['claim_nonce']!=peer['nonce'] or record['returncode']!=0:raise RuntimeError('Node subprocess failed: '+str(index))
        wait_records(folder,'DONE',check_done)
    if args.mode=='smoke':
        first=output/'save3';second=output/'resume6'
        phase('save3',port,['--output',str(first),'--stop-at','3','--audit'])
        phase('resume6',port+20,['--output',str(second),'--resume',str(first/'checkpoints/step000003.zero2'),'--stop-at','6','--audit'])
        report=verify_smoke(output,binding)
        if node==0:atomic_json(output/'SMOKE_REPORT.json',report)
    else:
        resume=env.get('CLOUDROBO_WM_RESUME')
        phase('train',port,['--output',str(output)]+(['--resume',resume] if resume else []))

if __name__=='__main__':main()
