"""Read-only acceptance of fresh-process, complete-state resume evidence."""
import json,math,itertools
from pathlib import Path
from .platform.io import digest

def read(path):
    return json.loads(Path(path).read_text())

def verify_global_telemetry(root,statuses,expected_steps,world):
    from .training_metrics import METRICS_PROTOCOL,IDENTITY_FIELDS,RESERVED_METRIC_NAMES
    root=Path(root)
    rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines() if line]
    if not rows or [row['step'] for row in rows]!=list(expected_steps):
        raise ValueError('Global telemetry step stream differs')
    latest=read(root/'GLOBAL_STATUS.json')
    if latest!=rows[-1]:raise ValueError('Global status and metrics journal differ')
    if len(statuses)!=world or sorted(row['rank'] for row in statuses)!=list(range(world)):
        raise ValueError('Global telemetry requires all distinct rank statuses')
    for row in rows:
        names=row.get('metric_names',[])
        if (row.get('metrics_protocol')!=METRICS_PROTOCOL or row.get('metrics_scope')!='global_rank_mean'
                or row.get('rank')!=0 or row.get('world')!=world
                or not names or names!=sorted(set(names)) or RESERVED_METRIC_NAMES.intersection(names)
                or set(row)-IDENTITY_FIELDS-{'loss','gradnorm'}!=set(names)
                or names!=latest['metric_names']
                or any(not math.isfinite(float(row[key])) for key in ['loss','gradnorm',*names])):
            raise ValueError('Global telemetry schema or scope differs')
    for status in statuses:
        if (status.get('metrics_protocol')!=METRICS_PROTOCOL
                or status.get('metrics_scope')!='rank_local_losses_with_model_global_diagnostics'
                or status.get('step')!=latest['step'] or status.get('world')!=world
                or set(status)-IDENTITY_FIELDS-{'loss','gradnorm'}!=set(latest['metric_names'])):
            raise ValueError('Per-rank telemetry schema, step or scope differs')
    for key in ['loss','gradnorm',*latest['metric_names']]:
        expected=math.fsum(float(status[key]) for status in statuses)/world
        if not math.isclose(float(latest[key]),expected,rel_tol=2e-5,abs_tol=2e-7):
            raise ValueError('Global telemetry does not equal the rank mean: '+key)
    return latest

def verify_smoke(output,binding):
    from .training_metrics import METRICS_PROTOCOL,METRICS_CONTRACT
    output=Path(output);contracts=[];phases={}
    for phase,step in (('save3',3),('resume6',6)):
        root=output/phase;nodes=[]
        for node in range(18):
            claim=read(output/'launch'/phase/f'node{node:03d}.CLAIM.json')
            done=read(output/'launch'/phase/f'node{node:03d}.DONE.json')
            if (claim['node']!=node or done['node']!=node or done['returncode']!=0
                    or claim['binding']!=binding or done['claim_nonce']!=claim['nonce']):
                raise ValueError(f'Node exit/identity mismatch: {phase}/{node}')
            nodes.append(claim)
        if len({r['host'] for r in nodes})!=18:raise ValueError('18 distinct node hosts are required')
        contract=read(root/'RUN_CONTRACT.json');contracts.append(contract)
        checkpoint=root/'checkpoints'/f'step{step:06d}.zero2'
        receipt=read(checkpoint/'RECEIPT.json')
        export=read(root/'checkpoints'/f'step{step:06d}.weights.safetensors.manifest.json')
        finished=read(root/'FINISHED.json')
        if (not receipt['complete'] or receipt['world']!=288 or receipt['step']!=step
                or receipt['ranks']!=list(range(288)) or receipt['contract_sha256']!=digest(contract)
                or not export['complete'] or export['step']!=step or export['contract_sha256']!=digest(contract)
                or not finished['complete'] or finished['step']!=step or finished['consumed_windows']!=step*3456):
            raise ValueError('Checkpoint/export/cursor acceptance failed: '+phase)
        if contract.get('metrics')!=METRICS_CONTRACT:raise ValueError('Global telemetry contract differs')
        history=contract['model_config'].get('history')
        if history:
            from .models.history_attention import HistoryConfig
            if contract['data'].get('training_history')!=HistoryConfig.parse(history).contract():
                raise ValueError('Observed history model/data contracts differ')
            if (export.get('dtype')!='bfloat16' or export.get('tensors')!=2509
                    or export.get('bytes')!=(root/'checkpoints'/f'step{step:06d}.weights.safetensors').stat().st_size):
                raise ValueError('Complete history tensor export differs')
        if len(receipt['files'])!=865 or len({row['path'] for row in receipt['files']})!=865:
            raise ValueError('Complete 288-rank checkpoint must contain 865 distinct files')
        for row in receipt['files']:
            relative=Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts or row['bytes']<=0:
                raise ValueError('Invalid checkpoint manifest member')
            path=checkpoint/relative
            if not path.is_file() or path.stat().st_size!=row['bytes']:
                raise ValueError('Checkpoint member size differs: '+str(path))
        saved=[];statuses=[]
        for rank in range(288):
            logs=root/'ranks';prefix=f'rank{rank:06d}'
            started=read(logs/(prefix+'.STARTED.json'));status=read(logs/(prefix+'.STATUS.json'))
            startup=read(logs/(prefix+'.STARTUP.json'))
            record=read(checkpoint/(prefix+'.json'))
            saved_log=read(logs/(prefix+'.SAVED.json'))
            if (started['rank']!=rank or started['world']!=288 or not started['deterministic']
                    or started['communication']!=binding['communication']
                    or started['host']!=nodes[rank//16]['host'] or saved_log!=record
                    or startup['rank']!=rank or startup['world']!=288 or startup['process_nonce']!=started['nonce']
                    or startup['platform']!='npu' or startup['communication_backend']!='hccl'
                    or startup['optimizer']!=contract['optimizer'] or not startup['deterministic'] or startup['seed']!=42+rank
                    or status['step']!=step or status['skipped_updates']!=0 or status['replacements']!=0
                    or status['process_nonce']!=started['nonce'] or status['rank']!=rank
                    or any(not math.isfinite(float(status[key])) for key in ('loss','gradnorm','seconds','learning_rate'))
                    or record['rank']!=rank or record['world']!=288 or record['step']!=step
                    or record['process_nonce']!=started['nonce'] or record['contract_sha256']!=digest(contract)
                    or set(record.get('fingerprints',{}))!={'model','master','optimizer','scheduler','rng','sampler'}
                    or (logs/(prefix+'.FAILED.json')).exists()):
                raise ValueError(f'Rank status/checkpoint mismatch: {phase}/{rank}')
            saved.append(record)
            statuses.append(status)
        expected_steps=list(range(1,4)) if phase=='save3' else list(range(4,7))
        verify_global_telemetry(root,statuses,expected_steps,288)
        phases[phase]=saved
    if contracts[0]!=contracts[1]:raise ValueError('Save/resume run contracts differ')
    if contracts[0]['configuration']['runtime']['gc_freeze']!=binding['gc_freeze']:raise ValueError('GC binding differs')
    for rank,old in enumerate(phases['save3']):
        prefix=f'rank{rank:06d}';logs=output/'resume6/ranks'
        restored=read(logs/(prefix+'.RESTORED.json'));next_input=read(logs/(prefix+'.NEXT_INPUT.json'))
        if (not restored['passed'] or restored['rank']!=rank or restored['step']!=3
                or restored['previous_process_nonce']!=old['process_nonce'] or restored['process_nonce']==old['process_nonce']
                or restored['fingerprints']!=old['fingerprints'] or next_input['step']!=4
                or next_input['global_cursor']!=10368 or next_input['process_nonce']!=restored['process_nonce']):
            raise ValueError(f'Fresh complete-state resume failed: rank {rank}')
    pairs=[pair for rank in range(288) for pair in read(output/'resume6/ranks'/f'rank{rank:06d}.NEXT_INPUT.json')['pairs']]
    if sorted(p[1] for p in pairs)!=list(range(10368,13824)):raise ValueError('Next input positions are duplicated or missing')
    from .datasets.multisource.sampling import CoverageSampler
    state=phases['save3'][0]['sampler'];cfg=state['contract']
    sampler=CoverageSampler(cfg['sizes'],weights=cfg['weights'],samples_per_epoch=cfg['samples_per_epoch'],
        seed=cfg['seed'],batch_size=cfg['batch_size'],num_processes=cfg['world_size'],
        accumulation_steps=cfg['accumulation_steps'],block_size=cfg['block_size'],manifest_hash=cfg['manifest_hash'])
    sampler.load_state_dict(state);sampler.set_epoch_offset(0);sampler.set_epoch(0)
    expected=list(itertools.islice(sampler,3456))
    if [tuple(p) for p in sorted(pairs,key=lambda p:p[1])]!=expected:raise ValueError('Next window indices differ from the saved global stream')
    return dict(protocol='cloudrobo_wm.as2_smoke_acceptance.v1',passed=True,binding=binding,world=288,nodes=18,
                saved_step=3,resumed_through_step=6,fingerprints=['model','master','optimizer','scheduler','rng','sampler'],
                fresh_processes=288,next_input_cursor=10368,contract_sha256=digest(contracts[0]),
                global_telemetry_verified=True,metrics_protocol=METRICS_PROTOCOL,
                inference_interface_with_actual_npu_export='pending_separate_acceptance')
