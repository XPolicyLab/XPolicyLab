"""Translate ModelArts topology into the unchanged NPU ZeRO-2 launcher."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
PROJECT = '/mnt/egowm_sfs/posttrain/as2_8636_ascend_20261104'

def integer_aliases(env, names, label, fallback=None):
    values = {}
    for name in names:
        value = env.get(name)
        if value in (None, ''):
            continue
        if re.fullmatch('0|[1-9][0-9]*', str(value)) is None:
            raise ValueError(name + ' must be a nonnegative integer')
        values[name] = int(value)
    if len(set(values.values())) > 1:
        raise ValueError('Conflicting ' + label + ': ' + repr(values))
    if values:
        return next(iter(values.values()))
    if fallback is not None:
        return fallback
    raise ValueError('Missing ' + label + ': ' + '/'.join(names))

def npu_visibility(env, processes):
    """Resolve logical IDs before torch_npu import; never wrap local ranks.

    Container allocation IDs supply a count only. They are not runtime indices.
    An explicit logical list handles a whole dual-die node whose platform reports
    eight physical cards while the job needs sixteen logical devices.
    """

    def ids(name):
        value = env.get(name)
        if value in (None, ''):
            return []
        parts = str(value).split(',')
        if any((re.fullmatch('[0-9]+', part.strip()) is None for part in parts)):
            raise ValueError(name + ' must contain distinct nonnegative numeric device IDs')
        numbers = [int(part.strip()) for part in parts]
        if len(set(numbers)) != len(numbers):
            raise ValueError(name + ' contains duplicate device IDs')
        return sorted(numbers)
    before = env.get('ASCEND_RT_VISIBLE_DEVICES')
    explicit = ids('CLOUDROBO_WM_NPU_DEVICES')
    runtime = ids('ASCEND_RT_VISIBLE_DEVICES')
    allocated = integer_aliases(env, ('MA_NUM_GPUS',), 'platform device allocation', 0)
    backing = max(allocated, len(ids('ASCEND_VISIBLE_DEVICES')), len(ids('NPU_VISIBLE_DEVICES')))
    expanded = False
    if explicit:
        selected, source = (explicit, 'CLOUDROBO_WM_NPU_DEVICES')
        if len(selected) != processes:
            raise ValueError('CLOUDROBO_WM_NPU_DEVICES must name exactly NPROC_PER_NODE logical devices')
    elif runtime:
        selected, source = (runtime, 'ASCEND_RT_VISIBLE_DEVICES')
        if len(selected) < processes and backing >= processes:
            selected, source, expanded = (list(range(processes)), 'allocated_container_logical_range', True)
        if len(selected) < processes:
            raise ValueError('NPROC_PER_NODE=' + str(processes) + ' but ASCEND_RT_VISIBLE_DEVICES selects only ' + str(len(selected)) + ' logical devices. For an allocated 8-card dual-die node, set CLOUDROBO_WM_NPU_DEVICES to its 16 logical IDs; workers still verify the real runtime count.')
    elif backing >= processes:
        selected, source = (list(range(processes)), 'allocated_container_logical_range')
    else:
        selected, source = ([], 'runtime_default')
    if selected:
        env['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, selected))
    after = env.get('ASCEND_RT_VISIBLE_DEVICES')
    return dict(runtime_before=before, runtime_after=after, source=source, modified=before != after, expanded=expanded, requested_processes=processes, platform_allocation_count=backing, allocation_ids_used_for_count_only=True, real_device_count_verified=False, rank_wrapping=False)

def resolve(env, code_root, *, expected_world=288):
    env = dict(env)
    obsolete = [name for name in ('PER_DEVICE_BATCH_SIZE', 'BATCH_SIZE', 'GRADIENT_ACCUMULATION_STEPS', 'EPOCHS', 'LR_PEAK', 'LR_DECAY', 'NUM_WORKERS', 'LOG_EVERY_N_STEPS', 'SAVE_EVERY_N_EPOCHS', 'SAVE_EVERY_N_STEPS', 'CONFIG_PATH', 'EXP_NAME') if env.get(name)]
    if obsolete:
        raise ValueError('Remove old recipe variables ' + ', '.join(obsolete) + '; configure CLOUDROBO_WM_* values or configs/train_fp32.json')
    for key in ('DRY_RUN', 'PREFLIGHT'):
        if env.get(key) not in (None, '', '0', 'false', 'False'):
            raise ValueError('Use FULL_TRAIN.sh dry-run instead of the old ' + key + ' variable')
    hosts = [item for item in re.split(r'[,;\s]+', env.get('VC_WORKER_HOSTS', '').strip()) if item]
    if len(set(hosts)) != len(hosts):
        raise ValueError('VC_WORKER_HOSTS contains duplicate nodes')
    nodes = integer_aliases(env, ('NNODES', 'NUM_NODES', 'VC_WORKER_NUM', 'MA_NUM_HOSTS'), 'node count', len(hosts) if hosts else None)
    if nodes < 1 or (hosts and len(hosts) != nodes):
        raise ValueError('Node count must be positive and agree with VC_WORKER_HOSTS')
    rank = integer_aliases(env, ('NODE_RANK', 'MACHINE_RANK', 'VC_TASK_INDEX', 'VK_TASK_INDEX', 'MA_TASK_INDEX'), 'node rank', 0 if nodes == 1 else None)
    if not 0 <= rank < nodes:
        raise ValueError('NODE_RANK must be in 0..NNODES-1')
    count = integer_aliases(env, ('NPROC_PER_NODE',), 'NPU processes per node', 16)
    if count < 1 or nodes * count != expected_world:
        raise ValueError('This deployment needs exactly ' + str(expected_world) + ' devices: NNODES * NPROC_PER_NODE; got ' + str(nodes * count))
    master = env.get('MASTER_ADDR') or env.get('MA_MASTER_IP') or (hosts[0] if hosts else None) or env.get('MA_CHIEF_HOST')
    if not master and nodes == 1:
        master = '127.0.0.1'
    if not master or re.search(r'[\s/,]', master) or master in ('0.0.0.0', '::') or (nodes > 1 and master in ('localhost', '127.0.0.1', '::1')):
        raise ValueError('Supply a reachable MASTER_ADDR or platform VC_WORKER_HOSTS/MA_MASTER_IP')
    port = integer_aliases(env, ('MASTER_PORT',), 'master port', 29500)
    if not 1 <= port <= 65535:
        raise ValueError('MASTER_PORT is outside 1..65535')
    if env.get('CLOUDROBO_WM_CODE_ROOT') and Path(env['CLOUDROBO_WM_CODE_ROOT']).resolve() != code_root:
        raise ValueError('CLOUDROBO_WM_CODE_ROOT differs from this OBS code package: ' + str(code_root))
    output = env.get('CLOUDROBO_WM_OUTPUT_ROOT')
    if not output:
        run = env.get('CLOUDROBO_WM_RUN_NAME') or env.get('MA_VJ_NAME') or env.get('MA_JOB_NAME')
        if not run or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', run) is None:
            raise ValueError('Set CLOUDROBO_WM_RUN_NAME to a unique simple directory name, or CLOUDROBO_WM_OUTPUT_ROOT to shared SFS')
        output = PROJECT + '/runs/' + run
    path = Path(output).expanduser()
    if '://' in output or not path.is_absolute():
        raise ValueError('CLOUDROBO_WM_OUTPUT_ROOT must be an absolute shared filesystem path')
    for linked_root in ('/mnt/egowm_sfs/cloudrobo-wam', '/mnt/cloudrobo/sfs_turbo_wam/cloudrobo-wam'):
        if path.resolve().is_relative_to(Path(linked_root).resolve()):
            raise ValueError('Checkpoint output requires an unlinked SFS directory: the OBS-linked directory does not support atomic rename')
    if str(path).startswith('/home/ma-user/modelarts/outputs'):
        raise ValueError('OBS output channels are not the shared POSIX checkpoint directory; set CLOUDROBO_WM_OUTPUT_ROOT on SFS')
    try:
        path.resolve().relative_to(code_root)
    except ValueError:
        pass
    else:
        raise ValueError('Do not write checkpoints/configuration inside the OBS code directory')
    env.update(NNODES=str(nodes), NODE_RANK=str(rank), NPROC_PER_NODE=str(count), MASTER_ADDR=master, MASTER_PORT=str(port), CLOUDROBO_WM_CODE_ROOT=str(code_root), CLOUDROBO_WM_OUTPUT_ROOT=str(path), PYTHONDONTWRITEBYTECODE='1')
    visibility = npu_visibility(env, count)
    report = dict(nodes=nodes, node_rank=rank, devices_per_node=count, world_size=nodes * count, master_addr=master, master_port=port, code_root=str(code_root), output_root=str(path), topology_variables={name: env[name] for name in ('NNODES', 'VC_WORKER_NUM', 'MA_NUM_HOSTS', 'NODE_RANK', 'VC_TASK_INDEX', 'VK_TASK_INDEX', 'MA_TASK_INDEX') if name in env}, visibility_modified=visibility['modified'], visibility=visibility)
    return (env, report)
