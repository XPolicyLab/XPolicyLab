"""Ascend launch and distributed admission; independent of data/model logic.

Importing this module does not import torch or initialize an accelerator. The
separate CPU control store exchanges failures *before* HCCL initialization, so
one rank's missing torch_npu/device does not leave its peers in HCCL startup.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import tempfile
import time
from typing import Any
RUNTIME_VERSION = 'cloudrobo_wm.ascend.runtime.v1'

def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a positive integer')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be a positive integer') from error
    if str(parsed) != str(value) or parsed < 1:
        raise ValueError(name + ' must be a positive integer')
    return parsed

def _timeout(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(name + ' must be positive and finite')
    return datetime.timedelta(seconds=float(value))

def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Topology:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    node_rank: int
    master_addr: str
    master_port: int
    control_port: int

    @property
    def nodes(self):
        return self.world_size // self.local_world_size

def topology_from_environment(cfg, environment=None):
    """Validate torchrun's logical ranks, without interpreting physical NPU IDs."""
    env = os.environ if environment is None else environment
    world = _positive(env.get('WORLD_SIZE', '1'), 'WORLD_SIZE')
    local_world = _positive(env.get('LOCAL_WORLD_SIZE', env.get('NPROC_PER_NODE', '1')), 'LOCAL_WORLD_SIZE')
    rank = int(env.get('RANK', '0'))
    local = int(env.get('LOCAL_RANK', '0'))
    if not 0 <= rank < world or not 0 <= local < local_world:
        raise ValueError('RANK/LOCAL_RANK is outside its allocated world')
    if world % local_world or rank % local_world != local:
        raise ValueError('RANK, LOCAL_RANK and LOCAL_WORLD_SIZE describe inconsistent node groups')
    node = rank // local_world
    for key in ('NODE_RANK', 'GROUP_RANK'):
        if key in env and int(env[key]) != node:
            raise ValueError(key + " disagrees with torchrun's RANK/LOCAL_WORLD_SIZE")
    for key in ('NNODES', 'NUM_NODES'):
        if key in env and int(env[key]) != world // local_world:
            raise ValueError(key + ' disagrees with WORLD_SIZE/LOCAL_WORLD_SIZE')
    if world > 1 and any((key not in env for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'))):
        raise ValueError('Distributed training requires a complete torchrun environment')
    addr = str(env.get('MASTER_ADDR', '127.0.0.1'))
    if not addr or re.search('[\\s/,]', addr) or addr in ('0.0.0.0', '::'):
        raise ValueError('MASTER_ADDR must identify one reachable chief host')
    if world > local_world and addr in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Multi-node MASTER_ADDR cannot be loopback')
    port = _positive(env.get('MASTER_PORT', '29500'), 'MASTER_PORT')
    control = _positive(cfg.get('runtime_control_port', port + 1), 'runtime_control_port')
    if port > 65535 or control > 65535 or control == port:
        raise ValueError('Training/control ports must be distinct and in 1..65535')
    return Topology(rank, world, local, local_world, node, addr, port, control)

def validate_settings(cfg):
    if cfg.get('backend', 'npu') != 'npu' or cfg.get('distributed_backend', 'zero2') != 'zero2':
        raise ValueError('Only NPU + ZeRO-2 is supported by this training entry')
    storage = cfg.get('storage', 'shared')
    if storage not in ('shared', 'node_local'):
        raise ValueError('storage must be shared or node_local')
    if cfg.get('recipe', 'complete_once') != 'complete_once':
        raise ValueError('AS2 requires the coverage sampler')
    timeout = _timeout(cfg.get('runtime_timeout_seconds', 1800), 'runtime_timeout_seconds')
    control = _timeout(cfg.get('runtime_control_timeout_seconds', 180), 'runtime_control_timeout_seconds')
    return ('npu', 'zero2', storage, timeout, control)

def _release(version):
    match = re.match('^(\\d+)\\.(\\d+)\\.(\\d+)', str(version))
    return tuple(map(int, match.groups())) if match else None

def inspect_framework(torch, kind, local_rank, *, npu_module=None):
    """Required APIs and numerical torch/torch_npu releases; no model execution."""
    raw = {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torch_npu': None, 'packages': {}, 'warnings': []}
    for name in ('deepspeed', 'deepspeed-npu', 'torchvision', 'av'):
        try:
            raw['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raw['packages'][name] = None
    if raw['packages']['deepspeed-npu']:
        raw['warnings'].append('Legacy deepspeed-npu is installed; do not assume it is needed by modern native DeepSpeed NPU support')
    if npu_module is None:
        raise RuntimeError('torch_npu must be imported before NPU admission')
    raw['torch_npu'] = str(getattr(npu_module, '__version__', 'unknown'))
    left, right = (_release(raw['torch']), _release(raw['torch_npu']))
    if left is None or right is None or left != right:
        raise RuntimeError('torch/torch_npu numerical releases differ or cannot be parsed: ' + repr((raw['torch'], raw['torch_npu'])))
    paths = ['distributed.is_available', 'distributed.is_gloo_available', 'distributed.init_process_group', 'distributed.new_group', 'distributed.destroy_process_group', 'distributed.all_gather_object', 'distributed.all_reduce', 'distributed.all_gather_into_tensor', 'distributed.reduce_scatter_tensor', 'autocast', 'nn.functional.scaled_dot_product_attention', 'optim.AdamW']
    paths.extend((kind + '.' + name for name in ('is_available', 'device_count', 'set_device', 'synchronize', 'max_memory_allocated')))
    rng_apis = ('get_rng_state', 'set_rng_state', 'manual_seed')
    paths.extend((kind + '.' + name for name in rng_apis))
    for path in paths:
        current = torch
        for component in path.split('.'):
            current = getattr(current, component, None)
            if current is None:
                break
        if not callable(current):
            raise RuntimeError('Required framework API is missing: torch.' + path)
    if not torch.distributed.is_available():
        raise RuntimeError('torch.distributed is unavailable')
    if not torch.distributed.is_gloo_available():
        raise RuntimeError('Gloo is required for CPU metadata and checkpoint admission')
    api = getattr(torch, kind)
    if not api.is_available():
        raise RuntimeError('Requested ' + kind + ' accelerator is unavailable')
    count = int(api.device_count())
    if not 0 <= local_rank < count:
        raise RuntimeError('LOCAL_RANK ' + str(local_rank) + ' is outside the runtime-visible ' + kind + ' range 0..' + str(count - 1))
    api.set_device(local_rank)
    raw['visible_devices'] = count
    if callable(getattr(api, 'get_device_name', None)):
        raw['device_name'] = str(api.get_device_name(local_rank))
    raw['required_apis_present'] = True
    return raw

def _configuration_signature(cfg):
    local_keys = {'node_id', 'output_root', 'runtime_report_root', '_ascend_launch', 'asset_root', 'init_weights', 'resume', 'model_config', 'deepspeed_config'}
    if cfg.get('storage', 'shared') == 'node_local':
        local_keys.add('data_root')
    normalized = {key: value for key, value in cfg.items() if key not in local_keys}
    for key in ('model_config', 'deepspeed_config'):
        if cfg.get(key):
            normalized[key + '_sha256'] = hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest()
    return _fingerprint(normalized)

def validate_rank_records(records, topology):
    """CPU-only cross-rank agreement. Return all errors instead of rank-local exits."""
    errors = []
    if len(records) != topology.world_size:
        return ['Control plane did not collect the expected number of ranks']
    ranks = [record.get('rank') for record in records]
    if sorted(ranks) != list(range(topology.world_size)):
        errors.append('Control plane rank IDs are incomplete or duplicated')
    for record in records:
        if record.get('error'):
            errors.append('rank ' + str(record.get('rank')) + ': ' + str(record['error']))
    for key in ('world_size', 'local_world_size', 'kind', 'distributed_backend', 'storage', 'configuration'):
        if len({_fingerprint(record.get(key)) for record in records}) != 1:
            errors.append('Cross-rank ' + key + ' differs')
    frameworks = [r.get('framework', {}) for r in records if not r.get('error')]
    for key in ('torch', 'torch_npu', 'python'):
        if len({f.get(key) for f in frameworks}) > 1:
            errors.append('Cross-rank framework ' + key + ' differs')
    for key in ('deepspeed',):
        if len({f.get('packages', {}).get(key) for f in frameworks}) > 1:
            errors.append('Cross-rank package ' + key + ' differs')
    for node in range(topology.nodes):
        local = [r for r in records if r.get('node_rank') == node]
        if sorted((r.get('local_rank', -1) for r in local)) != list(range(topology.local_world_size)):
            errors.append('Node ' + str(node) + ' has inconsistent local rank coverage')
        if len({r.get('hostname') for r in local}) != 1:
            errors.append('Logical node ' + str(node) + ' spans multiple hosts')
    return errors

@dataclass
class RuntimeContext:
    kind: str
    device: Any
    topology: Topology
    storage: str
    environment: dict
    cpu_group: Any = None
    _torch: Any = field(default=None, repr=False)
    _control_store: Any = field(default=None, repr=False)
    _exchange_number: int = field(default=0, repr=False)
    _owned_process_group: bool = field(default=False, repr=False)

    @property
    def rank(self):
        return self.topology.rank

    @property
    def world_size(self):
        return self.topology.world_size

    @property
    def local_rank(self):
        return self.topology.local_rank

    @property
    def local_world_size(self):
        return self.topology.local_world_size

    @property
    def node_rank(self):
        return self.topology.node_rank

    def gather_objects(self, value):
        """CPU metadata only. The HCCL default group never receives Python objects."""
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        self._torch.distributed.all_gather_object(values, value, group=self.cpu_group)
        return values

    def agree_or_raise(self, error=None, stage='runtime'):
        records = self.gather_objects({'rank': self.rank, 'error': None if error is None else str(error)})
        failed = [r for r in records if r['error']]
        if failed:
            raise RuntimeError(stage + ' failed on ' + '; '.join(('rank ' + str(r['rank']) + ': ' + r['error'] for r in failed)))

    def close(self):
        if self._owned_process_group and self._torch.distributed.is_initialized():
            self._torch.distributed.destroy_process_group()
            self._owned_process_group = False

def initialize_runtime(cfg, *, torch_module=None):
    """Admit all ranks on CPU, initialize env:// HCCL/Gloo, then CPU metadata group.

    Caller owns context.close(). For single-process calls, no
    process group is needed. torchrun/ZeRO-2 calls initialize even at world=1.
    """
    kind, distributed, storage, timeout, control_timeout = validate_settings(cfg)
    topology = topology_from_environment(cfg)
    torch = torch_module if torch_module is not None else importlib.import_module('torch')
    dist = torch.distributed
    if dist.is_initialized():
        raise RuntimeError('initialize_runtime must run before any process-group/DeepSpeed initialization')
    framework, local_error = ({}, None)
    try:
        npu_module = importlib.import_module('torch_npu')
        framework = inspect_framework(torch, kind, topology.local_rank, npu_module=npu_module)
        if kind == 'npu' and hasattr(dist, 'is_hccl_available') and (not dist.is_hccl_available()):
            raise RuntimeError('HCCL is unavailable after importing torch_npu')
    except Exception as error:
        local_error = type(error).__name__ + ': ' + str(error)
    try:
        configuration = _configuration_signature(cfg)
    except Exception as error:
        configuration = None
        local_error = (local_error + '; ' if local_error else '') + type(error).__name__ + ': ' + str(error)
    record = dict(rank=topology.rank, world_size=topology.world_size, local_rank=topology.local_rank, local_world_size=topology.local_world_size, node_rank=topology.node_rank, hostname=socket.gethostname(), kind=kind, distributed_backend=distributed, storage=storage, configuration=configuration, framework=framework, error=local_error)
    control_store = None
    if topology.world_size > 1:
        try:
            control_store = dist.TCPStore(topology.master_addr, topology.control_port, topology.world_size, topology.rank == 0, timeout=control_timeout, wait_for_workers=True)
            control_store.set('admission/rank/' + str(topology.rank), json.dumps(record))
            keys = ['admission/rank/' + str(rank) for rank in range(topology.world_size)]
            control_store.wait(keys, control_timeout)
            records = [json.loads(control_store.get(key).decode()) for key in keys]
        except Exception as error:
            raise RuntimeError('CPU admission timed out/failed before HCCL; verify all nodes/ranks, MASTER_ADDR and runtime_control_port=' + str(topology.control_port) + ': ' + str(error)) from error
    else:
        records = [record]
    errors = validate_rank_records(records, topology)
    if errors:
        raise RuntimeError('Distributed admission rejected before accelerator collectives:\n' + '\n'.join(errors))
    communication = 'hccl'
    initialized = topology.world_size > 1 or 'RANK' in os.environ or distributed == 'zero2' or (kind == 'npu')
    cpu_group = None
    if initialized:
        for key, value in (('RANK', topology.rank), ('WORLD_SIZE', topology.world_size), ('MASTER_ADDR', topology.master_addr), ('MASTER_PORT', topology.master_port)):
            os.environ.setdefault(key, str(value))
        dist.init_process_group(backend=communication, init_method='env://', timeout=timeout, rank=topology.rank, world_size=topology.world_size)
        cpu_group = dist.new_group(ranks=list(range(topology.world_size)), backend='gloo', timeout=timeout)
    environment = dict(version=RUNTIME_VERSION, platform=kind, framework=framework, communication=communication, metadata_communication='gloo', topology=vars(topology), admission=records, distributed_backend=distributed, storage=storage, process_group_initialized=initialized, timeout_seconds=timeout.total_seconds(), control_timeout_seconds=control_timeout.total_seconds(), production_npu_hccl_verified=False)
    device = torch.device(kind + ':' + str(topology.local_rank))
    return RuntimeContext(kind, device, topology, storage, environment, cpu_group, torch, control_store, _owned_process_group=initialized)

def _launch_topology(env, cfg=None):
    if any((key in env for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'OMPI_COMM_WORLD_LOCAL_RANK'))):
        raise ValueError('train_npu_zero2.sh must run once per node; do not wrap it in torchrun/accelerate/mpirun')
    nodes = _positive(env.get('NNODES', '1'), 'NNODES')
    processes = _positive(env.get('NPROC_PER_NODE', '8'), 'NPROC_PER_NODE')
    node = int(env.get('NODE_RANK', '0'))
    if not 0 <= node < nodes:
        raise ValueError('NODE_RANK must be in 0..NNODES-1')
    if nodes > 1 and (not env.get('MASTER_ADDR')):
        raise ValueError('Set the same reachable MASTER_ADDR on every node')
    master = env.get('MASTER_ADDR', '127.0.0.1')
    port = _positive(env.get('MASTER_PORT', '29500'), 'MASTER_PORT')
    topology_from_environment(cfg or {}, dict(RANK=str(node * processes), WORLD_SIZE=str(nodes * processes), LOCAL_RANK='0', LOCAL_WORLD_SIZE=str(processes), NODE_RANK=str(node), MASTER_ADDR=master, MASTER_PORT=str(port)))
    return (nodes, node, processes, master, port)

def launch_main(arguments=None):
    """Resolve a JSON platform config and exec torchrun; no training imports here."""
    parser = argparse.ArgumentParser(description='One-node-one-launch Ascend portable pretraining')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Write auditable platform config and print command; do not launch')
    parser.add_argument('--check-only', action='store_true', help='Run communication/operator preflight instead of training')
    parser.add_argument('--transport-diagnostic', choices=('roce','hccs'), help='Communication diagnostic only; never trains or grants training admission')
    args = parser.parse_args(arguments)
    if args.transport_diagnostic and args.check_only:
        raise ValueError('Choose a transport diagnostic or the approved training preflight')
    if args.transport_diagnostic:
        os.environ['HCCL_INTER_HCCS_DISABLE'] = 'TRUE' if args.transport_diagnostic == 'roce' else 'FALSE'
    source = args.config.resolve()
    source_bytes = source.read_bytes()
    cfg = json.loads(source_bytes)
    if not isinstance(cfg, dict):
        raise ValueError('Training config must be a JSON object')
    cfg['backend'] = cfg.get('backend', 'npu')
    cfg['distributed_backend'] = cfg.get('distributed_backend', 'zero2')
    validate_settings(cfg)
    nodes, node, processes, master, port = _launch_topology(os.environ, cfg)
    topology_from_environment(cfg, dict(RANK=str(node * processes), WORLD_SIZE=str(nodes * processes), LOCAL_RANK='0', LOCAL_WORLD_SIZE=str(processes), MASTER_ADDR=master, MASTER_PORT=str(port), NODE_RANK=str(node)))
    if 'output_root' not in cfg:
        raise ValueError('Set output_root in the training JSON to record launch configuration')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns())
    audit = Path(cfg['output_root']).expanduser().resolve() / 'platform' / ('launch-' + stamp + '-node' + str(node))
    audit.mkdir(parents=True, exist_ok=False)
    cfg['_ascend_launch'] = dict(version=RUNTIME_VERSION, source_config=str(source), source_sha256=hashlib.sha256(source_bytes).hexdigest(), node_rank=node, nodes=nodes, devices_per_node=processes)
    resolved = audit / 'runtime_config.json'
    _atomic_json(resolved, cfg)
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes', str(nodes), '--node_rank', str(node), '--nproc_per_node', str(processes), '--master_addr', master, '--master_port', str(port), '--rdzv_backend', 'static', '--max_restarts', '0']
    command += ['--log-dir', str(audit / 'worker-logs'), '--redirects', '3', '--tee', '3']
    if args.transport_diagnostic:
        command += [str(root / 'scripts/check_hccl_transport.py'), '--config', str(resolved), '--transport', args.transport_diagnostic, '--report-root', str(audit / 'transport-diagnostic')]
    elif args.check_only:
        command += [str(root / 'scripts/check_ascend_distributed.py'), '--config', str(resolved), '--report-root', str(audit / 'distributed-preflight')]
    else:
        command += ['--module', 'cloudrobo_wm.portable.train', '--config', str(resolved)]
    receipt = dict(version=RUNTIME_VERSION, command=command, source_config=str(source), resolved_config=str(resolved), dry_run=args.dry_run, check_only=args.check_only, transport_diagnostic=args.transport_diagnostic, scope='platform admission only; no data/model logic overrides', source_sha256=hashlib.sha256(source_bytes).hexdigest())
    _atomic_json(audit / 'LAUNCH.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print(shlex.join(command), flush=True)
        return receipt
    os.execv(sys.executable, command)
if __name__ == '__main__':
    launch_main()
