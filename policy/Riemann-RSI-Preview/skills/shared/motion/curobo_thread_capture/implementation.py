"""Scope cuRobo graph-capture error checks to its calling thread."""
import hashlib
import importlib
from pathlib import Path

GRAPH_RELATIVE = 'third_party/curobo/curobo/_src/util/cuda_graph_util.py'
GRAPH_SHA256 = '404358daccb304e8f1632653ec653a6c50b143a25313ecdeac58a3533e805447'


class _CudaView:
    def __init__(self, original):
        self.original = original
        self.capture_modes = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def graph(self, *args, **kwargs):
        # The capture function/stream/pool and graph replay remain unchanged.
        # Permit unrelated RTX threads to initialize their own CUDA resources.
        mode = kwargs.get('capture_error_mode', 'global')
        if mode == 'global': kwargs['capture_error_mode'] = 'thread_local'
        self.capture_modes.append(kwargs['capture_error_mode'])
        return self.original.graph(*args, **kwargs)


class _TorchView:
    def __init__(self, original):
        self.original = original
        self.cuda = _CudaView(original.cuda)

    def __getattr__(self, name):
        return getattr(self.original, name)


def install(official_root):
    """Adapt only one pinned module's torch binding in this child process."""
    import torch
    module = importlib.import_module('curobo._src.util.cuda_graph_util')
    path = Path(module.__file__).resolve()
    expected = Path(official_root).resolve()/GRAPH_RELATIVE
    if path != expected or hashlib.sha256(path.read_bytes()).hexdigest() != GRAPH_SHA256:
        raise ValueError('cuRobo capture module is not the pinned official source')
    if module.torch is not torch:
        raise ValueError('cuRobo torch binding already adapted')
    view = _TorchView(torch)
    module.torch = view
    return view


def report(view):
    return {'scope':'pinned_curobo_cuda_graph_util_torch_binding_only',
        'module_sha256':GRAPH_SHA256,'default_capture_error_mode':'thread_local',
        'capture_modes_observed':list(view.cuda.capture_modes),
        'global_torch_binding_changed':False,'kernels_inputs_streams_pools_unchanged':True,
        'official_source_files_changed':False,'numerical_or_performance_equivalence_claimed':False}
