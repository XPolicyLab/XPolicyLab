"""Select this host's GPU and dependency paths, then run the original worker."""
import json
import os
from pathlib import Path
import runpy
import sys

worker, cfg_path = sys.argv[1:3]
cfg = json.loads(Path(cfg_path).read_text())
physical = int(cfg['options']['physical_gpu'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
native = Path(cfg['environment_dir'])
sys.path[:0] = [str(native), str(native/'XPolicyLab'), str(native/'XPolicyLab/policy/Pi_05/openpi/packages/openpi-client/src')]
# Keep the installed NumPy loaded before Kit inserts bundled Python modules.
import numpy
from isaaclab.app import AppLauncher

original_resolve = AppLauncher._resolve_device_settings


def resolve(self, options):
    original_resolve(self, options)
    options['active_gpu'] = physical
    options['physics_gpu'] = 0
    current = options.get('kit_args', '')
    options['kit_args'] = current + ' --/renderer/multiGpu/enabled=false'


AppLauncher._resolve_device_settings = resolve
sys.argv = [worker, cfg_path]
runpy.run_path(worker, run_name='__main__')
