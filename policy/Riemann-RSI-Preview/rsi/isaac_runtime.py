"""Locate installed Kit extensions and private caches for the official CLI."""
import json
import os
from pathlib import Path
import sysconfig
import tempfile


def client_arguments(arguments, *, environment=None, package=None, cache=None):
    environment = os.environ if environment is None else environment
    package = Path(package) if package is not None else Path(sysconfig.get_path('purelib')) / 'isaacsim'
    folders = [package / name for name in ('exts', 'extscache', 'apps')]
    if not all(path.is_dir() for path in folders):
        raise ValueError('Use the complete installed Isaac Sim extension distribution')
    cache = Path(cache) if cache is not None else Path(tempfile.mkdtemp(prefix='riemann-isaac-'))
    if any(any(c.isspace() for c in str(path)) for path in (*folders, cache)):
        raise ValueError('The official Kit argument parser requires paths without whitespace')
    for name in ('registry', 'data', 'logs', 'documents/app', 'documents/shared', 'cache'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    options = [*('--ext-folder=' + str(path) for path in folders),
               '--portable-root', str(cache),
               '--/app/extensions/registryCache=' + str(cache / 'registry'),
               '--/app/userConfigPath=' + str(cache / 'data/user.config.json'),
               '--/app/settings/loadUserConfig=false', '--/app/settings/persistent=false',
               '--/app/tokens/app_documents=' + str(cache / 'documents/app'),
               '--/app/tokens/shared_documents=' + str(cache / 'documents/shared'),
               '--/app/tokens/omni_documents=' + str(cache / 'documents'),
               '--/app/tokens/cache=' + str(cache / 'cache'),
               '--/app/tokens/data=' + str(cache / 'data'),
               '--/log/file=' + str(cache / 'logs/kit.log')]
    physical = environment.get('RIEMANN_PHYSICAL_ENV_GPU')
    if physical is not None:
        if not physical.isdecimal():
            raise ValueError('The leased physical renderer GPU must be an explicit index')
        renderer = environment.get('RIEMANN_RENDERER_GPU_INDEX', physical)
        if not renderer.isdecimal():
            raise ValueError('The Vulkan renderer index must be resolved from the leased UUID')
        # CUDA sees the leased UUID as logical zero. Vulkan enumerates physical
        # cards independently; these are public Kit device-placement options.
        options.extend(['--/renderer/activeGpu=' + renderer,
                        '--/renderer/multiGpu/enabled=false',
                        '--/renderer/multiGpu/autoEnable=false'])
    result = list(arguments)
    existing = []
    index = 0
    while index < len(result):
        value = result[index]
        if value == '--kit_args':
            if index + 1 == len(result):
                raise ValueError('Missing --kit_args value')
            existing.extend(result[index + 1].split())
            del result[index:index + 2]
        elif value.startswith('--kit_args='):
            existing.extend(value.partition('=')[2].split())
            del result[index]
        else:
            index += 1
    result.extend(['--kit_args', ' '.join([*existing, *options])])
    print(json.dumps({'event': 'official_client_kit_locations',
                      'installed_package': str(package), 'private_cache': str(cache),
                      'physical_renderer_gpu': physical,
                      'vulkan_renderer_index': environment.get('RIEMANN_RENDERER_GPU_INDEX', physical),
                      'official_camera_physics_and_scoring_unchanged': True}), flush=True)
    return result
