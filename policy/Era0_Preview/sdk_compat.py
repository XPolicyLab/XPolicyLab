"""Process-local, hash-checked SDK execution profiles; no SDK files are edited."""
from __future__ import annotations
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

POLICY = Path(__file__).resolve().parent
SOURCES = POLICY/'sdk_sources'


def profile_name():
    profile = os.environ.get('ERA0_EXECUTION_PROFILE','local')
    if profile not in {'local','official'}:
        raise ValueError('ERA0_EXECUTION_PROFILE must be local or official')
    return profile


def describe(root, profile=None):
    root = Path(root).resolve()
    profile = profile or profile_name()
    manifest = json.loads((SOURCES/'manifest.json').read_text())
    if profile not in manifest['profiles']: raise ValueError(f'Unknown execution profile: {profile}')
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None
              for name in manifest['host_files']}
    incompatible = [name for name,value in hashes.items() if value not in manifest['host_files'][name]]
    spec = dict(execution_profile=profile, ik_guard_enabled=profile=='local',
        ik_max_joint_delta_rad_default=0.5 if profile=='local' else None,
        random_env_limit='sdk_clutter_env_limit',
        output_paths='sdk_native_eval_result',
        sdk_disk_modified=False, sdk_host_file_sha256=hashes, unsupported_sdk_files=incompatible,
        source_manifest_sha256=hashlib.sha256((SOURCES/'manifest.json').read_bytes()).hexdigest(),
        overlay_sha256={name:row['sha256'] for name,row in manifest['profiles'][profile].items()})
    return spec


def validate(root, profile=None):
    spec = describe(root,profile)
    if spec['unsupported_sdk_files']:
        raise RuntimeError('SDK source differs from the audited official/local versions; review before running: '+
                           ', '.join(spec['unsupported_sdk_files']))
    manifest = json.loads((SOURCES/'manifest.json').read_text())
    for name,row in manifest['profiles'][spec['execution_profile']].items():
        if hashlib.sha256((POLICY/row['file']).read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Packaged SDK overlay hash mismatch: '+name)
    return spec


def source(root, name, profile=None):
    profile = profile or profile_name()
    manifest = json.loads((SOURCES/'manifest.json').read_text())
    if profile == 'official':
        return (Path(root)/name).read_bytes()
    return (POLICY/manifest['profiles'][profile][name]['file']).read_bytes()


class OverlayLoader(importlib.abc.Loader):
    def __init__(self, root, name, profile): self.root,self.name,self.profile=Path(root),name,profile
    def create_module(self,spec): return None
    def exec_module(self,module):
        module.__file__=str(self.root/self.name)
        exec(compile(source(self.root,self.name,self.profile),module.__file__,'exec'),module.__dict__)


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self,root,profile):
        self.root,self.profile=Path(root),profile
        manifest=json.loads((SOURCES/'manifest.json').read_text())
        self.modules={name[:-3].replace('/','.'):name for name in manifest['profiles'][profile] if name!='src/eval_client/main.py'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in self.modules:return None
        name=self.modules[fullname]
        return importlib.util.spec_from_file_location(fullname,self.root/name,
                    loader=OverlayLoader(self.root,name,self.profile))


def install(root,profile=None):
    spec=validate(root,profile)
    finder=OverlayFinder(root,spec['execution_profile'])
    for name in finder.modules:
        if name in sys.modules:
            raise RuntimeError('SDK overlay must be installed before module import: '+name)
    sys.meta_path.insert(0,finder)
    return spec
