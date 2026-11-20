"""Verify the existing resource contract with an explicit owning task."""
from pathlib import Path

from rsi.resource_locations import load_manifest
from harness.runtime.external_rule_assets import resource, asset_root as located_asset_root


def asset(task, name):
    return resource(Path(task), name)


def asset_root(task):
    task = Path(task)
    manifest = load_manifest(task / 'resources.json')
    for name in manifest['files']:
        resource(task, name)
    return located_asset_root(task, manifest)
