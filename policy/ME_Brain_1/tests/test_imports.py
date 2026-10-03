"""Check adapter source resolution and runtime options."""

import ast
import os
import sys
from pathlib import Path


from XPolicyLab.policy.ME_Brain_1.replan import replan_steps_from_env

ROOT = Path(__file__).parents[1]


def test_policy_uses_release_environment():
    assert replan_steps_from_env({'FOCUS_VLWA_REPLAN_STEPS': '8'}) == 8
    assert replan_steps_from_env({}) == 25


def source_configurator():
    # Compile only the path resolver so these tests do not require XPolicyLab or a GPU.
    path = ROOT / 'model.py'
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_configure_focus_vlwa_import')
    namespace = {'os': os, 'sys': sys, 'Path': Path, 'Any': object,
                 '_POLICY_DIR': path.parent, '__package__': 'XPolicyLab.policy.ME_Brain_1'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_configure_focus_vlwa_import'], namespace


def test_explicit_source_precedes_link(monkeypatch, tmp_path):
    configure, namespace = source_configurator()
    source = tmp_path / 'src'
    source.mkdir()
    policy = tmp_path / 'policy'
    policy.mkdir()
    (policy / 'focus-vlwa').symlink_to(tmp_path / 'missing')
    namespace['_POLICY_DIR'] = policy
    monkeypatch.setattr(sys, 'path', sys.path[:])
    monkeypatch.setenv('FOCUS_VLWA_SOURCE', str(source))
    configure({'focus_vlwa_source': str(tmp_path / 'also_missing')})
    assert sys.path[0] == str(source)


def test_linked_source_uses_same_package(monkeypatch, tmp_path):
    from XPolicyLab.policy.ME_Brain_1 import _source

    configure, namespace = source_configurator()
    policy = tmp_path / 'policy'
    policy.mkdir()
    repository = tmp_path / 'repository'
    (repository / 'src/focus_vlwa').mkdir(parents=True)
    (policy / 'focus-vlwa').symlink_to(repository)
    namespace['_POLICY_DIR'] = policy
    monkeypatch.setattr(_source, '__file__', str(policy / '_source.py'))
    monkeypatch.setattr(sys, 'path', sys.path[:])
    monkeypatch.delenv('FOCUS_VLWA_SOURCE', raising=False)
    configure({})
    assert str(repository / 'src') in sys.path


def test_relative_source_resolves_from_policy_directory(monkeypatch, tmp_path):
    configure, namespace = source_configurator()
    policy = tmp_path / 'policy'
    source = policy / 'checkout' / 'src'
    source.mkdir(parents=True)
    namespace['_POLICY_DIR'] = policy
    monkeypatch.setattr(sys, 'path', sys.path[:])
    monkeypatch.delenv('FOCUS_VLWA_SOURCE', raising=False)
    configure({'focus_vlwa_source': 'checkout/src'})
    assert sys.path[0] == str(source)
