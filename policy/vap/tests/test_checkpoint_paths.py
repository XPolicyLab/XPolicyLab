"""Relocation contracts; no model weights or GPUs are needed."""
import copy
import importlib.util
from pathlib import Path

import pytest

POLICY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'vap_checkpoint_paths', POLICY / 'upstream/wall_x/config/paths.py')
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


def test_config_paths_survive_cwd_and_runtime_view_changes(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'moved package' / 'baseline'
    checkpoint.mkdir(parents=True)
    stats = checkpoint / 'stats.json'
    stats.write_text('{}')
    raw = {
        'model_type': 'vapv2',
        'checkpoint': {'resume_from': None, 'save_path': 'training/checkpoints'},
        'model': {'qwen35_model_path': '../Qwen', 'text_feature_cache_path': None,
                  'model_id': 'Wan-AI/Wan2.2:diffusion_model*.safetensors'},
        'data': {'normalizer_config': {'customized_action_statistic_dof': './stats.json'}},
    }
    monkeypatch.chdir(tmp_path)
    paths.resolve_vap_config_paths(raw, checkpoint / 'config.yml')
    assert Path(raw['data']['normalizer_config']['customized_action_statistic_dof']).read_text() == '{}'
    assert raw['model']['qwen35_model_path'] == str(checkpoint.parent / 'Qwen')
    assert raw['checkpoint']['resume_from'] is None
    assert raw['model']['text_feature_cache_path'] is None
    assert raw['model']['model_id'].startswith('Wan-AI/')
    resolved = copy.deepcopy(raw)
    paths.resolve_vap_config_paths(raw, tmp_path / 'runtime/checkpoint_online/config.yml')
    assert raw == resolved  # writing a runtime view must not rebase paths again


@pytest.mark.parametrize('keys', paths.VAP_PATH_FIELDS)
def test_every_supported_field_accepts_relative_and_legacy_absolute(tmp_path, keys):
    for value, expected in [('assets/file', str(tmp_path / 'assets/file')),
                            ('/legacy/assets/file', '/legacy/assets/file')]:
        raw = {'model_type': 'vapv2'}
        node = raw
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
        paths.resolve_vap_config_paths(raw, tmp_path / 'config.yml')
        assert node[keys[-1]] == expected


def test_non_vap_config_is_unchanged(tmp_path):
    raw = {'model_type': 'pi0', 'checkpoint': {'resume_from': 'relative/path'}}
    before = copy.deepcopy(raw)
    paths.resolve_vap_config_paths(raw, tmp_path / 'config.yml')
    assert raw == before
