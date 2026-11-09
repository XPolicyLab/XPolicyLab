"""Load real checkpoint processor metadata without importing the research package."""
import copy
import importlib.abc
import json
from pathlib import Path
import sys

from hydra.utils import instantiate
import pytest
import torch

from bagel_cp1000.checkpoint_compat import action_config

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/checkpoint_processor.json').read_text())


class RejectResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vpp2', 'fastwam', 'robodojo_joint2b'}:
            raise AssertionError(f'Standalone processor requested research import: {fullname}')


def replace_targets(value, prefix):
    if isinstance(value, list):
        return [replace_targets(x, prefix) for x in value]
    if not isinstance(value, dict):
        return value
    return {key: (item.replace('vpp2.', prefix, 1) if key == '_target_' else replace_targets(item, prefix))
            for key, item in value.items()}


@pytest.mark.parametrize('prefix', ['vpp2.', 'fastwam.', 'robodojo_joint2b.'])
def test_real_checkpoint_processor_instantiates_without_research_imports(prefix):
    config = replace_targets(copy.deepcopy(FIXTURE['processor']), prefix)
    original = copy.deepcopy(config)
    payload = {'format': FIXTURE['format'], 'resolved_config': {'processor': config}}
    guard = RejectResearchImports()
    sys.meta_path.insert(0, guard)
    try:
        processor = instantiate(action_config(payload)['processor']).eval()
        assert type(processor).__module__.startswith('bagel_cp1000.')
        assert type(processor.action_state_merger).__module__.startswith('bagel_cp1000.')
        assert processor.action_output_dim == processor.proprio_output_dim == 16
        # EE16 inference must preserve values/order before normalization.
        state = torch.arange(16, dtype=torch.float32).reshape(1, 16)
        action = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
        batch = {'state': {'default': state.clone()}, 'action': {'default': action.clone()}}
        result = processor.action_state_merger.forward(processor.action_state_transform(batch))
        torch.testing.assert_close(result['state'], state, rtol=0, atol=0)
        torch.testing.assert_close(result['action'], action, rtol=0, atol=0)
        assert not result['action_dim_is_pad'].any()
        pixels = torch.tensor([0, 127, 255], dtype=torch.uint8)
        torch.testing.assert_close(processor.val_transforms[0](pixels), pixels.float() / 255, rtol=0, atol=0)
        assert config == original  # serialized metadata must not be mutated
    finally:
        sys.meta_path.remove(guard)


def test_target_isolation_preserves_non_target_strings_and_input():
    source = {'_target_': 'fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor',
              'path': 'vpp2/checkpoint.pt', 'prompt': 'vpp2.example',
              'nested': [{'_target_': 'torch.nn.Identity'}]}
    expected = copy.deepcopy(source)
    result = action_config({'format': 'fastwam_wan21_action_v1', 'resolved_config': source})
    assert result['_target_'] == 'bagel_cp1000.datasets.lerobot.processors.vpp2_processor.VPP2Processor'
    assert result['path'] == expected['path'] and result['prompt'] == expected['prompt']
    assert result['nested'] == expected['nested'] and source == expected
