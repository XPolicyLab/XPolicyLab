import builtins
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from XPolicyLab.policy.Liber_0_lite.liber0.training import entrypoint
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from .test_training import ToyModel, sample

POLICIES = entrypoint.LITE_DIR.parent


def arguments(policy, mode='train', seed=42):
    args = ['--policy-dir', str(POLICIES / policy), mode, 'RoboDojo', 'sft', 'arx_x5', 'joint']
    if mode == 'train':
        args += [str(seed), '0,1', '--assets', './assets', '--batch-size', '16', '--accumulation', '32', '--num-workers', '0']
    return args + ['--dataset', './dataset', '--checkpoint', './base', '--dry-run']


@pytest.mark.parametrize('policy', ['Liber_0_lite', 'Liber_0_preview'])
def test_standard_train_contract(policy, monkeypatch):
    monkeypatch.delenv('NNODES', raising=False)
    args = entrypoint.parser().parse_args(arguments(policy, seed=17))
    command = entrypoint.command_for(args)
    expected = POLICIES / policy / 'checkpoints' / build_run_dir_name(
        dict(bench_name='RoboDojo', ckpt_name='sft', env_cfg_type='arx_x5', action_type='joint', seed=17))
    assert command[command.index('--output') + 1] == str(expected)
    assert command[command.index('--seed') + 1] == '17'
    assert command[command.index('--nproc_per_node') + 1] == '2'
    assert command[command.index('--checkpoint') + 1] == str(POLICIES / policy / 'base')
    assert command[command.index('--config') + 1] == str(entrypoint.LITE_DIR / 'configs/robodojo_sft.yaml')
    assert not expected.exists()


@pytest.mark.parametrize('mode', ['train', 'data'])
@pytest.mark.parametrize('policy', ['Liber_0_lite', 'Liber_0_preview'])
def test_shell_entrypoint_dry_run(policy, mode):
    args = arguments(policy, mode)
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'], NNODES='1')
    script = POLICIES / policy / ('train.sh' if mode == 'train' else 'process_data.sh')
    completed = subprocess.run(['bash', str(script), *args[3:]], env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    result = json.loads(completed.stdout)
    assert result['policy'] == policy
    assert entrypoint.MODULE + ('.train' if mode == 'train' else '.check_data') in result['command']
    if mode == 'train':
        assert result['cuda_visible_devices'] == '0,1'


def test_multi_node_contract(monkeypatch):
    monkeypatch.setenv('NNODES', '2')
    monkeypatch.setenv('NODE_RANK', '1')
    monkeypatch.setenv('MASTER_ADDR', 'localhost')
    monkeypatch.setenv('MASTER_PORT', '29511')
    args = entrypoint.parser().parse_args(arguments('Liber_0_preview'))
    args.accumulation = 16
    command = entrypoint.command_for(args)
    assert command[command.index('--nnodes') + 1] == '2'
    assert command[command.index('--node_rank') + 1] == '1'
    assert '--standalone' not in command


def test_invalid_batch_is_not_launched(monkeypatch):
    monkeypatch.setenv('NNODES', '1')
    args = entrypoint.parser().parse_args(arguments('Liber_0_lite'))
    args.accumulation = 1
    with pytest.raises(ValueError, match='Global batch'):
        entrypoint.command_for(args)


def test_data_subset_not_silently_ignored():
    args = entrypoint.parser().parse_args(arguments('Liber_0_preview', 'data'))
    args.expert_data_num = 10
    with pytest.raises(ValueError, match='does not truncate'):
        entrypoint.command_for(args)


def test_training_routes_are_identical_without_extra_model(monkeypatch):
    monkeypatch.setenv('NNODES', '1')
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if 'Liber_0_preview' in name or name.startswith('flash_attn'):
            raise AssertionError(f'Unexpected training import: {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked_import)
    results = []
    for policy in ('Liber_0_lite', 'Liber_0_preview'):
        command = entrypoint.command_for(entrypoint.parser().parse_args(arguments(policy)))
        name = command[command.index('-m', 3) + 1]
        module = importlib.import_module(name)
        torch.manual_seed(42)
        model = ToyModel()
        loss, _ = module.TrainingModel(model)(sample())
        loss.backward()
        results.append((loss.detach(), model.weight.grad.clone()))
    assert all(torch.equal(left, right) for left, right in zip(*results))


def test_both_policy_wrappers_stay_equal():
    for name in ('train.sh', 'process_data.sh'):
        assert (POLICIES / 'Liber_0_lite' / name).read_bytes() == (POLICIES / 'Liber_0_preview' / name).read_bytes()
