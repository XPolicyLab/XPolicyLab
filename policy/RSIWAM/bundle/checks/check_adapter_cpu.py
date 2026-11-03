"""Load real adapter/parent into a CPU structural skeleton, not the full VPP2.

Base Linear weights use expanded scalar storage and are never evaluated.
This checks real tensor structure, composition, frozen state and exact loading;
it does not establish forward-output equivalence or simulator success.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runtime')]
import torch
from rsiwam_loader import check_package, compose


def linear(inputs, outputs):
    result = torch.nn.Linear.__new__(torch.nn.Linear)
    torch.nn.Module.__init__(result)
    result.in_features, result.out_features = inputs, outputs
    result.weight = torch.nn.Parameter(torch.zeros(1).expand(outputs, inputs), requires_grad=False)
    result.register_parameter('bias', None)
    return result


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU check must not initialize CUDA')
    artifacts = check_package()
    adapter = torch.load(artifacts['adapter'], map_location='cpu', weights_only=True)
    parent = torch.load(artifacts['parent'], map_location='cpu', weights_only=True)
    saved = adapter['state_dict']
    assert len(saved) == 160
    assert sum(value.numel() for value in saved.values()) == 3_932_160
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in saved.values())
    assert parent['config']['threshold'] == 0.9
    assert sum(value.numel() for value in parent['state_dict'].values()) == 1_597_697
    model = torch.nn.Module()
    model.device = torch.device('cpu')
    model.raw_action_dim = model.proprio_dim = 16
    model.action_chunk_size = 1
    action = torch.nn.Module()
    action.hidden_dim = parent['config']['action_dim']
    action.text_embedding = torch.nn.ModuleList([linear(parent['config']['context_dim'], action.hidden_dim)])
    action.blocks = torch.nn.ModuleList()
    for index in range(40):
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        for kind in ['q', 'v']:
            prefix = f'blocks.{index}.self_attn.{kind}'
            a, b = saved[prefix + '.lora_A'], saved[prefix + '.lora_B']
            assert a.shape[0] == b.shape[1] == 8
            setattr(block.self_attn, kind, linear(a.shape[1], b.shape[0]))
        action.blocks.append(block)
    model.action_expert = action
    model.video_expert = torch.nn.Module()
    video_block = torch.nn.Module()
    video_block.self_attn = torch.nn.Module()
    video_block.self_attn.v = linear(1, parent['config']['video_dim'])
    model.video_expert.blocks = torch.nn.ModuleList([video_block])
    model.mot = torch.nn.Module()
    model.mot.mixtures = {'action': action}
    model.train_video_scheduler = SimpleNamespace(train_sampling_strategy='pure_noise')
    policy = SimpleNamespace(model=model, action_horizon=32)
    before = {id(p): (p.data_ptr(), p._version) for p in model.parameters()}
    compose(policy, artifacts)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all((p.data_ptr(), p._version) == before[id(p)] for p in model.parameters() if id(p) in before)
    assert len(before) == sum(id(p) in before for p in model.parameters())
    from recoverwam.recovery.lora import lora_modules
    modules = list(lora_modules(policy))
    assert len(modules) == 80
    for name, module in modules:
        for suffix in ['lora_A', 'lora_B']:
            assert torch.equal(getattr(module, suffix), saved[name + '.' + suffix])
    assert model.recovery_mode == 'auto' and not model.training
    assert not model.action_expert.use_gradient_checkpointing
    assert not model.mot.mot_checkpoint_mixed_attn
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='REAL_WEIGHTS_CPU_STRUCTURAL_COMPOSITION_PASSED',
                         torch=torch.__version__, adapter_tensors=len(saved),
                         lora_parameters=3_932_160, parent_parameters=1_597_697,
                         lora_modules=80, lora_rank=8, lora_alpha=8,
                         gate_threshold=0.9, all_parameters_frozen=True,
                         base_storage_preserved=True, adapter_loaded_exact=True,
                         gpu_used=False, base_forward_evaluated=False,
                         clean_environment_validated=False), indent=2))


if __name__ == '__main__':
    main()
