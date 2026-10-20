import copy

import torch
from transformers import Qwen3VLConfig

from XPolicyLab.policy.Liber_0_lite.liber0.backbone import JointModel, LiberModel, _install_target_only_global_norm_patch16
from XPolicyLab.policy.Liber_0_lite.liber0.batching import joint_forward
from XPolicyLab.policy.Liber_0_lite.liber0.vendor.qwen3_vl_transformers import Qwen3VLForConditionalGeneration


def test_actual_decoder_backward_with_checkpointing():
    torch.manual_seed(8)
    config = Qwen3VLConfig(
        text_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         vocab_size=256, pad_token_id=0,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [4, 6, 6]}),
        vision_config=dict(hidden_size=64, intermediate_size=128, out_hidden_size=128,
                           depth=2, num_heads=4, deepstack_visual_indexes=[0], num_position_embeddings=64),
    )
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    backbone = Qwen3VLForConditionalGeneration(config)
    model = LiberModel()
    model.target_projection_scale = _install_target_only_global_norm_patch16(backbone)
    model.dit = JointModel(backbone, 14, 14, 64)
    model.device, model.torch_dtype = torch.device('cpu'), torch.float32
    model.train()
    saved = copy.deepcopy(model.state_dict())
    values, gradients = [], []
    for checkpointing in (False, True):
        model.load_state_dict(saved)
        model.zero_grad()
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            backbone.gradient_checkpointing_disable()
        torch.manual_seed(19)
        text = backbone.model.get_input_embeddings()(torch.tensor([[3, 4, 5], [3, 4, 5]]))
        condition = dict(text=text, tms_mask=torch.tensor([[False, False, True]]).expand(2, -1),
                         valid=torch.ones(2, 3, dtype=torch.bool),
                         positions=torch.arange(11).reshape(1, 1, -1).expand(3, 2, -1),
                         visual_mask=torch.zeros(2, 3, dtype=torch.bool), deepstack=[])
        video, action = joint_forward(model, condition, torch.randn(2, 4, 768), torch.randn(2, 14),
                                      torch.randn(2, 4, 768), torch.randn(2, 32, 14), torch.rand(2), torch.rand(2))
        loss = video.square().mean() + action.square().mean()
        loss.backward()
        assert model.dit.action_head.weight.grad is not None
        assert backbone.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None
        values.append(loss.detach())
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    assert torch.equal(*values)
    assert gradients[0].keys() == gradients[1].keys()
    assert all(torch.equal(gradients[0][name], gradients[1][name]) for name in gradients[0])
