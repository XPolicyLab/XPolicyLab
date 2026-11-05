"""Resolve portable assets through the shared XPolicyLab checkpoint resolver."""
from pathlib import Path
import json
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR = Path(__file__).resolve().parent

def resolve_assets(cfg):
    root = resolve_checkpoint_root(cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    model = root / "model" if (root / "model/config.json").is_file() else root
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"No model/config.json or config.json under {root}")
    norm = cfg.get("norm_path")
    norm = Path(norm).expanduser() if norm else next(
        (v for v in (root/'norm_stats.pkl', model/'norm_stats.pkl', model.parent/'norm_stats.pkl', model.parent.parent/'norm_stats.pkl') if v.is_file()),
        root/'norm_stats.pkl')
    if not norm.is_absolute(): norm = POLICY_DIR / norm
    required = [model/'hy_xvla_adapter.json', norm, model/'tokenizer_config.json']
    for v in required:
        if not v.is_file(): raise FileNotFoundError(f"Required model asset: {v}")
    if not any((model/name).is_file() for name in ('processor_config.json','preprocessor_config.json')):
        raise FileNotFoundError(f'Missing image processor configuration under {model}')
    if not any(model.glob('*.safetensors')):
        raise FileNotFoundError(f"No safetensors weights under {model}")
    spec = json.loads((model/'config.json').read_text())
    expected = dict(vlm_backend='qwen3_5', architecture_version='vlm_fast_action_bridge',
        qwen_temporal_embedding='vlm_history_current_v1',
        qwen_image_attention_mode='bidirectional_image_blocks_current_only',
        action_context_mode='prefix_kv', num_actions=25, action_loss_mode='flow_matching')
    for key, value in expected.items():
        if spec.get(key) != value: raise ValueError(f"Checkpoint {key}: {spec.get(key)!r} != {value!r}")
    adapter = json.loads((model/'hy_xvla_adapter.json').read_text())
    for key,value in dict(domain_id=16,action_representation='relative_chunk_ee_RT',
                          action_normalization='zscore',action_loss_mode='flow_matching').items():
        if adapter.get(key) != value: raise ValueError(f'Adapter {key}: {adapter.get(key)!r} != {value!r}')
    if spec.get('vlm_policy_contract') != 'qwen35_224_letterbox_premerge196_semantic_pyramid_v4':
        raise ValueError('Unexpected Qwen image preprocessing contract')
    return dict(root=root, model=model, norm=norm, config=spec)
