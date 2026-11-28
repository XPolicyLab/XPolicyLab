"""Create a per-user runtime; never mutate source checkpoints or shared code."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml
from wall_x.config.paths import resolve_vap_config_paths

root = Path(__file__).resolve().parent
runtime = Path(os.environ['VAP_RUN_DIR']) / 'runtime'
source = Path(os.environ['VAP_SOURCE_CHECKPOINT']).expanduser().resolve()
cfg = yaml.safe_load((source / 'config.yml').read_text())
# Resolve against the original config before writing the per-run view elsewhere.
resolve_vap_config_paths(cfg, source / 'config.yml')
m = cfg['model']
# Training augmentation metadata identifies how weights were trained; it is not
# an inference compatibility constraint. Preserve the checkpoint's values.
training_augmentation = {
    key: m.get(key) for key in (
        'action_prefix_noise_enabled', 'action_prefix_noise_probability',
        'action_prefix_noise_std', 'action_prefix_noise_exclude_dims',
    )
}
expected = {
    'architecture_version': 'vapv2.qwen_joint_ivava.v2',
    'action_steps_per_chunk': 32, 'video_frames_per_chunk': 16,
    'width': 320, 'height': 256, 'history_offsets': [32, 64, 96, 128, 256],
    'history_include_first_frame': True,
    'first_frame_memory_enabled': True, 'first_frame_memory_grid': [8, 10],
    'first_frame_memory_injection_mode': 'prefix_self_attention',
    'first_frame_memory_position_mode': 'relative_age',
    'first_frame_memory_view_keys': ['view1'], 'state_film_enabled': False,
    'qwen35_text_recipe': 'plain_text_head_truncate_eos_last_hidden_v1', 'qwen35_truncation': 'head',
    'action_expert_dim': 1024, 'text_feature_hidden_dim': 4096,
    'text_feature_seq_len': 512, 'text_projector_output_dim': 3072,
}
assert cfg['model_type'] == 'vapv2'
assert cfg['data']['dataset_config_effective']['dataset']['window']['future_interval'] == 2
assert cfg['data']['dataset_config_effective']['dataset']['window']['future_length'] == 32
assert cfg['data']['dataset_config_effective']['dataset']['window']['action_horizon'] == 64
for key, value in expected.items():
    if m.get(key) != value:
        raise ValueError(f'Unsupported checkpoint recipe: {key}={m.get(key)!r}, expected {value!r}')
assert cfg['data']['dataset_config_effective']['_wall_x_runtime']['first_frame_memory']['history_offsets'] == m['history_offsets']
# This checkpoint's saved HEAD recipe is authoritative; never borrow the old
# no-truncation reference config or its cache identity.
training_cache_id = m['text_feature_cache_id']
assert training_cache_id == '5c745ef1bf674f6bf7c8f5b2f7fce5633856ff2fa7d928c04663c6fde18554f6'
from wall_x.model.vap.qwen35_online import model_identity
identity, hidden_width = model_identity(Path(os.environ['VAP_QWEN_MODEL']))
assert identity == m['qwen35_model_id'] == '2384e8ef3084e64f12056620249e011609a2cfa813227ad3c8b9050bc68d1474'
assert hidden_width == m['text_feature_hidden_dim']
assert m.get('qwen35_model_id') in (None, identity)
m['qwen35_model_id'] = identity
memory_contract = cfg['data']['dataset_config_effective']['_wall_x_runtime']['first_frame_memory']
assert memory_contract['enabled'] is True
assert memory_contract.get('anchor') in (None, 'resolved_episode_local_index_0')
memory_contract['anchor'] = 'resolved_episode_local_index_0'
for name in ['model.safetensors', 'normalizer_action.pth']:
    if not (source / name).is_file():
        raise FileNotFoundError(source / name)
# Require this checkpoint's own normalizer and verify it against the explicitly
# named training statistics. Never borrow another model's normalizer.
canonical_stats = root / 'configs/normalizer_keys_robodojo_ee_v3_full_master_relative_global.json'
stats = Path(os.environ.get('VAP_NORMALIZER_STATS') or
             cfg['data']['normalizer_config']['customized_action_statistic_dof']).expanduser()
if not stats.is_absolute():
    stats = source / stats
stats = stats.resolve()
stats_sha = hashlib.sha256(stats.read_bytes()).hexdigest()
assert stats_sha == hashlib.sha256(canonical_stats.read_bytes()).hexdigest()
assert stats_sha == 'bc30897670ed5210b66dcc52aaf39bc9ea7726be56f1d7ff9c3c08d48fc73f75'
norm_cfg = cfg['data']['normalizer_config']
action_cfg = cfg['data']['dataset_config_effective']['dataset']['processors']['action']['params']
assert Path(norm_cfg['customized_action_statistic_dof']).name == canonical_stats.name
assert Path(action_cfg['normalizer_config']['stats_file']).name == canonical_stats.name
for key, value in {'min_key': 'q001', 'delta_key': 'delta_001', 'clip': True}.items():
    assert norm_cfg[key] == action_cfg['normalizer_config'][key] == value
assert action_cfg['predict_action_keys'] == cfg['task']['dof_config']
assert action_cfg['relative_action_chunk_size'] == 32
assert action_cfg['prediction_start_offset'] == 0
assert cfg['data']['action_chunks_already_local'] is True
if runtime.exists():
    raise FileExistsError(f'Refuse to overwrite runtime: {runtime}')
runtime.mkdir(parents=True)
(runtime / 'tmp').mkdir()
view = runtime / 'checkpoint_online'
view.mkdir()
for name in ['model.safetensors']:
    (view / name).symlink_to((source / name).resolve())
cfg['checkpoint']['resume_from'] = None
cfg['checkpoint']['strict_model_load'] = False  # Serving subsequently strictly loads learned weights.
m.update(text_encoder_backend='qwen35_online', qwen35_model_path=os.environ['VAP_QWEN_MODEL'],
         text_feature_cache_path=None, text_feature_cache_id=None)
cfg['data']['normalizer_config']['customized_action_statistic_dof'] = str(stats)
action_cfg['normalizer_config']['stats_file'] = str(stats)
m['action_flow']['inference_steps'] = int(os.environ['VAP_ACTION_INFERENCE_STEPS'])
m['action_flow']['shift'] = float(os.environ['VAP_ACTION_INFERENCE_SHIFT'])
(view / 'config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
import torch
from wall_x.config.loader import load_config
from wall_x.model.vap.serving import _build_action_normalizer, _load_matching_normalizer_state
normalizer, norm_key = _build_action_normalizer(load_config(str(view/'config.yml')), 'robodojo_ee_v3')
if (source/'normalizer_action.pth').exists():
    _load_matching_normalizer_state(normalizer, source/'normalizer_action.pth', label='source checkpoint normalizer')
    (view/'normalizer_action.pth').symlink_to((source/'normalizer_action.pth').resolve())
    norm_origin = 'source_checkpoint_verified_against_packaged_training_stats'
else:
    torch.save(normalizer.state_dict(), view/'normalizer_action.pth')
    norm_origin = 'reconstructed_from_explicit_training_config_and_canonical_global_stats'

(runtime/'checkpoint_preparation.json').write_text(json.dumps({'source':str(source),'source_config_sha256':hashlib.sha256((source/'config.yml').read_bytes()).hexdigest(),'normalizer_stats_sha256':stats_sha,'training_cache_id':training_cache_id,'qwen_identity':identity,'recipe':expected,'training_augmentation':training_augmentation,'normalizer_origin':norm_origin},indent=2))
print('CHECKPOINT_PREPARED',runtime,flush=True)
