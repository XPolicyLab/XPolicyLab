"""Select checked-in RoboDojo robot profiles without changing checkpoint lineage."""
import json
from pathlib import Path

SUPPORTED_ROBOTS = ('arx_x5', 'piper', 'piper_x')
PROMPT_KEY = 'mixtures.action.embodiment_prompt.embedding.weight'


def select_robot_assets(paths, robot_type):
    if robot_type not in SUPPORTED_ROBOTS:
        raise ValueError(f'Unsupported Phoenix_WAM robot: {robot_type}; expected {SUPPORTED_ROBOTS}')
    profile_path = Path(paths['robot_profiles']) / f'{robot_type}.json'
    profile = json.loads(profile_path.read_text())
    if profile['robot_type'] != robot_type or profile['action_type'] != 'joint':
        raise ValueError('Robot profile identity mismatch')
    registry = json.loads(Path(paths['registry']).read_text())
    slot = profile['embodiment_id']
    if (type(slot) is not int or not 0 <= slot < registry['num_embodiments']
            or registry['embodiments'][slot] != profile['registry_name']
            or registry['slot_status'][slot] != 'assigned'):
        raise ValueError('Robot profile / registry slot mismatch')
    relative = Path(profile['dataset_stats'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Robot statistics must use a relative source path')
    result = dict(paths)
    result['robot_profile'] = profile
    result['robot_profile_path'] = profile_path
    result['dataset_stats'] = profile_path.parent.parent / relative
    return result


def checkpoint_robot_support(payload, profile):
    import torch
    prompt = payload['joint96_metadata']['soft_prompt']
    registry = prompt['registry']
    slot = profile['embodiment_id']
    if (registry['embodiments'][slot] != profile['registry_name']
            or registry['slot_status'][slot] != 'assigned'):
        raise ValueError('Checkpoint registry does not match the selected robot')
    weight = payload.get('mot', {}).get(PROMPT_KEY)
    expected = (prompt['num_embodiments'], prompt['num_tokens'] * prompt['context_dim'])
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected:
        raise ValueError('Checkpoint soft prompt tensor is missing or has the wrong shape')
    selected = weight[slot].float()
    if not torch.isfinite(selected).all() or not torch.count_nonzero(selected):
        raise ValueError('Checkpoint robot soft prompt must be finite and nonzero')
    trained = prompt.get('training_contract', {}).get('embodiment_ids', {})
    return dict(robot_type=profile['robot_type'], embodiment_id=slot,
                registry_name=profile['registry_name'], tensor_shape=list(weight.shape),
                prompt_norm=float(selected.norm()), structurally_supported=True,
                posttrained=trained.get(profile['robot_type']) == slot,
                posttraining_embodiment_ids=trained, task_success_verified=False)
