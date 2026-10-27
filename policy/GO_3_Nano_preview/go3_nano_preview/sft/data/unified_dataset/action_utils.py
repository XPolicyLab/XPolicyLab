import numpy as np
import torch

def norm_and_filter(action, state, stat, norm_action=True, filter_action=True, action_key='action', state_key='state', debug=True):
    if norm_action or filter_action:
        action_mean = np.array(stat[action_key]['mean'])
        action_std = np.array(stat[action_key]['std'])
        state_mean = np.array(stat[state_key]['mean'])
        state_std = np.array(stat[state_key]['std'])
    if norm_action:
        assert stat is not None
        action = (action - action_mean) / action_std
        state = (state - state_mean) / state_std
        if filter_action:
            if (np.abs(action) > 50).sum() > 0 or (np.abs(state) > 150).sum() > 0:
                if debug:
                    action_indices = np.where(np.abs(action) > 50)
                    state_indices = np.where(np.abs(state) > 150)
                    print(f'Debug: Action values exceeding threshold at indices {action_indices}: {action[action_indices]}')
                    print(f'Debug: State values exceeding threshold at indices {state_indices}: {state[state_indices]}')
                raise ValueError('Error: action or state value exceeds threshold')
    elif filter_action:
        if (np.abs((action - action_mean) / action_std) > 50).sum() > 0 or (np.abs((state - state_mean) / state_std) > 100).sum() > 0:
            if debug:
                action_indices = np.where(np.abs((action - action_mean) / action_std) > 50)
                state_indices = np.where(np.abs((state - state_mean) / state_std) > 50)
                print(f'Debug: Action values exceeding threshold at indices {action_indices}: {(action - action_mean)[action_indices]}')
                print(f'Debug: State values exceeding threshold at indices {state_indices}: {(state - state_mean)[state_indices]}')
            raise ValueError('Error: action or state value exceeds threshold')
    return (action, state)

def get_processed_actions(action_path, end_frames_action, n_previous_action, action_slices, stat=None, action_mode='abs', norm_action=True, filter_action=True):
    assert action_mode in {'abs', 'delta', 'rela'}
    if action_mode != 'abs':
        assert n_previous_action > 0, 'To use delta or rela action, n_previous_action is needed.'
    action_slices = action_slices[:-end_frames_action] if end_frames_action != 0 else action_slices
    loaded_action = torch.load(action_path, weights_only=False)
    if isinstance(loaded_action, torch.Tensor):
        loaded_action = loaded_action.detach().cpu().numpy()
    else:
        loaded_action = np.asarray(loaded_action)
    action = loaded_action[action_slices].astype(np.float32, copy=False)
    state = action[n_previous_action - 1:n_previous_action].copy()
    action = action[n_previous_action:].copy()
    if action.shape[1] != 14:
        raise ValueError('Robotwin actions must have 14 dimensions')
    joint_dim = 6
    if action_mode == 'abs':
        if end_frames_action > 0:
            action = np.concatenate((action, np.tile(action[-1:], (end_frames_action, 1))), axis=0)
        (action, state) = norm_and_filter(action, state, stat, norm_action, filter_action)
    elif action_mode == 'rela':
        action[:, :joint_dim] = action[:, :joint_dim] - state[:, :joint_dim]
        action[:, joint_dim + 1:2 * joint_dim + 1] = action[:, joint_dim + 1:2 * joint_dim + 1] - state[:, joint_dim + 1:2 * joint_dim + 1]
        if end_frames_action > 0:
            action = np.concatenate((action, np.tile(action[-1:], (end_frames_action, 1))), axis=0)
        (action, state) = norm_and_filter(action, state, stat, norm_action, filter_action=filter_action)
    elif action_mode == 'delta':
        action[1:, :joint_dim] = action[1:, :joint_dim] - action[:-1, :joint_dim]
        action[0, :joint_dim] = action[0, :joint_dim] - state[0, :joint_dim]
        action[1:, joint_dim + 1:2 * joint_dim + 1] = action[1:, joint_dim + 1:2 * joint_dim + 1] - action[:-1, joint_dim + 1:2 * joint_dim + 1]
        action[0, joint_dim + 1:2 * joint_dim + 1] = action[0, joint_dim + 1:2 * joint_dim + 1] - state[0, joint_dim + 1:2 * joint_dim + 1]
        if end_frames_action > 0:
            action = np.concatenate((action, np.zeros((end_frames_action, action.shape[-1]))), axis=0)
        (action, state) = norm_and_filter(action, state, stat, norm_action, filter_action)
    return (action, state, joint_dim)
