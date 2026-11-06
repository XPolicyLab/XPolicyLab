@staticmethod
def _state_from_obs(obs: dict) -> np.ndarray:
    """16-d state [L: xyz qw qx qy qz grip, R: same] from runtime obs."""
    st = obs.get('state') or {}
    try:
        v = np.concatenate([np.asarray(st['left_ee_pose'], np.float32).reshape(-1), np.asarray(st['left_ee_joint_state'], np.float32).reshape(-1)[:1], np.asarray(st['right_ee_pose'], np.float32).reshape(-1), np.asarray(st['right_ee_joint_state'], np.float32).reshape(-1)[:1]])
    except KeyError as e:
        raise KeyError(f"obs['state'] missing {e}; have {list(st)}. Ensure the env cfg collects world_ee_state + joint_states.") from e
    if v.shape[0] != _STATE_DIM:
        raise ValueError(f'expected {_STATE_DIM}-d state, got {v.shape}')
    return v
