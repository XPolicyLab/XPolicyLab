"""Forward insert-key observations/actions without rebinding native methods."""
from __future__ import annotations

FIELDS = (
    'left_arm_joint_state', 'right_arm_joint_state',
    'left_ee_pose', 'right_ee_pose',
    'left_ee_joint_state', 'right_ee_joint_state',
)


def clean(observation: dict) -> dict:
    """Keep the frozen policy's existing RGB/proprioception contract."""
    return {
        'vision': {key: {'color': value['color']}
                   for key, value in observation['vision'].items()},
        'state': {key: observation['state'][key] for key in FIELDS},
        'instruction': observation['instruction'],
    }


def eval_one_episode_batch(TASK_ENV, model_client) -> None:
    """Leave reset, IK, stepping, termination and scoring to the evaluator."""
    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(indices)
        model_client.call(
            func_name='update_obs_batch',
            obs=[dict(clean(observation), env_idx=index)
                 for index, observation in zip(indices, observations)],
        )
        chunks = model_client.call(func_name='get_action_batch', obs=indices)
        actions = [
            {key.replace('_gripper', '_ee_joint_state'): value
             for key, value in chunk[0].items()}
            for chunk in chunks
        ]
        TASK_ENV.take_action_batch(actions, indices)
