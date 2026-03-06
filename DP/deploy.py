import numpy as np
import os
from .model import DP
import yaml

def get_model(usr_args):
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    ckpt_file = os.path.join(parent_dir, f"checkpoints/{usr_args['task_name']}-{usr_args['env_cfg']}-{usr_args['expert_data_num']}-{usr_args['action_type']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt")
    
    action_dim = usr_args['action_dim']
    load_config_path = os.path.join(parent_dir, f'diffusion_policy/config/robot_dp_{action_dim}.yaml')
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    action_type = usr_args['action_type']
    
    model =  DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, action_type=action_type)
    return model

def eval_one_episode(TASK_ENV, model_client):

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            
            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs() # Get Observation
                model_client.call(func_name="update_obs", obs=obs)
