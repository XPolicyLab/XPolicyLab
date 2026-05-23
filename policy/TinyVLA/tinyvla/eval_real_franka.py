import os
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch
from torchvision import transforms
import cv2
from copy import deepcopy
from itertools import repeat
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import numpy as np
import time
from aloha_scripts.constants import FPS

from data_utils.datasets import set_seed
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import matplotlib.pyplot as plt
import sys

def get_image(ts, camera_names, rand_crop_resize=False):
    """
    Retrieves and processes images from the specified cameras.

    Args:
        ts: The timestamp or data structure containing observations.
        camera_names: List of camera names to retrieve images from.
        rand_crop_resize: Boolean indicating whether to apply random crop and resize.

    Returns:
        A tensor containing the processed images.
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def pre_process(robot_state_value, key, stats):
    """
    Pre-processes the robot state value using provided statistics.

    Args:
        robot_state_value: The raw robot state value.
        key: The key to access the corresponding statistics.
        stats: Dictionary containing mean and standard deviation for normalization.

    Returns:
        The normalized robot state value.
    """
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def get_obs():
    """
    Retrieves observations (images and robot states) from the robot environment.

    Returns:
        A tuple containing images and states.
    """
    return None, None # images, states


def time_ms():
    return time.time_ns() // 1_000_000


def convert_actions(pred_action):
    # pred_action = torch.from_numpy(actions)
    # pred_action = actions.squeeze(0)
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    # print(f'cur_xyz size: {cur_xyz.shape}')
    # print(f'cur_euler size: {cur_euler.shape}')
    # print(f'cur_gripper size: {cur_gripper.shape}')
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    # print(f'4. pred_action size: {pred_action.shape}')
    print(f'4. after convert pred_action: {pred_action}')

    return pred_action

class llava_pythia_act_policy:
    """
    Policy class for Llava-Pythia action generation.

    Attributes:
        policy_config: Configuration dictionary for the policy.
    """
    def __init__(self, policy_config, data_args=None):
        super(llava_pythia_act_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        # self.conv = conv_templates[policy_config['conv_mode']].copy()
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_name = get_model_name_from_path(policy_config['model_path'])
        model_path = policy_config["model_path"]

        self.tokenizer, self.policy, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base,
                                                                                                    model_name, False,
                                                                                                    False)
        self.config = LlavaPythiaConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        Processes a batch of data for Llava-Pythia model input.

        Args:
            curr_image: Current image tensor.
            robo_state: Current robot state tensor.
            raw_lang: Raw language input.

        Returns:
            A dictionary containing processed data for the model.
        """
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()

        if len(curr_image.shape) == 5: # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        # for k,v in sample.items():
        #     print(k, v.shape)
        image, image_r = torch.chunk(curr_image, 2, dim=0)

        image = self.expand2square(image, tuple(x for x in self.image_processor.image_mean))
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']

        image_tensor = image_tensor.to(self.policy.device, dtype=self.policy.dtype)

        image_r = self.expand2square(image_r, tuple(x for x in self.image_processor.image_mean))
        image_tensor_r = self.image_processor.preprocess(image_r, return_tensors='pt', do_normalize=True, do_rescale=False,
                                              do_center_crop=False)['pixel_values']
        image_tensor_r = image_tensor_r.to(self.policy.device, dtype=self.policy.dtype)

        # print('raw_lang')
        inp = raw_lang
        assert image is not None, 'image must be provided.'
        # first message
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        image = None

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        prompt += " <|endoftext|>"

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state.to(self.policy.device, dtype=self.policy.dtype)
        # print(input_ids.dtype, attn_mask.dtype, image_tensor.dtype, image_tensor_r.dtype, states.dtype)

        data_dict = dict(input_ids=input_ids,
                         attention_mask=attn_mask,
                         images=image_tensor,
                         images_r=image_tensor_r,
                         states=states)

        # print(f"@@@@@@@@@@@@@@@{image_tensor.shape}")
        return data_dict

    def expand2square(self, pil_imgs, background_color):
        batch_size, channels, height, width = pil_imgs.shape
        max_dim = max(height, width)
        expanded_imgs = np.full((batch_size, max_dim, max_dim, channels), background_color, dtype=np.float32)

        if height == width:
            expanded_imgs = pil_imgs.permute(0,2,3,1).cpu().numpy()
        elif height > width:
            offset = (max_dim - width) // 2
            # expanded_imgs[:, :height, offset:offset + width] = pil_imgs
            expanded_imgs[:, :height, offset:offset + width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        else:
            offset = (max_dim - height) // 2
            # expanded_imgs[:, offset:offset + height, :width] = pil_imgs
            expanded_imgs[:, offset:offset + height, :width, :] = pil_imgs.permute(0,2,3,1).cpu().numpy()
        expanded_imgs = torch.tensor(expanded_imgs).to(dtype=pil_imgs.dtype, device=pil_imgs.device) # B H W C
        return expanded_imgs


def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None):
    """
    Evaluates the behavior cloning policy in the deployment environment.

    Args:
        policy: The policy to evaluate.
        deploy_env: The deployment environment.
        policy_config: Configuration dictionary for the policy.
        save_episode: Whether to save the episode data.
        num_rollouts: Number of rollouts to perform.
        raw_lang: Raw language input for the policy.

    Returns:
        None
    """
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    if policy_config["action_head"] == 'act':
        rand_crop_resize = False
        temporal_agg = True
    else:
        rand_crop_resize = True
        temporal_agg = True

    action_dim = policy.config['action_dim']

    policy.policy.eval()

    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"] == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif policy_config["action_head"] == 'transformer_diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env

    query_frequency = policy.config['chunk_size'] / 2 # specify the exact executed action steps, must be smaller than chunk size
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config['chunk_size']

    max_timesteps = int(10000)  # may increase for real-world tasks

    for rollout_id in range(num_rollouts):
        rollout_id += 0
        env.reset(randomize=False)

        print(f"env has reset!")

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],dtype=torch.float16).cuda()
            # print(f'all_time_actions size: {all_time_actions.size()}')

        image_list = []  # for visualization
        robot_state_list = []
        target_action_list = []

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):

                obs = deploy_env.get_observation()

                traj_rgb_np, robot_state = get_obs(obs, stats)
                image_list.append(traj_rgb_np)

                robot_state = torch.from_numpy(robot_state).float().cuda()

                if t % query_frequency == 0:
                    curr_image = torch.from_numpy(traj_rgb_np / 255.0).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize(original_size, antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        policy.policy(**batch, eval=True)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if policy_config['action_head_type'] == "act":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)

                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif policy_config['action_head_type'] == "droid_diffusion":
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        all_actions = policy.policy(**batch, eval=True)
                            
                    if temporal_agg:
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")
                # target_qpos = action

                action = convert_actions(action)
                time5 = time.time()
                ### step the environment
                # ts = env.step(action)
                print(f'step {t}, pred action: {action}')
                action_info = deploy_env.step(action)

                ### for visualization
                robot_state_list.append(robot_state)
                target_action_list.append(action)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(
                        f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()


    return

if __name__ == '__main__':
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'droid_diffusion' # specify the action head type
    policy_config = {
        "model_path": f"/path/to/trained/VLA", # mainly includes the lora weights
        "model_base": f"/path/to/pretrained/VLM", # used for lora merge weights
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
    }
    global im_size
    im_size = 320 # default 480
    raw_lang = 'put the tennis ball on the right side into the tennis bucket'
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # make policy
    policy = llava_pythia_act_policy(policy_config)

    ############################################################################################################
    # This is your own robot environment, you should init a new env object
    deploy_env = None
    ############################################################################################################

    eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang)

