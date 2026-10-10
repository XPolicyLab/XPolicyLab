"""In-process inference engine."""
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from .loading import load_model
from .preprocessing import build_norm, denormalize, pack_images
from .batching import infer_actions


class Policy:
    def __init__(self, checkpoint_root, assets, device, weights_file, num_inference_steps):
        cfg = OmegaConf.load(checkpoint_root / 'config.yaml')
        model = OmegaConf.to_container(cfg.model, resolve=False)
        for name in ('_target_', 'model_path', 'source_path'):
            del model[name]
        inference = OmegaConf.to_container(cfg.inference, resolve=True)
        self.robotwin_camera_layout = inference['robotwin_camera_layout']
        if self.robotwin_camera_layout != 'four_grid_288x384':
            raise ValueError('Unsupported image layout.')
        proc = inference['processor']
        if proc['norm_default_mode'] != 'z-score' or proc['use_stepwise_action_norm']:
            raise ValueError('Unsupported normalization configuration.')
        transforms = proc['val_transforms']
        if (len(transforms) != 2 or set(transforms[0]) != {'_target_'}
                or transforms[0]['_target_'].rsplit('.', 1)[-1] != 'ToTensor'
                or set(transforms[1]) != {'_target_', 'size'}
                or transforms[1]['_target_'] != 'torchvision.transforms.Resize'):
            raise ValueError('Unsupported image transforms.')
        self.resize = transforms[1]['size']
        self.image_crop_scale = inference['image_crop_scale']
        self.action_horizon = int(inference['action_horizon'])
        self.steps = int(num_inference_steps)
        self.device = torch.device(device)
        with (checkpoint_root / 'dataset_stats.json').open() as handle:
            norm = build_norm(json.load(handle))
        self.norm = {key: value.to(self.device) if key.startswith('state') else value for key, value in norm.items()}
        self.model = load_model(model, assets, checkpoint_root / weights_file, device)

    @torch.no_grad()
    def infer_action_chunk(self, observation, instruction, request_seed):
        images = [observation[key] for key in ('head_camera_rgb', 'left_camera_rgb', 'right_camera_rgb', 'visual_cue_rgb')]
        image = pack_images(images, self.resize, self.image_crop_scale).to(self.device, torch.bfloat16)
        state = torch.from_numpy(np.asarray(observation['joint_action_vector'], dtype=np.float32).reshape(-1).copy()).to(self.device)
        state = torch.clamp(state * self.norm['state_scale'] + self.norm['state_offset'], -5, 5).unsqueeze(0).to(torch.bfloat16)
        output = self.model.infer_action(
            prompt="A video recorded from a robot's point of view executing the following instruction: " + instruction,
            input_image=image, proprio=state, action_horizon=self.action_horizon,
            num_inference_steps=self.steps, seed=request_seed, rand_device='cpu',
        )['action'].detach().to(device='cpu', dtype=torch.float32)
        return denormalize(output, self.norm['action_scale'], self.norm['action_offset']).numpy()

    @torch.no_grad()
    def infer_action_batch(self, observations, instructions, request_seeds):
        if not observations or len(observations) != len(instructions) or len(observations) != len(request_seeds):
            raise ValueError('Inconsistent inference batch lengths.')
        if len(observations) == 1:
            return self.infer_action_chunk(observations[0], instructions[0], request_seeds[0])[None]
        keys = ('head_camera_rgb', 'left_camera_rgb', 'right_camera_rgb', 'visual_cue_rgb')
        images = torch.cat([pack_images([obs[key] for key in keys], self.resize, self.image_crop_scale)
                            for obs in observations]).to(self.device, torch.bfloat16)
        states = np.stack([np.asarray(obs['joint_action_vector'], dtype=np.float32).reshape(-1)
                           for obs in observations])
        states = torch.from_numpy(states).to(self.device)
        states = torch.clamp(states * self.norm['state_scale'] + self.norm['state_offset'], -5, 5).to(torch.bfloat16)
        prompts = ["A video recorded from a robot's point of view executing the following instruction: " + instruction
                   for instruction in instructions]
        actions = infer_actions(self.model, prompts, images, states, self.action_horizon, request_seeds, self.steps)
        return denormalize(actions, self.norm['action_scale'], self.norm['action_offset']).numpy()
