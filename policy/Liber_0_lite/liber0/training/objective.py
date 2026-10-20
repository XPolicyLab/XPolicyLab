import torch
from torch import nn

from ..batching import prepare_condition, joint_forward
from .scheduler import TrainingScheduler


class TrainingModel(nn.Module):
    def __init__(self, model, *, video_shift=1.0, action_shift=5.0,
                 video_weight=0.5, action_weight=1.0):
        super().__init__()
        self.model = model
        self.video_schedule = TrainingScheduler(1000, video_shift)
        self.action_schedule = TrainingScheduler(1000, action_shift)
        self.video_weight = video_weight
        self.action_weight = action_weight

    def forward(self, sample):
        model = self.model
        video = sample['video'].to(model.device, model.sample_dtype)
        actions = sample['action'].to(model.device, model.sample_dtype)
        proprio = sample['proprio'].to(model.device, model.sample_dtype)
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        reference, target = video[:, :, 0], video[:, :, -1]
        clean = model._patchify(target, model.patch_size)
        batch = clean.shape[0]
        video_noise = torch.randn_like(clean) * model.noise_scale
        video_t = self.video_schedule.sample_training_t(batch, model.device, clean.dtype)
        noisy_video = self.video_schedule.add_noise(clean, video_noise, video_t)
        video_target = self.video_schedule.training_target(clean, video_noise, video_t)
        video_sigma = video_t / self.video_schedule.num_train_timesteps
        action_noise = torch.randn_like(actions)
        action_t = self.action_schedule.sample_training_t(batch, model.device, actions.dtype)
        noisy_actions = self.action_schedule.add_noise(actions, action_noise, action_t)
        action_target = self.action_schedule.training_target(actions, action_noise, action_t)
        action_sigma = action_t / self.action_schedule.num_train_timesteps
        prompts = sample['prompt']
        if len(prompts) != batch:
            raise ValueError('Expected one prompt per sample.')
        if batch == 1:
            pred_x0, pred_action, _ = model._joint_forward_single(
                prompts[0], (reference,), target.shape[-2], target.shape[-1], proprio,
                noisy_video, 1 - video_sigma, noisy_actions, 1 - action_sigma)
        else:
            condition = prepare_condition(model, prompts, reference, clean.shape[1])
            pred_x0, pred_action = joint_forward(model, condition,
                model._patchify(reference, model.patch_size), proprio, noisy_video,
                noisy_actions, 1 - video_sigma, 1 - action_sigma)
        velocity = (noisy_video - pred_x0) / video_sigma[:, None, None].clamp_min(model.timestep_eps)
        video_loss = (velocity.float() - video_target.float()).square().mean((1, 2))
        video_loss = (video_loss * self.video_schedule.training_weight(video_t)).mean()
        valid = ~sample['action_is_pad'].to(model.device).bool().unsqueeze(-1).expand_as(actions)
        error = (pred_action.float() - action_target.float()).square()
        action_loss = (error * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        action_loss = (action_loss * self.action_schedule.training_weight(action_t)).mean()
        total = self.video_weight * video_loss + self.action_weight * action_loss
        return total, {'loss_video': (self.video_weight * video_loss).detach(),
                       'loss_action': (self.action_weight * action_loss).detach()}
