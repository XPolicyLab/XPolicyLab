from __future__ import annotations
import copy
import inspect
import torch
from torch import nn
from .multisource_cloudrobo_wm import MultiSourceCloudRoboWM
from .backbone import CloudRoboWMBackbone
from .dual_cooperative_action import ACTION_TOPOLOGY, JG_INDICES, EEF_INDICES, DualCooperativeActionExpert, DualCooperativeMoT, validate_robotwin_action_valid

class DualCooperativeMultiSourceCloudRoboWM(MultiSourceCloudRoboWM):
    action_topology = ACTION_TOPOLOGY

    def _set_multisource_context(self, sample):
        valid = sample['action_valid']
        validate_robotwin_action_valid(valid)
        if valid.shape[1] != 32:
            raise ValueError('dual_cooperative_robotwin_v1 requires H32')
        super()._set_multisource_context(sample)

    def _dual_action_input_hook(self, indices):

        def hook(module, args):
            if not self._ms_active:
                return args
            valid = self._ms_masks['action'][..., list(indices)]
            return (torch.where(valid, args[0], 0), *args[1:])
        return hook

    def _dual_action_embedding_hook(self, branch, indices):

        def hook(module, args, output):
            if not self._ms_active:
                return output
            mask = self._ms_masks['action'][..., list(indices)]
            valid = mask.any(-1)
            tokens = output + getattr(self.multisource_conditioning, 'action_valid_' + branch)(mask.to(output.dtype))
            physical_time = self.multisource_conditioning.time('action_' + branch, self._ms_sample['time_grid']['action'], self._ms_sample['time_valid']['action'], tokens.dtype)
            tokens = (tokens + physical_time) * valid[..., None]
            self._ms_token_valid['action_' + branch] = valid
            if 'action_joint' in self._ms_token_valid and 'action_eef' in self._ms_token_valid:
                self._ms_token_valid['action'] = torch.cat([self._ms_token_valid['action_joint'], self._ms_token_valid['action_eef']], 1)
            return tokens
        return hook

    def _dual_head_audit_hook(self, branch):

        def hook(module, args, output):
            if getattr(self, '_dual_inference_calls', None) is not None:
                self._dual_inference_calls[branch] += 1
            return output
        return hook

    def _sequence_valid(self, branch, length, tpf):
        if branch == 'action':
            raw = self._ms_masks['action']
            valid = torch.cat((raw[..., list(JG_INDICES)].any(-1), raw[..., list(EEF_INDICES)].any(-1)), dim=1)
            if valid.shape[1] != length:
                raise ValueError('Expected packed JG(H), EEF(H) token layout')
            return valid
        return super()._sequence_valid(branch, length, tpf)

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        if args:
            raise ValueError('Dual inference requires explicit keyword arguments for its protocol')
        for key in ('joint_video', 'joint_dino', 'joint_pointmap'):
            if kwargs.get(key) is not True:
                raise ValueError('Dual inference requires joint_video/dino/pointmap=True')
        if kwargs.get('joint_flow', False) or kwargs.get('dynamic_step_skip', False):
            raise ValueError('Dual inference requires joint_flow=False and no step skipping')
        if kwargs.get('text_cfg_scale', 1.0) != 1.0:
            raise ValueError('Dual inference protocol requires text_cfg_scale=1')
        if kwargs.get('num_inference_steps', 4) != 4:
            raise ValueError('Dual inference protocol requires 4 denoising steps')
        kwargs.setdefault('num_inference_steps', 4)
        kwargs.setdefault('action_horizon', 32)
        self._dual_inference_calls = dict(joint=0, eef=0)
        try:
            result = super().infer_action(**kwargs)
            if self._dual_inference_calls != dict(joint=4, eef=4):
                raise RuntimeError('Dual inference did not execute both towers exactly four times')
            valid = kwargs['action_valid'][0].to(result['action'].device).bool()
            result['action'] = torch.where(valid, result['action'], 0)
            self.last_dual_inference_audit = dict(action_topology=ACTION_TOPOLOGY, joint_calls=4, eef_calls=4, joint_denoising_rounds=4, prediction_horizon=32, execution_horizon_contract=32, execution_performed_by_model=False)
            result['dual_action_audit'] = dict(self.last_dual_inference_audit)
            return result
        finally:
            self._dual_inference_calls = None

    def prepare_for_inference(self):
        super().prepare_for_inference()

def enable_dual_cooperative_action(model):
    if isinstance(model, DualCooperativeMultiSourceCloudRoboWM):
        return model
    for handle in model._ms_handles:
        handle.remove()
    model._ms_handles = []
    facade = DualCooperativeActionExpert.from_single(model.action_expert)
    mot = DualCooperativeMoT.from_single(model.mot, facade)
    conditioning = model.multisource_conditioning
    old_valid = conditioning.action_valid
    old_time = conditioning.time_projection['action']
    for branch, indices in (('joint', JG_INDICES), ('eef', EEF_INDICES)):
        projection = copy.deepcopy(old_valid)
        projection.in_features = len(indices)
        projection.weight = nn.Parameter(old_valid.weight[:, list(indices)].detach().clone(), requires_grad=old_valid.weight.requires_grad)
        setattr(conditioning, 'action_valid_' + branch, projection)
        conditioning.time_projection['action_' + branch] = copy.deepcopy(old_time)
    del conditioning.action_valid
    del conditioning.time_projection['action']
    model.__class__ = DualCooperativeMultiSourceCloudRoboWM
    for name in ('video_expert', 'action_expert', 'dit'):
        model._modules.pop(name, None)
        model.__dict__.pop(name, None)
    model.mot = mot
    object.__setattr__(model, 'video_expert', mot.mixtures['video'])
    object.__setattr__(model, 'action_expert', facade)
    object.__setattr__(model, 'dit', mot)
    for branch, module in (('video', model.video_expert.patch_embedding), ('flow', model.flow_patch_embedding), ('pointmap', model.pt_patch_embedding), ('dino', model.dino_embedder)):
        model._ms_handles.append(module.register_forward_hook(model._embedding_hook(branch)))
    for branch, tower, indices in (('joint', facade.joint, JG_INDICES), ('eef', facade.eef, EEF_INDICES)):
        model._ms_handles.append(tower.action_encoder.register_forward_pre_hook(model._dual_action_input_hook(indices)))
        model._ms_handles.append(tower.action_encoder.register_forward_hook(model._dual_action_embedding_hook(branch, indices)))
        model._ms_handles.append(tower.head.register_forward_hook(model._dual_head_audit_hook(branch)))
    return model
