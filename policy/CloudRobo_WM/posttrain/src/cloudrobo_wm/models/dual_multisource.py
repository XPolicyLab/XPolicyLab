import copy
import inspect
import torch
from torch import nn
from .multisource_cloudrobo_wm import MultiSourceCloudRoboWM
from .dual_cooperative_action import ACTION_TOPOLOGY, JG_INDICES, EEF_INDICES, DualCooperativeActionExpert, DualCooperativeMoT, validate_robotwin_action_valid
from .multisource_loss import _action_error_and_weight

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

    def _flow_metadata(self):
        from .dual_action_initialization import dual_action_contract
        return dict(super()._flow_metadata(), action_topology=ACTION_TOPOLOGY, dual_action_contract=dual_action_contract())

    def _validate_checkpoint_payload(self, payload, *, strict_shape):
        if strict_shape and payload.get('history_contract') is not None and not hasattr(self, 'history_config'):
            raise ValueError('History checkpoint requires a matching history model')
        if strict_shape:
            from .dual_action_initialization import validate_dual_checkpoint_contract
            validate_dual_checkpoint_contract(payload, model=self)
        return super()._validate_checkpoint_payload(payload, strict_shape=strict_shape)

    def _multisource_losses(self, values):
        losses = super()._multisource_losses(values)
        with torch.no_grad():
            valid = self._ms_masks['action']
            errors, weights = _action_error_and_weight(values['pred_action'], values['target_action'], valid, self.multisource_field_weights, self.multisource_horizon_weights)
            denominator = weights.sum((1, 2))
            active = denominator > 0
            diffusion = self.train_action_scheduler.training_weight(values['t_action']).reshape(-1)
            parts = []
            for indices in (JG_INDICES, EEF_INDICES):
                indices = list(indices)
                numerator = errors[..., indices].sum((1, 2))
                contribution = numerator / denominator.clamp_min(1e-12) * diffusion
                p = torch.where(valid[..., indices], values['pred_action'][..., indices].float(), 0)
                t = torch.where(valid[..., indices], values['target_action'][..., indices].float(), 0)
                parts.extend(((contribution * active).sum(), active.sum(), (p - t).square().sum(), valid[..., indices].sum()))
            diagnostics = torch.stack(parts).float()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(diagnostics)
            for branch, (total, count, squared, elements) in zip(('joint_gripper', 'eef'), diagnostics.reshape(2, 4)):
                self._multisource_loss_metrics['dual_' + branch + '_contribution'] = self.loss_lambda_action * total / count.clamp_min(1)
                self._multisource_loss_metrics['dual_' + branch + '_mse'] = squared / elements.clamp_min(1)
                self._multisource_loss_metrics['dual_' + branch + '_valid_elements'] = elements
        return losses

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
