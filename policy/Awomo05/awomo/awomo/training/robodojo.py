# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Awomo-0.5I RoboDojo SFT: conditioning, checkpoint loading and weighted losses."""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from awomo.awomo.model import ActionPatchModel, VideoVAE
from awomo.awomo.model_vl import VLPromptEncoder
from awomo.awomo.factory import load_checkpoint
from awomo.awomo.datasets.robodojo.fixed_budget_dense import FixedBudgetDenseBatchSampler
from awomo.awomo.training.trainer import Trainer
from awomo.awomo.training.distributed.fsdp import setup_fsdp
from awomo.awomo.training.frozen_state_ema import CompleteFrozenStateEMA
from awomo.awomo.training.gradient_utils import clip_grad_norm
from awomo.awomo.training.losses import gripper_train_step

class RoboDojoTrainer(Trainer):
    train_step = gripper_train_step
    _setup_fsdp = setup_fsdp

    @property
    def _model_cfg(self):
        return self.cfg.action_patch

    def _compute_theoretical_flops_per_step(self) -> None:
        self._flops_per_micro_batch = float('nan')
        self._peak_flops_per_gpu = 1.0
        logging.warning('MFU unavailable for RoboDojo multiview/history/VLM; report measured step time/throughput instead')


    def _ensure_encoders_on_device(self) -> None:
        self.vae.to(device=self.device, dtype=torch.bfloat16)

    def _encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._ensure_encoders_on_device()
        vlm = {key: value.to(self.device, non_blocking=True) for (key, value) in batch['vlm'].items()}
        with torch.no_grad():
            context, text_mask = self.vl_encoder.forward_vlm(vlm)
        with torch.no_grad():
            video = batch['video'].to(self.device, dtype=torch.bfloat16, non_blocking=True)
            (b, views, channels, frames, height, width) = video.shape
            dataset = self.dataloader.dataset
            expected_views = 3
            if dataset.image_layout == 'hero_top':
                expected_views = 1
            assert (views, channels, frames, height, width) == (expected_views, 3, 2, *dataset.video_size)
            (refs, targets) = ([], [])
            for view in range(views):
                refs.append(self.vae.encode(video[:, view, :, :1])[:, :, 0].flatten(2).transpose(1, 2))
            for view in range(views):
                targets.append(self.vae.encode(video[:, view, :, -1:])[:, :, 0].flatten(2).transpose(1, 2))
            reference = torch.cat(refs, 1)
            target = torch.cat(targets, 1)
            ref_ids = torch.cat([self.action_patch_model.image_ids(height // 16, width // 16, 10 + view, self.device) for view in range(views)])
            target_ids = torch.cat([self.action_patch_model.image_ids(height // 16, width // 16, view, self.device) for view in range(views)])
            history = batch['history_frames'].to(self.device, dtype=torch.bfloat16, non_blocking=True)
            valid = batch['history_valid_mask'].to(self.device, non_blocking=True)
            assert history.shape == (b, 20, 3, height, width) and valid.shape == (b, 20)
            pooled = history.new_zeros(b * 20, 16, 128)
            selected = history.reshape(b * 20, 3, height, width)[valid.reshape(-1)]
            encoded = []
            for start in range(0, len(selected), 16):
                latent = self.vae.encode(selected[start:start + 16].unsqueeze(2))[:, :, 0]
                encoded.append(F.adaptive_avg_pool2d(latent.float(), (4, 4)).to(latent.dtype).flatten(2).transpose(1, 2))
            if encoded:
                pooled[valid.reshape(-1)] = torch.cat(encoded)
            history_ids = torch.cat([self.action_patch_model.image_ids(4, 4, slot - 20, self.device) for slot in range(20)])
            reference = torch.cat([pooled.reshape(b, 320, 128), reference], 1)
            reference_ids = torch.cat([history_ids, ref_ids])
            condition_mask = torch.cat([valid.repeat_interleave(16, dim=1), torch.ones(b, target.shape[1], device=self.device, dtype=torch.bool)], 1)
            action = batch['action'].to(self.device, dtype=torch.bfloat16, non_blocking=True)
            state = batch['proprio'][:, 0].to(self.device, dtype=torch.bfloat16, non_blocking=True)
        return dict(reference=reference, target=target, action=action, context=context, text_mask=text_mask,
                    state=state, reference_ids=reference_ids, target_ids=target_ids, condition_mask=condition_mask)

    def _extend_train_log_dict(self, log_dict: dict[str, float], group_loss_mean: float, grad_norm: float, step_time: float, mfu: float) -> None:
        del group_loss_mean, grad_norm, mfu
        log_dict.pop('train/mfu', None)
        log_dict['train/samples_per_second'] = self.global_batch_size / step_time

    def _build_models(self, model_cfg: Any, model_cfg_inner: Any, *, load_pretrained_weights: bool=True) -> None:
        assert self.cfg.checkpoint.warm_start_from is None
        assert float(self.cfg.ema.decay) == 0.999
        del model_cfg, load_pretrained_weights
        config = model_cfg_inner
        assert config.dit.action_dim == config.dit.max_state_dim == 80
        assert config.action_patch_scale == config.loss_lambda_action == 1.0
        assert not config.self_flow_enabled
        transformer = Flux2Transformer2DModel.from_pretrained(config.dit.pretrained_path, torch_dtype=torch.float32)
        self.model = ActionPatchModel(transformer, action_dim=80, state_dim=80, action_scale=1.0)
        self.__dict__['action_patch_model'] = self.model
        self.lora_enabled = False
        self.loss_lambda_video = float(config.loss_lambda_video)
        vae = AutoencoderKLFlux2.from_pretrained(config.encoders.vae_path, torch_dtype=torch.bfloat16)
        self.__dict__['vae'] = VideoVAE(vae)
        encoder_kwargs = config.vlm.to_dict()
        encoder_kwargs['lora'] = dict(encoder_kwargs['lora'])
        self.__dict__['vl_encoder'] = VLPromptEncoder(config.encoders.text_encoder_path, device=str(self.device), torch_dtype=torch.bfloat16, **encoder_kwargs)
        self.vl_encoder.train()
        assert all((('lora_' in name) == parameter.requires_grad for (name, parameter) in self.vl_encoder.named_parameters()))
        self.__dict__['text_encoder'] = self.vl_encoder
        assert len(self.model.state_dict()) == 171, 'Preserve all DiT/state parameter keys'
        initialization = Path(self.cfg.checkpoint.base_checkpoint)
        proof = load_checkpoint(self.model, self.vl_encoder, initialization,
                                         verify_checksums=int(os.environ.get('RANK', '0')) == 0, allow_base_only=True)
        assert proof['model_keys'] == 171 and proof['lora_keys'] == 504
        assert self.model.transformer.proj_out.bias is None
        self.model.proprio_encoder.requires_grad_(False)
        self.vl_encoder.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)
        assert all((name.startswith('transformer.') == p.requires_grad for (name, p) in self.model.named_parameters()))
        self.__dict__['initialization_evidence'] = proof
        logging.info('BASE_CHECKPOINT_LOADED %s', json.dumps(proof))
        if int(os.environ['LOCAL_RANK']) == 0:
            print('SFT_BASE_LOADED ' + json.dumps(self.initialization_evidence), flush=True)

    def setup(self) -> None:
        assert self.cfg.data.dataloader.dataset.is_training_set
        assert self.cfg.data.dataloader.dataset.val_set_proportion == 0
        assert not self.cfg.validation.enabled and self.cfg.validation.every_steps == 0
        super().setup()
        assert self.global_batch_size == self.local_batch_size * self.world_size
        dataset = self.dataloader.dataset
        assert dataset.action_dim == dataset.state_dim == 80
        assert dataset.image_layout == 'hero_top' and dataset.image_color_order == 'bgr'
        assert dataset.video_size == (320, 288)
        restore = self.cfg.checkpoint.restore_from
        if restore:
            checkpoint = Path(restore)
            if not checkpoint.is_absolute():
                checkpoint = Path(self.cfg.checkpoint.checkpoint_dir) / checkpoint
            payload = torch.load(checkpoint / 'vl_sft_state.pt', map_location='cpu', weights_only=False)
            assert payload['frozen_vlm'] and payload['vl_text_layout'] == 'prefix_v2'
            self.vl_encoder.load_lora_state_dict(payload['vl_lora'])
        logging.info('SFT_MODEL_READY: trainable DiT; frozen state/VLM/VAE; action_dim80; complete EMA')
        assert self.world_size == 32 and self.global_batch_size == 256 and (self.local_batch_size == 8)
        assert isinstance(self.sampler, FixedBudgetDenseBatchSampler) and len(self.dataloader) == 6755
        assert self.step_scheduler.grad_acc_steps == 1 and self.dataloader.dataset.window_stride == 1
        assert self.cfg.fsdp.dp_size == 32 and self.cfg.fsdp.dp_replicate_size == 4
        assert self.cfg.gripper_experiment.loss_multiplier == 2.0 and self.cfg.gripper_experiment.sampling == 'uniform'
        assert self.cfg.gripper_experiment.stride == 1
        assert len(self.ema) == 171 and len(self.model.state_dict()) == 171
        assert self.model.transformer.proj_out.bias is None
        if int(os.environ['LOCAL_RANK']) == 0:
            print('SFT_READY ' + json.dumps({'world_size': 32, 'global_batch': 256, 'local_batch': 8, 'steps': 6755, 'stride': 1, 'windows': len(self.dataloader.dataset), 'padding': 4942, 'loss_multiplier': 2.0, 'model_keys': 171, 'ema_keys': 171}), flush=True)

    def _after_optimizer_setup(self) -> None:
        assert self.ema is not None and len(self.ema) == 169
        self.__dict__['ema'] = CompleteFrozenStateEMA(self.ema, self.model)
        assert len(self.ema) == 171
        assert 'ema' in self.__dict__['__state_tracked']
        assert not hasattr(self, 'vl_optimizer') and (not hasattr(self, 'vl_ema'))

    def _clip_gradients(self) -> float:
        return clip_grad_norm(self.clip_grad_max_norm, [self.model], device_mesh=self.device_mesh, foreach=True)

    def _run_train_batch_group(self, batch_group: list[dict[str, Any]], current_step: int, before_clip_callback: Any=None) -> tuple[dict[str, float], float, float]:
        (metrics, norm, seconds) = super()._run_train_batch_group(batch_group, current_step, before_clip_callback)
        if self._model_cfg.bsmntw_enabled:
            metrics.update(self._average_log_scalars_across_ranks(self._bsmntw_log_scalars))
        metrics.update(self._average_log_scalars_across_ranks(self._gripper_log_scalars))
        return (metrics, norm, seconds)

    def run_train_validation_loop(self):
        """Retry final EMA export when resuming an already completed run."""
        restore = self.cfg.checkpoint.get('restore_from')
        export = self.cfg.get('checkpoint.hf_export', None)
        max_steps = int(self.cfg.step_scheduler.max_steps)
        if restore and int(self.step_scheduler.step) >= max_steps and export is not None and export.get('enabled', True):
            checkpoint = Path(restore)
            if not checkpoint.is_absolute():
                checkpoint = Path(self.cfg.checkpoint.checkpoint_dir) / checkpoint
            if not (checkpoint / '.complete').is_file():
                raise ValueError('Cannot export an incomplete native checkpoint')
            self.__dict__['_inference_checkpoint_path'] = str(checkpoint)
            self.save_hf_checkpoint(int(self.step_scheduler.epoch), max_steps - 1)
        return super().run_train_validation_loop()

    def save_hf_checkpoint(self, epoch: int, step: int) -> None:
        """Write the final EMA in the inference format after the native save completes."""
        del epoch
        if step + 1 != int(self.cfg.step_scheduler.max_steps):
            return
        from awomo.awomo.training.checkpoint.inference import load_ema_snapshot, save_inference_checkpoint
        from awomo.awomo.training.trainer import _broadcast_bool_from_rank0
        self.checkpointer.async_wait()
        error = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                state = load_ema_snapshot(self._inference_checkpoint_path, self.model.state_dict())
                destination = Path(self.cfg.checkpoint.checkpoint_dir).parent / 'inference' / f'step_{step + 1}'
                save_inference_checkpoint(destination, state, self.action_patch_model.transformer.config,
                                          self.vl_encoder, self.cfg, allow_existing=True)
                logging.info('INFERENCE_CHECKPOINT_SAVED %s', destination)
            except Exception as exc:
                error = exc
        if not _broadcast_bool_from_rank0(error is None):
            if error is not None:
                raise error
            raise RuntimeError('Rank 0 failed to save the final inference checkpoint')

    def _save_checkpoint_artifacts(self, checkpoint_path: str) -> None:
        frozen_lora = {name: parameter.detach().cpu().clone() for (name, parameter) in self.vl_encoder.named_parameters() if 'lora_' in name}
        assert len(frozen_lora) == 504 and all((value.dtype == torch.float32 for value in frozen_lora.values()))
        payload = dict(vl_lora=self.vl_encoder.lora_state_dict(), ema=frozen_lora, vl_text_layout='prefix_v2', frozen_vlm=True)
        torch.save(payload, Path(checkpoint_path) / 'vl_sft_state.pt')
        self.__dict__['_inference_checkpoint_path'] = checkpoint_path
        metadata = dict(vl_text_layout='prefix_v2', action_dim=80, extra_padding48='zero_before_transformer', bsmntw_enabled=bool(self._model_cfg.bsmntw_enabled), self_flow_enabled=False, action_scale=1.0, output_bias_in_state_dict=False, trainable_scope='dit_no_output_bias', initialization=self.initialization_evidence, training_code=os.environ.get('GIT_COMMIT'), source_digest=os.environ.get('SOURCE_DIGEST'), resolved_config=self.cfg.to_yaml_dict())
        (Path(checkpoint_path) / 'robodojo_sft_contract.json').write_text(json.dumps(metadata, indent=2) + '\n')
        path = Path(checkpoint_path) / 'robodojo_sft_contract.json'
        metadata = json.loads(path.read_text())
        metadata['gripper_experiment'] = self.cfg.gripper_experiment.to_dict()
        path.write_text(json.dumps(metadata, indent=2) + '\n')
