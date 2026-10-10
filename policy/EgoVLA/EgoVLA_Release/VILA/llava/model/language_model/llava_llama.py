#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy
from llava.model.utils.packing import set_seqlens_in_batch
from llava.utils import distributed as dist

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

from dataclasses import dataclass, field
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

@dataclass
class HOILMOutputWithPast(ModelOutput):

  loss: Optional[torch.FloatTensor] = None
  recon_loss:  Optional[torch.FloatTensor] = None
  ee_l2_loss:  Optional[torch.FloatTensor] = None
  ee_2d_l2_loss:  Optional[torch.FloatTensor] = None
  ee_rot_loss:  Optional[torch.FloatTensor] = None
  hand_l2_loss:  Optional[torch.FloatTensor] = None
  hand_kp_loss: Optional[torch.FloatTensor] = None
  kl_loss: Optional[torch.FloatTensor] = None

  raw_action_labels:  Optional[torch.FloatTensor] = None
  raw_action_masks:  Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  prediction: torch.FloatTensor = None
  past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"
    def __init__(
      self,
      invalid_token_idx: int= 31733, # 32000 - 256 - 1 - 10
      invalid_token_weight: float=1.0,
      action_token_idx: int= 31733, # 32000 - 256 - 1 - 10
      kl_weight: float=1e-3,
      action_output_dim: int=2,
      proprio_size: int=16,
      use_proprio: bool=True,
      loss_use_l1: bool=False,
      sep_proprio: bool=False,
      sep_query_token: bool=False,
      cvae_hidden_size: int=256,
      cvae_latent_dim: int=256,
      action_output_ee_dim: int=None,
      action_output_ee_2d_dim: int=0,
      action_output_ee_rot_dim: int=0,
      action_output_hand_dim: int=None,
      next_token_loss_coeff: float=1,
      ee_loss_coeff: float=1,
      ee_2d_loss_coeff: float=1,
      ee_rot_loss_coeff: float=1,
      hand_kp_loss_coeff: float=0,
      use_movement_mask: bool=False,
      ee_movement_loss_weight: float=0,
      ee_rot_representation: str = "",
      hand_loss_coeff: float=1,
      hand_loss_dim: int=15,
      input_placeholder_start_token_idx: int=31732,
      input_placeholder_end_token_idx: int=31712,
      merge_hand: bool=False,
      traj_decoder_type: str = "",
      pred_head: bool=False,
      **kwargs
    ):
      super().__init__(**kwargs)
      self.invalid_token_idx = invalid_token_idx

      self.input_placeholder_start_token_idx = input_placeholder_start_token_idx
      self.input_placeholder_end_token_idx = input_placeholder_end_token_idx

      self.invalid_token_weight = invalid_token_weight
      self.action_token_idx = action_token_idx
      self.kl_weight = kl_weight
      self.action_output_dim = action_output_dim
      self.proprio_size = proprio_size
      self.use_proprio = use_proprio
      self.loss_use_l1 = loss_use_l1
      self.sep_proprio = sep_proprio
      self.sep_query_token = sep_query_token
      self.cvae_hidden_size = cvae_hidden_size
      self.cvae_latent_dim = cvae_latent_dim
      self.action_output_ee_dim = action_output_ee_dim
      self.action_output_ee_2d_dim = action_output_ee_2d_dim
      self.action_output_ee_rot_dim = action_output_ee_rot_dim
      self.action_output_hand_dim = action_output_hand_dim
      
      self.next_token_loss_coeff = next_token_loss_coeff
      self.ee_loss_coeff = ee_loss_coeff
      self.ee_2d_loss_coeff = ee_2d_loss_coeff
      self.ee_rot_loss_coeff = ee_rot_loss_coeff
      self.hand_loss_coeff = hand_loss_coeff
      self.hand_loss_dim = hand_loss_dim
      self.hand_kp_loss_coeff = hand_kp_loss_coeff

      self.use_movement_mask = use_movement_mask
      self.ee_movement_loss_weight = ee_movement_loss_weight

      self.ee_rot_representation = ee_rot_representation

      self.merge_hand = merge_hand

      self.traj_decoder_type = traj_decoder_type
      self.pred_head = pred_head


from human_plan.utils.mano.model import (
  mano_left,
  mano_right
)
from human_plan.utils.mano.forward import mano_forward

from .geodesic_loss import geodesic_loss
from pytorch3d.transforms import so3_exp_map, axis_angle_to_matrix
from pytorch3d.transforms import axis_angle_to_quaternion, matrix_to_axis_angle

from .rotation_convert import rot6d_to_rotmat, batch_axis2matrix, batch_matrix2axis
# FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # For Delta
        raw_proprio_inputs: Optional[torch.FloatTensor] = None,
        raw_proprio_inputs_2d: Optional[torch.FloatTensor] = None,
        raw_proprio_inputs_3d: Optional[torch.FloatTensor] = None,
        raw_proprio_inputs_rot: Optional[torch.FloatTensor] = None,
        raw_proprio_inputs_handdof: Optional[torch.FloatTensor] = None,
        raw_proprio_inputs_hand_finger_tip: Optional[torch.FloatTensor] = None,
        # 
        raw_action_labels: Optional[torch.FloatTensor] = None,
        raw_action_masks: Optional[torch.Tensor] = None,
        raw_ee_movement_masks: Optional[torch.Tensor] = None,
        packing: bool = True,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        inference: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        packing = False
        if packing and self.training and not dpo_forward:
            if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
            set_seqlens_in_batch(seqlens_in_batch)

            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        if self.training and getattr(self.config, "time_token_ids", []):
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        # Loss rescale for SP & DP loss match
        if dist.size() > 1:
            loss_weight = calculate_loss_weight(labels)
            outputs.loss = outputs.loss * loss_weight

        assert self.config.traj_decoder_type != "dummy"
        # output_mask = (new_labels != IGNORE_INDEX)
        output_mask = torch.bitwise_and(
          labels >= self.config.input_placeholder_end_token_idx,
          labels <= self.config.input_placeholder_start_token_idx,
        )

        traj_decoder_mask = torch.where(
            output_mask,
            torch.zeros_like(attention_mask),
            attention_mask,
        )

        action_output = outputs.hidden_states[-1][output_mask]

        if inference:
            action_output = self.get_traj_decoder().inference(
              action_output,
              {
                "proprio": raw_proprio_inputs,
                "proprio_2d": raw_proprio_inputs_2d,
                "proprio_3d": raw_proprio_inputs_3d,
                "proprio_rot": raw_proprio_inputs_rot,
                "proprio_handdof": raw_proprio_inputs_handdof,
                "proprio_hand_finger_tip": raw_proprio_inputs_hand_finger_tip,
                "action_label": raw_action_labels
              },
              memory=outputs.hidden_states[-1],
              # memory_mask=attention_mask,
              memory_mask=traj_decoder_mask,
              return_kl=True
            )
        else:
          action_output = self.get_traj_decoder()(
            action_output,
            {
              "proprio": raw_proprio_inputs,
              "proprio_2d": raw_proprio_inputs_2d,
              "proprio_3d": raw_proprio_inputs_3d,
              "proprio_rot": raw_proprio_inputs_rot,
              "proprio_handdof": raw_proprio_inputs_handdof,
              "proprio_hand_finger_tip": raw_proprio_inputs_hand_finger_tip,
              "action_label": raw_action_labels
            },
            memory=outputs.hidden_states[-1],
            # memory_mask=attention_mask,
            memory_mask=traj_decoder_mask,
          )

        preds_ = []
        if raw_action_labels is not None:
          l2_loss_list = []
          ee_l2_loss_list = []
          ee_2d_l2_loss_list = []
          ee_rot_loss_list = []
          hand_l2_loss_list = []
          KLD_list = []
          KLD_loss_list = []

          hand_kp_loss_list = []
        
        raw_action_label_list = []
        raw_action_mask_list = []
        
        outputs.prediction = action_output["pred"]
        
        label_index_multipler = 1
        if self.config.sep_query_token:
           label_index_multipler = 3

        prediction = action_output["pred"]

        
        if "KLD" in action_output:
          KLD = action_output["KLD"]
 

        if raw_action_labels is not None:
          raw_label = raw_action_labels
          raw_label_mask = raw_action_masks

          ### compute losses
          assert self.config.action_output_ee_dim is not None

          start_idx_2d, end_idx_2d = 0, self.config.action_output_ee_2d_dim
          ##### EE 2D
          if self.config.action_output_ee_2d_dim > 0:
            ee_2d_l2_loss_single = ((
                raw_label[..., :self.config.action_output_ee_2d_dim] - \
                prediction[..., :self.config.action_output_ee_2d_dim]
            ) ** 2)
            raw_label_mask_2d_single = raw_label_mask[..., :self.config.action_output_ee_2d_dim]
            if self.config.merge_hand:
              ee_2d_l2_loss_single = ee_2d_l2_loss_single.reshape(-1, 2, self.config.action_output_ee_2d_dim // 2)
              raw_label_mask_2d_single = raw_label_mask_2d_single.reshape(-1, 2, self.config.action_output_ee_2d_dim // 2)
            raw_label_mask_2d_single = raw_label_mask_2d_single[..., 0]
            ee_2d_l2_loss_single = ee_2d_l2_loss_single.sum(-1)[raw_label_mask_2d_single].mean()

          ###### EE 3D & Hand DOF
          ee_3d_label = raw_label[
            ..., 4:4 + 6
          ]
          ee_3d_prediction = prediction[
            ..., self.config.action_output_ee_2d_dim:self.config.action_output_ee_2d_dim + self.config.action_output_ee_dim
          ]

          if self.config.loss_use_l1:
            ee_l2_loss_single = (torch.abs(
              ee_3d_label - \
              ee_3d_prediction
            ))
          else:
            ee_l2_loss_single = ((
              ee_3d_label - \
              ee_3d_prediction
            ) ** 2)
          hand_label_component = raw_label[
            ..., 4 + 6:-6
          ]
          hand_prediction_component = prediction[
            ..., self.config.action_output_ee_2d_dim + self.config.action_output_ee_dim:-self.config.action_output_ee_rot_dim
          ]

          hand_l2_loss_single = ((
            hand_label_component - \
            hand_prediction_component
          ) ** 2)

          raw_label_mask_3d_single = raw_label_mask[
            ..., 4 : 4 + 6
          ]
          raw_label_mask_hand_single = raw_label_mask[
            ..., 4 + 6 :-6
          ]

          if raw_ee_movement_masks is not None:
            ee_movement_masks = raw_ee_movement_masks
            ee_movement_masks[torch.abs(ee_movement_masks) < 1e-5] = self.config.ee_movement_loss_weight
            ee_movement_masks = ee_movement_masks.reshape(-1, 2, 1)

            # batch_size, prediction_length = prediction.shape
            sequence_len = prediction.shape[0] // ee_movement_masks.shape[0]
            ee_movement_masks = ee_movement_masks.repeat(
                sequence_len, 1, 1
            )
      
          # if self.config.merge_hand:
          ee_l2_loss_single = ee_l2_loss_single.reshape(-1, 2, self.config.action_output_ee_dim // 2)
          hand_l2_loss_single = hand_l2_loss_single.reshape(-1, 2, hand_l2_loss_single.shape[-1] // 2)
          hand_l2_loss_single = hand_l2_loss_single[..., :self.config.hand_loss_dim]

          if self.config.use_movement_mask:
            ee_l2_loss_single = (
                ee_l2_loss_single * ee_movement_masks#.unsqueeze(-1)
            )
            hand_l2_loss_single = (
                hand_l2_loss_single * ee_movement_masks#.unsqueeze(-1)
            )

          raw_label_mask_3d_single = raw_label_mask_3d_single.reshape(-1, 2, self.config.action_output_ee_dim // 2)
          raw_label_mask_hand_single = raw_label_mask_hand_single.reshape(-1, 2, self.config.action_output_hand_dim // 2)

          ee_3d_label = ee_3d_label.reshape(-1, 2, self.config.action_output_ee_dim // 2)
          ee_3d_prediction = ee_3d_prediction.reshape(-1, 2, self.config.action_output_ee_dim // 2)

          hand_prediction_component = hand_prediction_component.reshape(-1, 2, self.config.action_output_hand_dim // 2)
          hand_label_component = hand_label_component.reshape(-1, 2, self.config.action_output_hand_dim // 2)


          raw_label_mask_3d_single = raw_label_mask_3d_single[..., 0]
          ee_l2_loss_single = ee_l2_loss_single.sum(-1)[raw_label_mask_3d_single].mean()

          raw_label_mask_hand_single = torch.any(raw_label_mask_hand_single, dim=-1)
          hand_l2_loss_single = hand_l2_loss_single.sum(-1)[raw_label_mask_hand_single].mean()

          ##### EE Rot
          # if self.config.action_output_ee_rot_dim > 0:
          if self.config.ee_rot_representation == "quat":
            import torch.nn.functional as F
            raw_q_pred = prediction[..., -self.config.action_output_ee_rot_dim:]
            raw_q_label = raw_label[..., -self.config.action_output_ee_rot_dim:]
            # if self.config.merge_hand:
            raw_q_pred = raw_q_pred.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)
            raw_q_label = raw_q_label.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)

            q_pred = axis_angle_to_quaternion(raw_q_pred)
            q_label = axis_angle_to_quaternion(raw_q_label)

            q_pred = F.normalize(q_pred, p=2, dim=-1)
            q_label = F.normalize(q_label, p=2, dim=-1)
            # Compute the dot product between the predicted and true quaternions
            dot_product = torch.abs(torch.sum(q_pred * q_label, dim=-1))
            ee_rot_loss_single = 1.0 - dot_product

            raw_label_mask_rot_single = raw_label_mask[..., -self.config.action_output_ee_rot_dim:]
            raw_label_mask_rot_single = raw_label_mask_rot_single.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)

            raw_label_mask_rot_single = raw_label_mask_rot_single[..., 0]

            ee_rot_loss_single = ee_rot_loss_single[raw_label_mask_rot_single].mean()
          elif self.config.ee_rot_representation == "rotvec":
            assert False
            q_pred = prediction[..., -self.config.action_output_ee_rot_dim:]
            q_label = raw_label[..., -self.config.action_output_ee_rot_dim:]
            raw_label_mask_rot_single = raw_label_mask[..., -self.config.action_output_ee_rot_dim:]
            if self.config.merge_hand:
              q_pred = q_pred.reshape(-1, self.config.action_output_ee_rot_dim // 2)
              q_label = q_label.reshape(-1, self.config.action_output_ee_rot_dim // 2)
              raw_label_mask_rot_single = raw_label_mask_rot_single.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)
                
            rot_mat_pred = so3_exp_map(q_pred)
            rot_mat_label = so3_exp_map(q_label)
            ee_rot_loss_single = geodesic_loss(rot_mat_pred, rot_mat_label)

            if self.config.merge_hand:
              ee_rot_loss_single = ee_rot_loss_single.reshape(-1, 2)

            raw_label_mask_rot_single = raw_label_mask_rot_single[..., 0]
            ee_rot_loss_single = ee_rot_loss_single[raw_label_mask_rot_single].mean()

            ee_rot_loss_list.append(ee_rot_loss_single)
          elif self.config.ee_rot_representation == "rot6d":
            raw_q_pred = prediction[
              ..., -self.config.action_output_ee_rot_dim:
            ]
            raw_q_label = raw_label[
              ..., 4 + 6 + 30 : 
            ]
            raw_q_pred = raw_q_pred.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)
            raw_q_label = raw_q_label.reshape(-1, 2, 6 // 2)

            pred_rotmat = rot6d_to_rotmat(raw_q_pred).view(-1, 2, 3, 3)
            label_rotmat = axis_angle_to_matrix(raw_q_label)
            ee_rot_loss_single = (pred_rotmat - label_rotmat).abs()

            raw_label_mask_rot_single = raw_label_mask[..., -self.config.action_output_ee_rot_dim:]
            assert self.config.merge_hand
            raw_label_mask_rot_single = raw_label_mask_rot_single.reshape(-1, 2, self.config.action_output_ee_rot_dim // 2)
            raw_label_mask_rot_single = raw_label_mask_rot_single[..., 0]

            if self.config.use_movement_mask:
              ee_rot_loss_single = (
                ee_rot_loss_single * ee_movement_masks.unsqueeze(-1)
              )
            ee_rot_loss_single = ee_rot_loss_single[raw_label_mask_rot_single].mean()
          else:
            raise f"rot representation {self.config.ee_rot_representation} is not supported"

          if self.config.hand_kp_loss_coeff > 0:
            assert self.config.merge_hand
            mano_left.to(hand_label_component.device).to(hand_label_component.dtype)
            mano_right.to(hand_label_component.device).to(hand_label_component.dtype)

            reference_kp_left = mano_forward(
              mano_left,
              hand_label_component[:, 0, :].to(hand_label_component.dtype),
              raw_q_label[:, 0, :].to(hand_label_component.dtype),
              ee_3d_label[:, 0, :].to(hand_label_component.dtype)
            )
            reference_kp_right = mano_forward(
              mano_right,
              hand_label_component[:, 1, :].to(hand_label_component.dtype),
              raw_q_label[:, 1, :].to(hand_label_component.dtype),
              ee_3d_label[:, 1, :].to(hand_label_component.dtype)
            )

            if self.config.ee_rot_representation == "rot6d":
              q_pred_rotvec = batch_matrix2axis(pred_rotmat.reshape(-1, 3, 3))
              q_pred_rotvec = q_pred_rotvec.reshape(-1, 2, 3)
            elif self.config.ee_rot_representation == "quat":
              q_pred_rotvec = raw_q_pred # raw q vec
              q_pred_rotvec = q_pred_rotvec.reshape(-1, 2, 3)

            prediction_kp_left = mano_forward(
              mano_left,
              hand_prediction_component[:, 0, :],
              # raw_q_pred[:, 0, :],
              q_pred_rotvec[:, 0, :],
              ee_3d_prediction[:, 0, :]  
            )
            prediction_kp_right = mano_forward(
              mano_right,
              hand_prediction_component[:, 1, :],
              # raw_q_pred[:, 1, :],
              q_pred_rotvec[:, 1, :],
              ee_3d_prediction[:, 1, :]
            )

            reference_kp = torch.cat([
                reference_kp_left.reshape(-1, 1, 21, 3),
                reference_kp_right.reshape(-1, 1, 21, 3)
            ], dim=1)

            prediction_kp = torch.cat([
                prediction_kp_left.reshape(-1, 1, 21, 3),
                prediction_kp_right.reshape(-1, 1, 21, 3)
            ], dim=1)

            if self.config.loss_use_l1:
              hand_kp_loss_single = torch.abs(
                  reference_kp - \
                  prediction_kp
              )
            else:
              hand_kp_loss_single = (
                  reference_kp - \
                  prediction_kp
              ) ** 2
            hand_kp_loss_single = hand_kp_loss_single.sum(-1).sum(-1)
            hand_kp_loss_single = hand_kp_loss_single[raw_label_mask_3d_single].mean()


          if self.config.action_output_ee_dim is not None:
            ee_l2_loss = ee_l2_loss_single[torch.isfinite(ee_l2_loss_single)].mean()
            hand_l2_loss = hand_l2_loss_single[torch.isfinite(hand_l2_loss_single)]
            hand_l2_loss = hand_l2_loss[..., :self.config.hand_loss_dim].mean()

            if self.config.action_output_ee_2d_dim > 0:
              ee_2d_l2_loss = ee_2d_l2_loss_single[torch.isfinite(ee_2d_l2_loss_single)].mean()
            else:
              ee_2d_l2_loss = torch.Tensor([0]).mean().to(ee_l2_loss.device).to(ee_l2_loss.dtype)

            if self.config.action_output_ee_rot_dim > 0:
              ee_rot_loss = ee_rot_loss_single[torch.isfinite(ee_rot_loss_single)].mean()
            else:
              ee_rot_loss = torch.Tensor([0]).mean().to(ee_l2_loss.device).to(ee_l2_loss.dtype)

            if self.config.hand_kp_loss_coeff > 0:
              hand_kp_loss = hand_kp_loss_single[torch.isfinite(hand_kp_loss_single)].mean()
            else:
              hand_kp_loss = torch.Tensor([0]).mean().to(ee_l2_loss.device).to(ee_l2_loss.dtype)
            l2_loss = self.config.ee_loss_coeff * ee_l2_loss + \
              self.config.hand_loss_coeff * hand_l2_loss + \
              self.config.ee_2d_loss_coeff * ee_2d_l2_loss + \
              self.config.ee_rot_loss_coeff * ee_rot_loss + \
              self.config.hand_kp_loss_coeff * hand_kp_loss
          else:
            l2_loss = torch.stack(l2_loss_list)
            l2_loss = l2_loss[torch.isfinite(l2_loss)].mean()
            ee_l2_loss = None
            hand_l2_loss = None
            ee_2d_l2_loss = None
            ee_rot_loss = None
          KL_loss = None
          next_token_prediction_loss = outputs.loss
          loss = l2_loss + self.config.next_token_loss_coeff * next_token_prediction_loss
          # if not inference:
          if "KLD" in action_output:
            # KL_loss = torch.stack(KLD_loss_list)
            KL_loss = KLD[torch.isfinite(KLD)].mean()
            KL_loss = self.config.kl_weight * KL_loss
            loss = loss + KL_loss

          outputs.loss = loss
          outputs = HOILMOutputWithPast(
            loss = outputs.loss,
            recon_loss = l2_loss,
            ee_l2_loss = ee_l2_loss,
            ee_2d_l2_loss = ee_2d_l2_loss,
            ee_rot_loss = ee_rot_loss,
            hand_l2_loss = hand_l2_loss,
            hand_kp_loss = hand_kp_loss,
            kl_loss=KL_loss,
            logits=outputs.logits,
            prediction=outputs.prediction,
            raw_action_labels=raw_action_labels,
            raw_action_masks=raw_action_masks,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
          )

        if dpo_forward:
            return outputs.logits, labels

        return outputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
