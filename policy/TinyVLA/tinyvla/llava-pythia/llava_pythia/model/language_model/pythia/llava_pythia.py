import os
from typing import List, Optional, Tuple, Union
from policy_heads.models import build_ACT_head
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    GPTNeoXModel, GPTNeoXPreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from ...llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.utils import logging
from .configuration_llava_pythia import LlavaPythiaConfig

logger = logging.get_logger(__name__)

class LLavaPythiaModel(LlavaMetaModel, GPTNeoXModel):
    config_class = LlavaPythiaConfig

    def __init__(self, config):
        super(LLavaPythiaModel, self).__init__(config)


class LlavaPythiaForCausalLM(GPTNeoXPreTrainedModel, LlavaMetaForCausalLM):
    """
    LlavaPythiaForCausalLM is a class that extends GPTNeoXPreTrainedModel and LlavaMetaForCausalLM.
    It is designed to handle causal language modeling tasks with additional capabilities for processing
    multimodal inputs, such as images, and generating actions based on different head types.
    """
    config_class = LlavaPythiaConfig

    # _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super(GPTNeoXPreTrainedModel, self).__init__(config)
        self.gpt_neox = LLavaPythiaModel(config)

        self.head_type = config.action_head_type
        self.visual_concat = config.concat
        self.action_dim = config.action_dim
        if config.action_head_type == 'act':
            self.embed_out = build_ACT_head(config.act['act'])
            middle_dim = int(max(config.hidden_size, config.act['act']['hidden_dim']) / 2)
            self.proj_to_action = nn.Sequential(
                nn.Linear(config.hidden_size, middle_dim),
                nn.LayerNorm(middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, config.act['act']['hidden_dim']),
                nn.LayerNorm(config.act['act']['hidden_dim']),
            )

        elif config.action_head_type == 'droid_diffusion':
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            from policy_heads.models import ConditionalUnet1D
            self.proj_to_action = nn.Identity()
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )
            self.embed_out = ConditionalUnet1D(
                input_dim=config.action_dim,
                global_cond_dim=config.hidden_size,
                state_dim=config.state_dim
            )
            self.num_queries = config.chunk_size
            self.noise_samples = 1
            self.num_inference_timesteps = 10

        self.post_init()

    def get_channel_proj(self, x):
        return self.channel_proj(x)

    def encode_images(self, images, proj=True):
        image_features = self.get_model().get_vision_tower()(images)
        if proj:  # 默认true，则会执行
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_mm_projector(self, image_features):
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_image_fusion_embedding(self, visual_concat=None, images=None, images_r=None, images_top=None, states=None):
        """
        Fuses image features based on the specified visual concatenation method.

        Args:
            visual_concat: Method for concatenating visual features.
            images: Main images to be encoded.
            images_r: Right-side images for concatenation.
            images_top: Top-side images for concatenation.
            states: Additional state information.

        Returns:
            Fused image features.
        """
        if "channel_cat" not in visual_concat:
            image_features = self.encode_images(images)
        if images_top is not None:
            image_features_top = self.encode_images(images_top)
        if images_r is not None:

            if visual_concat == 'token_cat':
                image_features_r = self.encode_images(images_r)
                # Concatenate right-side features
                image_features = torch.cat([image_features, image_features_r], dim=1)
                if images_top is not None:
                    # Concatenate top-side features
                    image_features = torch.cat([image_features, image_features_top], dim=1)
            else:
                raise ValueError(f"Unimplentmented concat style:{visual_concat}")
        # Return final concatenated features
        return image_features

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def get_model(self):
        return self.gpt_neox

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            actions=None,
            states=None,
            images_r=None,
            images_top=None,
            is_pad=None,
            eval=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the LlavaPythiaForCausalLM model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Mask to avoid performing attention on padding token indices.
            past_key_values: Past key values for caching.
            inputs_embeds: Input embeddings.
            labels: Labels for computing the loss.
            use_cache: Whether to use cache.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            images: Input images for multimodal processing.
            return_dict: Whether to return a dictionary or a tuple.
            actions: Target actions for training.
            states: Additional state information.
            images_r: Right-side images for concatenation.
            images_top: Top-side images for concatenation.
            is_pad: Mask indicating padded regions.
            eval: Evaluation mode flag.

        Returns:
            A tuple or CausalLMOutputWithPast containing the model outputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, images_r=images_r, images_top=images_top, visual_concat=self.visual_concat, states=states)

        outputs = self.get_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        if self.head_type == 'fc':
            loss, logits = self.forward_fc_head(labels, actions, hidden_states, states)

        elif self.head_type == 'act':
            if not eval:
                loss = self.forward_act_head(actions, hidden_states, states, is_pad)
            else:
                action = self.forward_act_head(actions, hidden_states, states, is_pad)
                return action
            logits = None
        elif self.head_type == 'droid_diffusion':
            if not eval:
                loss = self.forward_diffusion_head(actions, hidden_states, states, is_pad)
                logits = None
            else:
                action = self.forward_diffusion_head(actions, hidden_states, states, is_pad)
                return action
        # return outputs as a tuple instead of a structured dictionary
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_fc_head(self, labels, actions, hidden_states, states):
        """
        Forward pass for the fully connected head (default setting).
        Args:
            labels (torch.Tensor, optional): Ground truth labels for classification.
            actions (torch.Tensor, optional): Target actions for regression.
            hidden_states (torch.Tensor): The hidden states used as input features.
            states (torch.Tensor): The robot state information.
        
        Returns:
            tuple: (loss, logits)
                - loss (torch.Tensor or None): The computed loss if applicable.
                - logits (torch.Tensor): The predicted output logits.
        """
        logits = self.embed_out(input_feature=hidden_states, state_tensor=states)

        loss = None
        if labels is not None and actions is None: # training time
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if actions is not None: # inference time

            loss = torch.nn.functional.huber_loss(logits, actions)
        return loss, logits

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

    def forward_act_head(self, actions, hidden_states, states, is_pad=None, vq_sample=None):
        """
        Forward pass for the action head.
        This function processes actions using a VAE-based structure.
        Args:
            actions (torch.Tensor, optional): Target actions for training.
            hidden_states (torch.Tensor): The hidden states used as input features.
            states (torch.Tensor): The robot state information.
            is_pad (torch.Tensor, optional): Mask indicating padded regions.
            vq_sample (optional): Additional sampling parameter.
        
        Returns:
            dict or torch.Tensor:
                - If training, returns a dictionary containing `l1` loss, `kl` divergence, and total loss.
                - If inference, returns predicted actions (`a_hat`).
        """
        env_state = None

        hidden_states = self.proj_to_action(hidden_states)
        if actions is not None:  # training time
            actions = actions[:, :self.embed_out.num_queries]
            is_pad = is_pad[:, :self.embed_out.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.embed_out(qpos=states, hidden_states=hidden_states, env_state=env_state, actions=actions, is_pad=is_pad,
                                                                          vq_sample=vq_sample)

            total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)

            all_l1 = torch.nn.functional.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.config.act['act']['kl_weight']
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _, _ = self.embed_out(qpos=states, hidden_states=hidden_states, env_state=env_state, vq_sample=vq_sample)  # no action, sample from prior
            return a_hat


    def forward_diffusion_head(self, actions, hidden_states, states, is_pad):
        """
        Forward pass for the diffusion head.
        This function applies a diffusion-based process to predict actions.
        Args:
            actions (torch.Tensor, optional): Target actions for training.
            hidden_states (torch.Tensor): Hidden states used as input features.
            states (torch.Tensor): The robot state information.
            is_pad (torch.Tensor): Mask indicating padded regions.
        
        Returns:
            dict or torch.Tensor:
                - If training, returns a dictionary containing the MSE loss.
                - If inference, returns the predicted actions.
        """
        if actions is not None:  # training time
            B = actions.size(0)
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            num_noise_samples = self.noise_samples
            # sample noise to add to actions
            noise = torch.randn([num_noise_samples] + list(actions.shape), device=actions.device,
                                dtype=actions.dtype)  # num_noise, B, Ta, D
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=actions.device
            ).long()

            timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]

            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            assert hidden_states.ndim == 3

            hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            is_pad = is_pad.repeat(num_noise_samples, 1)
            states = states.repeat(num_noise_samples,  1)

            noise_pred = self.embed_out(noisy_actions, timesteps, global_cond=hidden_states, states=states)
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
            return {'loss': loss}
        else:  # inference time
            B = 1
            Tp = self.num_queries
            action_dim = self.action_dim

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, Tp, action_dim)).cuda()

            naction = noisy_action.to(dtype=hidden_states.dtype)
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=states)

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction


AutoConfig.register("llava_pythia", LlavaPythiaConfig)
AutoModelForCausalLM.register(LlavaPythiaConfig, LlavaPythiaForCausalLM)
