from typing import Any, Dict, List, Tuple, Union

import torch
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from transformers import AutoTokenizer, UMT5EncoderModel

from giga_train import Trainer


class WanTrainer(Trainer):
    """Trainer for WAN text-to-video fine-tuning.

    Sets up the pretrained tokenizer, text encoder, and VAE (frozen), and returns the trainable transformer model. Also implements helper utilities
    used by the training loop.
    """

    def get_models(self, model_config: Any) -> WanTransformer3DModel:
        """Instantiate components and return the trainable transformer.

        Args:
            model_config: Config object with at least `pretrained` and
                `flow_shift` fields.

        Returns:
            WanTransformer3DModel: The transformer in train mode.
        """
        pretrained = model_config.pretrained
        self.flow_shift = model_config.flow_shift
        # text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder='tokenizer')
        self.text_encoder = UMT5EncoderModel.from_pretrained(pretrained, subfolder='text_encoder', torch_dtype=self.dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device)
        # vae
        self.vae = AutoencoderKLWan.from_pretrained(pretrained, subfolder='vae')
        self.vae.requires_grad_(False)
        self.vae.to(self.device)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device)
        # transformer
        transformer = WanTransformer3DModel.from_pretrained(pretrained, subfolder='transformer', torch_dtype=self.dtype)
        transformer.train()
        return transformer

    def forward_step(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        """Compute the denoising MSE loss for one training step.

        Args:
            batch_dict (dict): Contains `images` (Tensor[B, T, C, H, W]) and
                `prompt` (List[str] or str).

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        with torch.no_grad():
            images = batch_dict['images'].to(self.vae.dtype)
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.latents_mean) * self.latents_std
            prompt_embeds = self.get_t5_prompt_embeds(batch_dict['prompt'])
        timestep, sigma = self.get_timestep_and_sigma(latents.shape[0], latents.ndim)
        noise = torch.randn_like(latents)
        target = noise - latents
        noisy_latents = noise * sigma + latents * (1 - sigma)
        model_pred = self.model(
            hidden_states=noisy_latents.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(self.dtype),
            return_dict=False,
        )[0]
        loss = (model_pred.float() - target.float()) ** 2
        loss = loss.mean()
        return loss

    def get_t5_prompt_embeds(self, prompt: Union[str, List[str]], max_sequence_length: int = 512) -> torch.Tensor:
        """Encode prompts with UMT5 and pad/truncate to a fixed length.

        Args:
            prompt (str | List[str]): The input text prompt(s).
            max_sequence_length (int): Maximum token length to keep.

        Returns:
            torch.Tensor: Tensor[B, max_sequence_length, hidden_size] of
            embeddings on CPU (move to device when used).
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), mask.to(self.device)).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)
        return prompt_embeds

    def get_timestep_and_sigma(self, batch_size: int, ndim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noise level and build broadcastable sigma tensor.

        The flow shift remaps a uniform sample in [0, 1] to a biased range
        suitable for target resolution training.

        Args:
            batch_size (int): Number of samples in the batch.
            ndim (int): Target number of dimensions for sigma.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (timestep int tensor, sigma float
            tensor with shape broadcastable to latents).
        """
        sigma = torch.rand(batch_size).to(self.device)
        # flow_shift: 5.0 for 720P, 3.0 for 480P
        sigma = self.flow_shift * sigma / (1 + (self.flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma
