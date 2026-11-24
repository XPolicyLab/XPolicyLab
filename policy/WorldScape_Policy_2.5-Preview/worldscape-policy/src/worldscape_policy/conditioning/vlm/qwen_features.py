"""Online Qwen3-VL-4B features for joint T5/VLM conditioning."""
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from worldscape_policy.checkpoint.processors import processor_directory
from .protocol import AutoPlanningFeatures
from .qwen_prompt import QwenObservationProcessor


def prepare_images(images, input_range, color_order, size=448):
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError('Expected images with shape [B,T,3,H,W]')
    pixels = images.detach().cpu()
    if input_range == 'uint8':
        if pixels.dtype != torch.uint8:
            raise TypeError('uint8 image inputs require uint8 tensors')
    else:
        minimum = {'zero_one': 0, 'minus_one_one': -1}.get(input_range)
        if (minimum is None or not pixels.is_floating_point() or not torch.isfinite(pixels).all()
                or (pixels < minimum).any() or (pixels > 1).any()):
            raise ValueError('Invalid normalized image input')
        pixels = pixels.float()
        if minimum == -1:
            pixels = (pixels + 1) / 2
        pixels = (pixels * 255).round().to(torch.uint8)
    if color_order == 'bgr':
        # Explicit checkpoint opt-in; framework observations and VAE stay RGB.
        pixels = pixels[:, :, [2, 1, 0]]
    elif color_order != 'rgb':
        raise ValueError('input_color_order must be rgb or bgr')
    pixels = pixels.permute(0, 1, 3, 4, 2).numpy()
    return [[Image.fromarray(np.ascontiguousarray(frame)).resize((size, size), Image.Resampling.BILINEAR)
             for frame in row] for row in pixels]


class QwenBackbone(nn.Module):
    """Standard Qwen parameters restored from the unified policy shards."""
    def __init__(self, config_dir):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        from transformers.modeling_utils import no_init_weights
        config = Qwen3VLConfig.from_pretrained(config_dir, local_files_only=True)
        if (config.text_config.hidden_size, config.text_config.num_hidden_layers) != (2560, 36):
            raise ValueError('This policy requires the Qwen3-VL-4B architecture')
        with no_init_weights():
            self.vl_model = Qwen3VLForConditionalGeneration._from_config(
                config, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        self.vl_model.config.use_cache = False


class QwenFeatureEncoder(nn.Module):
    output_dim = 7680

    def __init__(self, *, checkpoint_dir, input_color_order='rgb',
                 instruction_layers=(9, 18, 27), summary_layers=(18, 27, 36)):
        super().__init__()
        from transformers import AutoProcessor
        directory = processor_directory(Path(checkpoint_dir), 'vlm')
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        self.inputs = QwenObservationProcessor(processor)
        self.encoder = QwenBackbone(directory)
        self.input_color_order = input_color_order
        self.instruction_layers = tuple(instruction_layers)
        self.summary_layers = tuple(summary_layers)
        if any(len(group) != 3 or any(type(i) is not int or not 0 <= i <= 36 for i in group)
               for group in (self.instruction_layers, self.summary_layers)):
            raise ValueError('Qwen feature layers must each select three hidden states in [0,36]')
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def _compress_history(self, tokens, batch):
        merge = self.inputs.merge
        sizes = (batch['vision_grid'].prod(-1) // merge**2).tolist()
        images = list(tokens.split(sizes))
        indices = batch['history_mask'].nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return tokens
        grids = batch['vision_grid'][indices]
        if not torch.equal(grids, grids[:1].expand_as(grids)):
            raise ValueError('History image grids must have equal resolution')
        time, height, width = grids[0].tolist()
        _, target_h, target_w = batch['language_grid'][indices[0]].tolist()
        history = torch.stack([images[i] for i in indices])
        channels = history.shape[-1]
        history = history.reshape(-1, height // merge, width // merge, channels).permute(0, 3, 1, 2)
        history = F.adaptive_avg_pool2d(history.float(), (target_h // merge, target_w // merge)).to(tokens.dtype)
        history = history.permute(0, 2, 3, 1).reshape(len(indices), time * target_h * target_w // merge**2, channels)
        for row, index in enumerate(indices):
            images[index] = history[row]
        return torch.cat(images)

    def _hidden_states(self, batch):
        qwen = self.encoder.vl_model.model
        ids = batch['input_ids']
        embeddings = qwen.get_input_embeddings()(ids)
        vision = qwen.get_image_features(batch['pixel_values'].to(qwen.dtype), batch['vision_grid'])
        if hasattr(vision, 'pooler_output'):
            image_tokens, deepstack = vision.pooler_output, vision.deepstack_features
        else:
            image_tokens, deepstack = vision
        def pooled(value):
            return self._compress_history(torch.cat(list(value)) if isinstance(value, (list, tuple)) else value, batch)
        image_tokens = pooled(image_tokens).to(embeddings)
        if deepstack is not None:
            deepstack = [pooled(value) for value in deepstack]
        mask, _ = qwen.get_placeholder_mask(ids, inputs_embeds=embeddings, image_features=image_tokens)
        embeddings = embeddings.masked_scatter(mask, image_tokens)
        rope = {'image_grid_thw': batch['language_grid'], 'video_grid_thw': None,
                'attention_mask': batch['attention_mask']}
        if 'mm_token_type_ids' in inspect.signature(qwen.get_rope_index).parameters:
            rope['mm_token_type_ids'] = batch['modality_ids']
        positions, _ = qwen.get_rope_index(ids, **rope)
        return qwen.language_model(
            input_ids=None, inputs_embeds=embeddings, attention_mask=batch['attention_mask'],
            position_ids=positions, use_cache=False, past_key_values=None,
            cache_position=torch.arange(ids.shape[1], device=ids.device),
            visual_pos_masks=mask[..., 0], deepstack_visual_embeds=deepstack,
            output_hidden_states=True).hidden_states

    def _features(self, batch):
        hidden = self._hidden_states(batch)
        positions = batch['instruction_positions']
        valid = positions >= 0
        layers = torch.cat([hidden[i] for i in self.instruction_layers], dim=-1)
        instruction = layers.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, self.output_dim))
        rows = torch.arange(len(positions), device=positions.device)
        summary = torch.cat([hidden[i][rows, batch['summary_positions']] for i in self.summary_layers], dim=-1)
        compact = [torch.cat((text[keep], gist[None])) for text, keep, gist in zip(instruction, valid, summary, strict=True)]
        tokens = nn.utils.rnn.pad_sequence(compact, batch_first=True)
        lengths = torch.tensor([len(row) for row in compact], device=tokens.device)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
        return AutoPlanningFeatures(perception_features=tokens, direct_condition_mask=mask, semantic_supervision=False)

    @torch.no_grad()
    def encode_planning(self, *, images, planning_text, negative_text=None, training=False,
                        visual_input_range='zero_one', history_images=None, history_mask=None,
                        planning_supervision=False, planning_labels_text=None):
        if training or planning_supervision:
            raise ValueError('The Preview Qwen feature encoder supports inference only')
        if images.ndim != 5 or images.shape[1] != 1 or len(planning_text) != images.shape[0]:
            raise ValueError('Expected one current image and instruction per environment')
        if not all(isinstance(text, str) and text.strip() for text in planning_text):
            raise ValueError('Instructions must be nonempty strings')
        current = prepare_images(images, visual_input_range, self.input_color_order)
        if history_images is None:
            if history_mask is not None:
                raise ValueError('History mask requires history images')
            histories = [[] for _ in current]
        else:
            if (history_images.ndim != 5 or len(history_images) != len(current) or history_mask is None
                    or history_mask.dtype != torch.bool or history_mask.shape != history_images.shape[:2]
                    or (history_mask.sum(-1) > 20).any()):
                raise ValueError('Expected a boolean history mask and at most 20 past images')
            histories = [prepare_images(row[keep.to(row.device)][None], visual_input_range, self.input_color_order)[0]
                         for row, keep in zip(history_images, history_mask, strict=True)]
        batch = self.inputs.batch(planning_text, [row[0] for row in current], histories)
        device = next(self.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        return self._features(batch)
