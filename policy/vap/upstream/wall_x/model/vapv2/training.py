"""Training facade for VAPv2; data, losses and sampling stay VAP-compatible."""

from wall_x.model.vap.training import VAPTrainingModel
import torch

from .model import VAPv2TokenModel
from .text import JointTextProjector


class VAPv2TrainingModel(VAPTrainingModel):
    @classmethod
    def from_wan_pipeline(cls, pipeline, config, *, action_dim, state_dim):
        model = VAPTrainingModel.from_wan_pipeline(
            pipeline, config, action_dim=action_dim, state_dim=state_dim
        )
        model.__class__ = cls
        model.core = VAPv2TokenModel.from_existing(model.core)
        # Replace the legacy one-layer Qwen bridge with VAPv2's explicit
        # trainable Dq -> Wan-Dit projector. Qwen remains a frozen prompter;
        # per-layer text attention is trainable inside the policy blocks.
        model.core.text_input_projector = JointTextProjector(
            int(config.text_feature_hidden_dim),
            int(config.text_projector_output_dim),
        ).to(
            device=next(model.core.parameters()).device,
            dtype=next(model.core.parameters()).dtype,
        )
        return model

    def _encode_text_context(self, batch, *, batch_size, device, dtype):
        context = super()._encode_text_context(
            batch, batch_size=batch_size, device=device, dtype=dtype
        )
        provider = self.prompter
        if hasattr(provider, "lengths"):
            prompts = []
            from wall_x.model.vap.training import build_prompt_from_data
            prompt = build_prompt_from_data(
                batch,
                use_qwen25=False,
                prompt_template=getattr(self.config, "prompt_template", None),
                qwen25_prompt_template=None,
                prompt_priority_order=getattr(self.config, "prompt_priority_order", None),
            )
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            lengths = [provider.lengths[p] for p in prompts]
        elif getattr(provider, "tokenizer", None) is not None:
            from wall_x.model.vap.training import build_prompt_from_data
            prompt = build_prompt_from_data(
                batch,
                use_qwen25=False,
                prompt_template=getattr(self.config, "prompt_template", None),
                qwen25_prompt_template=None,
                prompt_priority_order=getattr(self.config, "prompt_priority_order", None),
            )
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            lengths = [len(provider.tokenizer.encode(p, add_special_tokens=False)) + 1 for p in prompts]
        else:
            raise ValueError("VAPv2 text provider must expose tokenizer or cache lengths")
        valid = torch.arange(context.shape[1], device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        self.core._v2_text_valid_mask = valid
        return context
