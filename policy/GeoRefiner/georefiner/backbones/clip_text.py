"""Hugging Face CLIP token-level text backbone."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from georefiner.backbones.base import TextBackboneBase, TextBackboneOutput


class CLIPTextBackbone(TextBackboneBase):
    """Encode raw instructions as CLIP token-level hidden states."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str = "main",
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.frozen = frozen
        hidden_size = getattr(model.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("CLIP text config must define a positive hidden_size.")
        self.output_dim = hidden_size
        self.max_length = int(getattr(model.config, "max_position_embeddings", 77))
        self._set_frozen(frozen)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_id: str | None = None,
        revision: str = "main",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        frozen: bool = True,
    ) -> "CLIPTextBackbone":
        """Load real CLIP text weights and tokenizer without mock fallback."""

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_fast=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained CLIP tokenizer "
                f"{model_name_or_path!r} (revision={revision!r}, "
                f"local_files_only={local_files_only}). No mock fallback was used."
            ) from exc
        try:
            from transformers import CLIPTextModel

            model = CLIPTextModel.from_pretrained(
                model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            model_path = Path(model_name_or_path)
            pytorch_bin = model_path / "pytorch_model.bin"
            if not model_path.is_dir() or not pytorch_bin.exists():
                raise RuntimeError(
                    "Failed to load pretrained CLIP text model "
                    f"{model_name_or_path!r} (revision={revision!r}, "
                    f"local_files_only={local_files_only}). No mock fallback was used."
                ) from exc
            try:
                model = cls._load_text_from_local_clip_bin(model_path, pytorch_bin)
            except Exception as local_exc:
                raise RuntimeError(
                    "Failed to load real CLIP text weights from local pytorch_model.bin "
                    f"at {str(pytorch_bin)!r}. No mock fallback was used."
                ) from local_exc
        return cls(
            model,
            tokenizer,
            model_id=model_id or model_name_or_path,
            revision=revision,
            frozen=frozen,
        )

    @staticmethod
    def _load_text_from_local_clip_bin(model_path: Path, weights_path: Path) -> nn.Module:
        """Load the text submodule from an official full-CLIP PyTorch checkpoint."""

        from transformers import AutoConfig, CLIPTextConfig, CLIPTextModel

        full_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        text_config_value = getattr(full_config, "text_config", None)
        if text_config_value is None:
            raise ValueError("Local CLIP config does not contain text_config.")
        text_config = CLIPTextConfig.from_dict(text_config_value.to_dict())
        model = CLIPTextModel(text_config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]
        expected = set(model.state_dict())
        text_state = {key: value for key, value in state.items() if key in expected}
        missing = expected - set(text_state)
        if missing:
            raise RuntimeError(
                f"Official CLIP checkpoint is missing {len(missing)} text keys."
            )
        model.load_state_dict(text_state, strict=True)
        return model

    @classmethod
    def from_config(
        cls,
        config_dict: Mapping[str, Any],
        tokenizer_path: str | Path,
        *,
        model_id: str,
        revision: str = "main",
        frozen: bool = True,
    ) -> "CLIPTextBackbone":
        """Construct an empty CLIP text model for unified checkpoint restore."""

        try:
            from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel

            config = CLIPTextConfig.from_dict(dict(config_dict))
            model = CLIPTextModel(config)
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, use_fast=True
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct offline CLIP text model from checkpoint config "
                f"and tokenizer artifact {str(tokenizer_path)!r}."
            ) from exc
        return cls(
            model,
            tokenizer,
            model_id=model_id,
            revision=revision,
            frozen=frozen,
        )

    def forward(
        self,
        instruction: str | Sequence[str] | None = None,
        token_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> TextBackboneOutput:
        """Return hidden states and ``True = padding`` attention mask."""

        if token_ids is None:
            texts = self._normalize_instruction(instruction)
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            token_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            if token_ids.ndim != 2 or token_ids.dtype != torch.long:
                raise ValueError("token_ids must be torch.long with shape [B, L].")
            if token_ids.shape[1] > self.max_length:
                raise ValueError(
                    f"token_ids length must be <= {self.max_length}; "
                    f"got {token_ids.shape[1]}."
                )
            if attention_mask is None:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                attention_mask = (
                    torch.ones_like(token_ids)
                    if pad_id is None
                    else token_ids.ne(int(pad_id)).to(dtype=torch.long)
                )
        if attention_mask is None or attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids shape [B, L].")

        parameter = next(self.model.parameters())
        token_ids = token_ids.to(device=parameter.device)
        attention_mask = attention_mask.to(device=parameter.device, dtype=torch.long)
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            output = self.model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        tokens = output.last_hidden_state
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("CLIP text encoder produced NaN or Inf tokens.")
        return TextBackboneOutput(
            tokens=tokens,
            padding_mask=attention_mask.eq(0),
        )

    def save_tokenizer(self, path: str | Path) -> None:
        """Save the tokenizer required for offline checkpoint restore."""

        self.tokenizer.save_pretrained(str(path))

    def train(self, mode: bool = True) -> "CLIPTextBackbone":
        """Keep a frozen text encoder in eval mode when its parent trains."""

        super().train(False if self.frozen else mode)
        return self

    def _set_frozen(self, frozen: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.model.eval()
            super().train(False)

    @staticmethod
    def _normalize_instruction(
        instruction: str | Sequence[str] | None,
    ) -> list[str]:
        if instruction is None:
            raise ValueError("CLIPTextBackbone requires instruction or token_ids.")
        if isinstance(instruction, str):
            return [instruction]
        texts = list(instruction)
        if not texts or not all(isinstance(text, str) for text in texts):
            raise TypeError("instruction must be a string or non-empty sequence of strings.")
        return texts
