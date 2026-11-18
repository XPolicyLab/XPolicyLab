"""Frozen Qwen feature encoder with the original numerical operations.

Original implementation credit: Westlake DI.
Checkpoint terms are retained in LICENSE-model.md.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt import TEXT_LAYOUT, history_offsets_frames
import logging

logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGET_MODULES = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class VLPromptEncoder(nn.Module):
    def __init__(self, vl_path, qwen_layers=(9, 18, 27), summary_layers=(18, 27, 36),
                 max_prompt_len=640, device="cuda", torch_dtype=torch.bfloat16, lora=None,
                 attn_implementation="sdpa",
                 image_size=(448, 448), history_image_size=(448, 448),
                 history_pool=(4, 4), history_num_slots=20, history_period_s=1.0, history_fps=25.0,
                 max_new_tokens=24, decode_every=4, load_pretrained=True):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(vl_path)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        self.merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        if load_pretrained:
            full = Qwen3VLForConditionalGeneration.from_pretrained(
                vl_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
            )
        else:
            # Native checkpoints restore every parameter with strict loading.
            # Keep real rotary buffers and original BF16/FP32 module dtypes.
            from transformers import Qwen3VLConfig
            from transformers.modeling_utils import no_init_weights
            config = Qwen3VLConfig.from_pretrained(vl_path, local_files_only=True)
            with no_init_weights():
                full = Qwen3VLForConditionalGeneration._from_config(
                    config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        full.requires_grad_(False)                   # freeze BEFORE injecting LoRA

        lora = dict(lora or {})
        self.lora_enabled = bool(lora.pop("enabled", False))
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(
                r=int(lora.pop("r", 64)),
                lora_alpha=int(lora.pop("alpha", 128)),
                lora_dropout=float(lora.pop("dropout", 0.05)),
                bias="none",
                target_modules=lora.pop("target_modules", DEFAULT_LORA_TARGET_MODULES),
            )
            if lora:
                raise ValueError(f"Unknown lora options: {sorted(lora)}")
            full = get_peft_model(full, cfg)
        self.vl_model = full.to(device)              # peft-wrapped when LoRA is on
        self.image_token_id = int(self.base_model.config.image_token_id)
        self.qwen_layers = tuple(int(l) for l in qwen_layers)
        self.summary_layers = tuple(int(l) for l in summary_layers)
        if len(self.summary_layers) != len(self.qwen_layers):
            raise ValueError(
                f"summary_layers={self.summary_layers} must have the same count as "
                f"qwen_layers={self.qwen_layers}: both must fill the DiT's txt_in width.")
        n_layers = int(self.vl.config.text_config.num_hidden_layers)
        for l in self.qwen_layers + self.summary_layers:
            if not 0 <= l <= n_layers:
                raise ValueError(f"layer index {l} outside [0, {n_layers}] (hidden_states has {n_layers + 1} entries)")
        self.max_prompt_len = int(max_prompt_len)
        hidden = int(self.vl.config.text_config.hidden_size)   # 2560
        self.dim = hidden * len(self.qwen_layers)              # 7680
        self.image_size = tuple(int(v) for v in image_size)          # = data.train.vlm.image_size
        self.history_image_size = tuple(int(v) for v in history_image_size)   # what the ViT sees
        self.history_pool = (int(history_pool[0]), int(history_pool[1]))     # tokens grid after pooling
        self.history_num_slots = int(history_num_slots)
        self.history_period_s = float(history_period_s)
        self.history_fps = float(history_fps)
        self.max_new_tokens = int(max_new_tokens)
        self.decode_every = int(decode_every)
        self.text_layout = TEXT_LAYOUT
        self.loaded_text_layout = None               # set from the checkpoint payload
        self._codec = None
        self.device = device
        self.torch_dtype = torch_dtype
        self.to(device)
        if not self.lora_enabled:
            self.eval()
        logger.info(
            "VLPromptEncoder(Qwen3-VL, layers=%s, summary_layers=%s, dim=%d, lora=%s, "
            "text_layout=%s, image=%s, history=%dx%s pooled %dx%d @%.2fs, "
            "decode_every=%d)",
            self.qwen_layers, self.summary_layers, self.dim,
            "on" if self.lora_enabled else "off (frozen feature extractor)",
            self.text_layout, self.image_size,
            self.history_num_slots, self.history_image_size, self.history_pool[0], self.history_pool[1],
            self.history_period_s, self.decode_every,
        )

    # ------------------------------------------------------------------ views
    @property
    def base_model(self):
        """Qwen3VLForConditionalGeneration, unwrapped from peft if needed."""
        m = self.vl_model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    @property
    def vl(self):
        return self.base_model.model



    def _inner(self):
        base = self.base_model
        return base.model, base.lm_head

    @property
    def codec(self):
        """Tokenizer-side codec built from this encoder's processor."""
        if self._codec is None:
            from .prompt import VLMPromptCodec

            self._codec = VLMPromptCodec(
                self.processor, self.image_size, history_image_size=self.history_image_size,
                history_pool=self.history_pool, history_period_s=self.history_period_s,
                max_history=self.history_num_slots)
        return self._codec

    def history_offsets(self) -> list[int]:
        return history_offsets_frames(self.history_num_slots, self.history_period_s * self.history_fps)

    def build_infer_batch(self, instructions, images, histories=None):
        from .prompt import collate_vlm

        if histories is None:
            histories = [[] for _ in instructions]
        if not (len(instructions) == len(images) == len(histories)):
            raise ValueError(f"#instructions={len(instructions)} #images={len(images)} "
                             f"#histories={len(histories)} must agree")
        items = [self.codec.encode_sample(str(i).strip(), "", img, list(h))
                 for i, img, h in zip(instructions, images, histories)]
        vb = collate_vlm(items, has_subtask_gt=[False] * len(items))
        dev = next(self.vl.parameters()).device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in vb.items()}

    # ------------------------------------------------------------- the gather
    def _gather_text(self, hs, vb):
        pos = vb["instr_pos"]                                            # [B,Li], -1 = pad
        valid = pos >= 0
        feats = torch.cat([hs[k] for k in self.qwen_layers], dim=-1)    # [B,L,7680]
        text = torch.gather(feats, 1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        text = text * valid.unsqueeze(-1).to(text.dtype)
        batch = int(pos.shape[0])
        last = (vb["prompt_len"] - 1).clamp(min=0).to(pos.device)       # [B]
        rows = torch.arange(batch, device=pos.device)
        summary = torch.cat([hs[k][rows, last] for k in self.summary_layers], dim=-1)  # [B,7680]
        text = torch.cat([text, summary.unsqueeze(1).to(text.dtype)], dim=1)
        valid = torch.cat([valid, torch.ones(batch, 1, dtype=torch.bool, device=pos.device)], dim=1)
        return text, valid

    # ------------------------------------------------------------- checkpoint

    def load_lora_state_dict(self, state_dict):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(self.vl_model, state_dict)

    # ------------------------------------------------- history pooling + forward
    def _image_token_counts(self, grid):
        return ((grid[:, 0] * grid[:, 1] * grid[:, 2]) // (self.merge_size ** 2)).tolist()

    def _pool_history(self, feats, vb):
        m = self.merge_size
        chunks = list(feats.split(self._image_token_counts(vb["image_grid_thw"]), dim=0))
        is_hist = vb["image_is_hist"].tolist()
        hist_idx = [i for i, h in enumerate(is_hist) if h]
        if not hist_idx:
            return feats
        g_act = vb["image_grid_thw"].tolist()
        g_llm = vb["image_grid_thw_llm"].tolist()
        ref_act, ref_llm = g_act[hist_idx[0]], g_llm[hist_idx[0]]
        for i in hist_idx:
            if g_act[i] != ref_act or g_llm[i] != ref_llm:
                raise ValueError(f"history frames must share one grid: {g_act[i]}/{g_llm[i]} vs {ref_act}/{ref_llm}")
        t, H, W = ref_act[0], ref_act[1] // m, ref_act[2] // m
        th, tw = ref_llm[1] // m, ref_llm[2] // m
        x = torch.stack([chunks[i] for i in hist_idx], dim=0)             # [n, t*H*W, D]
        n, _, D = x.shape
        x = x.reshape(n * t, H, W, D).permute(0, 3, 1, 2)                 # [n*t, D, H, W]
        x = F.adaptive_avg_pool2d(x.float(), (th, tw)).to(feats.dtype)    # [n*t, D, th, tw]
        x = x.permute(0, 2, 3, 1).reshape(n, t * th * tw, D)
        for j, i in enumerate(hist_idx):
            chunks[i] = x[j]
        return torch.cat(chunks, dim=0)

    def _forward(self, vb, use_cache=False, output_hidden_states=True, past_key_values=None):
        vl = self.vl
        input_ids, attention_mask = vb["input_ids"], vb["attention_mask"]
        inputs_embeds = vl.get_input_embeddings()(input_ids)
        if "pooled_vision" in vb:
            # Offline episode-local reuse: visual embeddings are independent of
            # language/history position. The caller supplies exact pooled rows.
            image_embeds, deepstack = vb["pooled_vision"]
        else:
            with torch.no_grad():
                vision = vl.get_image_features(vb["pixel_values"].to(vl.dtype), vb["image_grid_thw"])
                if hasattr(vision, "pooler_output"):
                    image_embeds, deepstack = vision.pooler_output, vision.deepstack_features
                else:
                    image_embeds, deepstack = vision[0], vision[1]
                if isinstance(image_embeds, (list, tuple)):
                    image_embeds = torch.cat(list(image_embeds), dim=0)
                image_embeds = self._pool_history(image_embeds, vb)
                if deepstack is not None:
                    deepstack = [self._pool_history(torch.cat(list(d), 0) if isinstance(d, (list, tuple)) else d, vb)
                                 for d in deepstack]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vl.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        # Transformers 4.57 infers image spans from special tokens; newer
        # versions additionally accept explicit modality IDs. Keep the same
        # input IDs and pooled image grids in either API (images only here).
        import inspect
        rope_kwargs = dict(image_grid_thw=vb["image_grid_thw_llm"],
                           video_grid_thw=None, attention_mask=attention_mask)
        if "mm_token_type_ids" in inspect.signature(vl.get_rope_index).parameters:
            rope_kwargs["mm_token_type_ids"] = vb["mm_token_type_ids"]
        position_ids, rope_deltas = vl.get_rope_index(input_ids, **rope_kwargs)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        out = vl.language_model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack, output_hidden_states=output_hidden_states)
        return out, position_ids, rope_deltas

    def _run(self, vb):
        return self._forward(vb)[0].hidden_states


    @torch.no_grad()
    def infer_vlm(self, vb):
        hs = self._run(vb)
        text, valid = self._gather_text(hs, vb)
        return text.to(self.torch_dtype), valid

    @staticmethod
    def _row_image_slice(vb, i: int):
        grid = vb["image_grid_thw"]
        n_img = vb["n_images"].tolist()
        first = int(sum(n_img[:i]))
        last = first + int(n_img[i])
        per_image = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
        lo = int(sum(per_image[:first]))
        hi = lo + int(sum(per_image[first:last]))
        return lo, hi, first, last

    @torch.no_grad()
    def generate_subtask(self, vb, max_new_tokens=None, return_details=False):
        from transformers import DynamicCache

        _, lm_head = self._inner()
        vl = self.vl
        eos = int(self.tok.eos_token_id)
        max_new = int(max_new_tokens or self.max_new_tokens)
        outs = []
        for i in range(int(vb["input_ids"].shape[0])):
            n = int(vb["prompt_len"][i])
            lo, hi, first, last = self._row_image_slice(vb, i)
            row = {"input_ids": vb["input_ids"][i:i + 1, :n],
                   "attention_mask": vb["attention_mask"][i:i + 1, :n],
                   "mm_token_type_ids": vb["mm_token_type_ids"][i:i + 1, :n],
                   "pixel_values": vb["pixel_values"][lo:hi],
                   "image_grid_thw": vb["image_grid_thw"][first:last],
                   "image_grid_thw_llm": vb["image_grid_thw_llm"][first:last],
                   "image_is_hist": vb["image_is_hist"][first:last]}
            out, pos, _ = self._forward(row, use_cache=True, output_hidden_states=False,
                                        past_key_values=DynamicCache())
            cache = out.past_key_values
            hidden = out.last_hidden_state[:, -1]
            pos = pos[:, :, -1:]                                    # [3,1,1] last prompt position
            tokens = []
            for step in range(max_new):
                tok = int(lm_head(hidden).argmax(-1))
                if tok == eos:
                    break
                tokens.append(tok)
                pos = pos + 1
                emb = vl.get_input_embeddings()(torch.tensor([[tok]], device=hidden.device))
                out = vl.language_model(
                    input_ids=None, inputs_embeds=emb, position_ids=pos, past_key_values=cache,
                    use_cache=True, cache_position=torch.tensor([n + step], device=hidden.device))
                cache = out.past_key_values
                hidden = out.last_hidden_state[:, -1]
            text = self.tok.decode(tokens, skip_special_tokens=True).strip()
            outs.append({"subtask": text, "token_ids": tokens,
                         "prompt_token_ids": row["input_ids"][0].tolist(),
                         "prompt_tokens": n,
                         "history_count": int(row["image_is_hist"].sum())}
                        if return_details else text)
        return outs
