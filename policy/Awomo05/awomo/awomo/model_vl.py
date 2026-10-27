from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from awomo.awomo.model import ImageWAMActionPatch
from awomo.awomo.datasets.robodojo.vlm_prompt import TEXT_LAYOUT, history_offsets_frames
from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_LORA_TARGET_MODULES = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class VLPromptEncoder(nn.Module):
    def __init__(self, vl_path, qwen_layers=(9, 18, 27), summary_layers=(18, 27, 36),
                 max_prompt_len=640, device="cuda", torch_dtype=torch.bfloat16, lora=None,
                 attn_implementation="sdpa",
                 image_size=(448, 448), history_image_size=(448, 448),
                 history_pool=(4, 4), history_num_slots=20, history_period_s=1.0, history_fps=25.0,
                 max_new_tokens=24, decode_every=4, gradient_checkpointing=False):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(vl_path)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        self.merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        full = Qwen3VLForConditionalGeneration.from_pretrained(
            vl_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
        )
        full.requires_grad_(False)

        lora = dict(lora or {})
        self.lora_enabled = bool(lora.pop("enabled", False))
        self.lora_lr = float(lora.pop("lr", 2e-5))
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model

            if gradient_checkpointing:
                full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                full.enable_input_require_grads()
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
        self.vl_model = full.to(device)
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
        hidden = int(self.vl.config.text_config.hidden_size)
        self.dim = hidden * len(self.qwen_layers)
        self.image_size = tuple(int(v) for v in image_size)
        self.history_image_size = tuple(int(v) for v in history_image_size)
        self.history_pool = (int(history_pool[0]), int(history_pool[1]))
        self.history_num_slots = int(history_num_slots)
        self.history_period_s = float(history_period_s)
        self.history_fps = float(history_fps)
        self.max_new_tokens = int(max_new_tokens)
        self.decode_every = int(decode_every)
        self.text_layout = TEXT_LAYOUT
        self.loaded_text_layout = None
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

    @property
    def base_model(self):
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
        if self._codec is None:
            from awomo.awomo.datasets.robodojo.vlm_prompt import VLMPromptCodec

            self._codec = VLMPromptCodec(
                self.processor, self.image_size, history_image_size=self.history_image_size,
                history_pool=self.history_pool, history_period_s=self.history_period_s,
                max_history=self.history_num_slots)
        return self._codec

    def history_offsets(self) -> list[int]:
        return history_offsets_frames(self.history_num_slots, self.history_period_s * self.history_fps)

    def build_infer_batch(self, instructions, images, histories=None):
        from awomo.awomo.datasets.robodojo.vlm_prompt import collate_vlm

        if histories is None:
            histories = [[] for _ in instructions]
        if not (len(instructions) == len(images) == len(histories)):
            raise ValueError(f"#instructions={len(instructions)} #images={len(images)} "
                             f"#histories={len(histories)} must agree")
        items = [self.codec.encode_sample(str(i).strip(), img, list(h))
                 for i, img, h in zip(instructions, images, histories)]
        vb = collate_vlm(items)
        dev = next(self.vl.parameters()).device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in vb.items()}

    def _gather_text(self, hs, vb):
        pos = vb["instr_pos"]
        valid = pos >= 0
        feats = torch.cat([hs[k] for k in self.qwen_layers], dim=-1)
        text = torch.gather(feats, 1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        text = text * valid.unsqueeze(-1).to(text.dtype)
        batch = int(pos.shape[0])
        last = (vb["prompt_len"] - 1).clamp(min=0).to(pos.device)
        rows = torch.arange(batch, device=pos.device)
        summary = torch.cat([hs[k][rows, last] for k in self.summary_layers], dim=-1)
        text = torch.cat([text, summary.unsqueeze(1).to(text.dtype)], dim=1)
        valid = torch.cat([valid, torch.ones(batch, 1, dtype=torch.bool, device=pos.device)], dim=1)
        return text, valid


    def load_lora_state_dict(self, state_dict):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(self.vl_model, state_dict)


    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]


    def lora_state_dict(self):
        from peft import get_peft_model_state_dict

        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.vl_model).items()}


    def forward_vlm(self, vb):
        hs = self._run(vb)
        text, valid = self._gather_text(hs, vb)
        return text.to(self.torch_dtype), valid

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
        x = torch.stack([chunks[i] for i in hist_idx], dim=0)
        n, _, D = x.shape
        x = x.reshape(n * t, H, W, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x.float(), (th, tw)).to(feats.dtype)
        x = x.permute(0, 2, 3, 1).reshape(n, t * th * tw, D)
        for j, i in enumerate(hist_idx):
            chunks[i] = x[j]
        return torch.cat(chunks, dim=0)

    def _forward(self, vb, use_cache=False, output_hidden_states=True, past_key_values=None):
        vl = self.vl
        input_ids, attention_mask = vb["input_ids"], vb["attention_mask"]
        inputs_embeds = vl.get_input_embeddings()(input_ids)
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
        position_ids, rope_deltas = vl.get_rope_index(
            input_ids, mm_token_type_ids=vb["mm_token_type_ids"], image_grid_thw=vb["image_grid_thw_llm"],
            video_grid_thw=None, attention_mask=attention_mask)
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
    def generate_subtask(self, vb, max_new_tokens=None):
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
            pos = pos[:, :, -1:]
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
            outs.append(self.tok.decode(tokens, skip_special_tokens=True).strip())
        return outs


class ImageWAMActionPatchVL(ImageWAMActionPatch):
    _infer_count = 0
    _last_subtasks: Optional[list] = None
    _last_n_hist: Optional[list] = None

    def attach_vl_encoder(self, enc: VLPromptEncoder) -> None:
        self.vl_encoder = enc

    def load_extra_checkpoint_payload(self, payload) -> None:
        enc = getattr(self, "vl_encoder", None)
        state = payload.get("vl_lora") if isinstance(payload, dict) else None
        if state is None:
            if enc is not None and enc.lora_enabled:
                logger.warning("Checkpoint has no `vl_lora`; the adapter starts from scratch.")
            return
        if enc is None or not enc.lora_enabled:
            logger.warning("Checkpoint carries `vl_lora` but this model has no LoRA-enabled VL encoder.")
            return
        enc.load_lora_state_dict(state)
        layout = payload.get("vl_text_layout")
        enc.loaded_text_layout = layout
        if layout != enc.text_layout:
            logger.warning(
                "layout mistach")


    def infer_action_flux2(self, prompt, input_image, *args, vl_images=None, vl_history=None, **kwargs):
        prev = (getattr(self, "_infer_input_image", None), getattr(self, "_infer_vl_pil", None),
                getattr(self, "_infer_vl_hist", None))
        self._infer_input_image = input_image
        self._infer_vl_pil = vl_images
        self._infer_vl_hist = vl_history
        try:
            return super().infer_action_flux2(prompt, input_image, *args, **kwargs)
        finally:
            self._infer_input_image, self._infer_vl_pil, self._infer_vl_hist = prev

    def reset_ar_state(self) -> None:
        self._infer_count = 0
        self._last_subtasks = None
        self._last_n_hist = None

    def _vl_infer_images(self, enc, batch_size: int):
        from PIL import Image

        stashed = getattr(self, "_infer_vl_pil", None)
        if stashed is not None:
            images = list(stashed)
            if len(images) != batch_size:
                raise ValueError(f"vl_images has {len(images)} frames but {batch_size} prompts were given.")
        else:
            img = getattr(self, "_infer_input_image", None)
            if img is None:
                raise ValueError("VL eval needs the observation frame stashed as _infer_input_image.")
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim == 5:
                img = img[:, 0]
            images = self._frame_to_pil(img)
        size = (int(enc.image_size[0]), int(enc.image_size[1]))
        return [im if im.size == size else im.resize(size, Image.BILINEAR) for im in images]

    def _vl_infer_history(self, enc, batch_size: int):
        from PIL import Image

        hist = getattr(self, "_infer_vl_hist", None)
        if hist is None:
            if not getattr(self, "_warned_no_vl_hist", False):
                logger.warning("infer_action_flux2 called without vl_history")
                self._warned_no_vl_hist = True
            return [[] for _ in range(batch_size)]
        hist = [list(h) for h in hist]
        if len(hist) != batch_size:
            raise ValueError(f"vl_history has {len(hist)} rows but {batch_size} prompts were given.")
        size = (int(enc.history_image_size[0]), int(enc.history_image_size[1]))
        out = []
        for row in hist:
            if len(row) > enc.history_num_slots:
                raise ValueError(f"vl_history row has {len(row)} frames > history_num_slots={enc.history_num_slots}")
            out.append([im if im.size == size else im.resize(size, Image.BILINEAR) for im in row])
        return out

    def _prepare_flux2_infer_text(self, prompt, context, context_mask):
        if context is not None or context_mask is not None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        enc = getattr(self, "vl_encoder", None)
        if enc is None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        images = self._vl_infer_images(enc, len(prompts))
        histories = self._vl_infer_history(enc, len(prompts))

        vb = enc.build_infer_batch(prompts, images, histories)
        text, mask = enc.infer_vlm(vb)
        if enc.decode_every > 0 and self._infer_count % enc.decode_every == 0:
            self._last_subtasks = enc.generate_subtask(vb)
        self._infer_count += 1
        self._last_n_hist = [len(h) for h in histories]
        return (text.to(device=self.device, dtype=self.torch_dtype),
                mask.to(device=self.device, dtype=torch.bool))
