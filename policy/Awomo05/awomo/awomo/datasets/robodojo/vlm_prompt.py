from __future__ import annotations

import torch

QUESTION = "What should the robot do now? Answer with one short imperative sentence."
TEXT_LAYOUT = "prefix_v2"
_MAX_LONGEST_EDGE = 16777216

def history_offsets_frames(num_slots: int, period_frames: float) -> list[int]:
    return [int(round(k * period_frames)) for k in range(int(num_slots), 0, -1)]


def build_messages(instruction, image, history=(), period_s: float = 1.0):

    history = list(history)
    content = []
    if history:
        content.append({"type": "text",
                        "text": f"Past {len(history)} frames, {period_s:g} s apart, oldest first:"})
        content.extend({"type": "image", "image": h} for h in history)
        content.append({"type": "text", "text": "Current frame:"})
    content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": f"Instruction: {instruction}\n{QUESTION}"})
    return [{"role": "user", "content": content}]


class VLMPromptCodec:

    def __init__(self, processor, image_size=(448, 448), history_image_size=(448, 448),
                 history_pool=(4, 4), history_period_s: float = 1.0, max_history: int = 20):
        self.proc = processor
        self.tok = processor.tokenizer
        self.image_size = tuple(int(v) for v in image_size)
        self.history_image_size = tuple(int(v) for v in history_image_size)
        self.history_pool = (int(history_pool[0]), int(history_pool[1]))
        self.history_period_s = float(history_period_s)
        self.max_history = int(max_history)
        ip = processor.image_processor
        self.patch_size = int(getattr(ip, "patch_size", 16))
        self.merge_size = int(getattr(ip, "merge_size", 2))
        factor = self.patch_size * self.merge_size
        ph, pw = self.history_pool
        self.history_llm_size = (pw * factor, ph * factor)
        self.history_llm_grid = torch.tensor([[1, ph * self.merge_size, pw * self.merge_size]],
                                             dtype=torch.long)

        self.history_images_kwargs = {"size": {"shortest_edge": self.history_llm_size[0] * self.history_llm_size[1],
                                               "longest_edge": _MAX_LONGEST_EDGE}}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.eos_id = self.tok.eos_token_id
        self.image_token_id = (getattr(processor, "image_token_id", None)
                               or self.tok.convert_tokens_to_ids("<|image_pad|>"))
        from PIL import Image

        self.tokens_per_image = self._count_tokens(ip(images=[Image.new("RGB", self.image_size)],
                                                      return_tensors="pt")["image_grid_thw"])
        self.tokens_per_history_frame = self._count_tokens(self.history_llm_grid)
        vit_grid = ip(images=[Image.new("RGB", self.history_image_size)], return_tensors="pt",
                      **self.history_images_kwargs)["image_grid_thw"]
        self.history_vit_grid = vit_grid[0].tolist()
        self.vit_tokens_per_history_frame = self._count_tokens(vit_grid)
        if self.vit_tokens_per_history_frame < self.tokens_per_history_frame:
            raise ValueError(f"history_image_size={self.history_image_size} yields {self.vit_tokens_per_history_frame} "
                             f"ViT tokens, fewer than the {self.tokens_per_history_frame} pooled slots")
        self._cache = {}

    def _count_tokens(self, grid) -> int:
        return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum()) // (self.merge_size ** 2)

    def prompt(self, instruction: str, n_hist: int = 0):
        n_hist = int(n_hist)
        if not 0 <= n_hist <= self.max_history:
            raise ValueError(f"n_hist={n_hist} outside [0, {self.max_history}]")
        key = (instruction, n_hist)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        from PIL import Image

        cur = Image.new("RGB", self.image_size)
        hist = [Image.new("RGB", self.history_llm_size) for _ in range(n_hist)]
        text = self.proc.apply_chat_template(
            build_messages(instruction, cur, hist, self.history_period_s),
            tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[text], images=hist + [cur], return_tensors="pt", padding=False,
                        images_kwargs=self.history_images_kwargs)
        ids = enc["input_ids"][0]
        n_pad = int((ids == self.image_token_id).sum())
        want = n_hist * self.tokens_per_history_frame + self.tokens_per_image
        if n_pad != want:
            raise RuntimeError(
                f"prompt expanded to {n_pad} image tokens, expected {want} "
                f"({n_hist}x{self.tokens_per_history_frame} + {self.tokens_per_image}); "
                "the image processor is not resizing the way this codec assumes.")
        span = self._find_span(ids.tolist(), instruction)
        if span is None:
            raise RuntimeError(f"instruction tokens not found inside prompt: {instruction!r}")
        cached = (ids, torch.arange(span[0], span[1], dtype=torch.long))
        self._cache[key] = cached
        return cached

    def _find_span(self, ids, instruction):

        s = instruction.strip()
        for variant in (" " + s, s, " " + s.rstrip("."), s.rstrip(".")):
            sub = self.tok(variant, add_special_tokens=False)["input_ids"]
            for drop_last in (0, 1):
                cand = sub[:len(sub) - drop_last]
                n = len(cand)
                if n < 3:
                    continue
                for i in range(len(ids) - n + 1):
                    if ids[i:i + n] == cand:
                        return i, i + n
        for variant in (" " + s, s):
            cand = self.tok(variant, add_special_tokens=False)["input_ids"]
            if not 0 < len(cand) < 3:
                continue
            for i in range(len(ids) - len(cand) + 1):
                if ids[i:i + len(cand)] == cand:
                    return i, i + len(cand)
        return None

    def subtask_ids(self, subtask):
        s = (subtask or "").strip()
        if not s:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(self.tok(s, add_special_tokens=False)["input_ids"] + [self.eos_id],
                            dtype=torch.long)

    def image_inputs(self, pil_image, history=()):
        ip = self.proc.image_processor
        history = list(history)
        pvs, grids, grids_llm = [], [], []
        if history:
            for im in history:
                if im.size != self.history_image_size:
                    raise ValueError(f"history frame is {im.size}, codec expects {self.history_image_size}")
            out = ip(images=history, return_tensors="pt", **self.history_images_kwargs)
            if out["image_grid_thw"].tolist() != [self.history_vit_grid] * len(history):
                raise RuntimeError(f"history ViT grid drifted: {out['image_grid_thw'].tolist()[:2]} vs {self.history_vit_grid}")
            pvs.append(out["pixel_values"])
            grids.append(out["image_grid_thw"])
            grids_llm.append(self.history_llm_grid.repeat(len(history), 1))
        if pil_image.size != self.image_size:
            raise ValueError(f"current frame is {pil_image.size}, codec expects {self.image_size}")
        out = ip(images=[pil_image], return_tensors="pt")
        pvs.append(out["pixel_values"])
        grids.append(out["image_grid_thw"])
        grids_llm.append(out["image_grid_thw"])
        is_hist = torch.tensor([True] * len(history) + [False], dtype=torch.bool)
        return torch.cat(pvs, dim=0), torch.cat(grids, dim=0), torch.cat(grids_llm, dim=0), is_hist

    def encode_sample(self, instruction, pil_image, history=()):
        history = list(history)
        p_ids, instr_pos = self.prompt(instruction, len(history))
        ids = p_ids
        pixel_values, grid, grid_llm, is_hist = self.image_inputs(pil_image, history)
        n_llm_tok = self._count_tokens(grid_llm)
        n_pad = int((p_ids == self.image_token_id).sum())
        if n_llm_tok != n_pad:
            raise RuntimeError(
                f"language-tower image tokens {n_llm_tok} != prompt placeholders {n_pad} (K={len(history)})"
            )
        return {
            "input_ids": ids,
            "prompt_len": int(p_ids.numel()),
            "instr_pos": instr_pos,
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
            "image_grid_thw_llm": grid_llm,
            "image_is_hist": is_hist,
            "n_images": len(history) + 1,
            "mm_token_type_ids": (ids == self.image_token_id).long(),
            "pad_id": self.pad_id,
        }


def collate_vlm(items):
    pad_id = items[0]["pad_id"]
    L = max(x["input_ids"].numel() for x in items)
    Li = max(x["instr_pos"].numel() for x in items)

    def pad1(t, n, v):
        return torch.cat([t, torch.full((n - t.numel(),), v, dtype=t.dtype)])

    out = {
        "input_ids": torch.stack([pad1(x["input_ids"], L, pad_id) for x in items]),
        "attention_mask": torch.stack([pad1(torch.ones_like(x["input_ids"]), L, 0) for x in items]),
        "mm_token_type_ids": torch.stack([pad1(x["mm_token_type_ids"], L, 0) for x in items]),
        "prompt_len": torch.tensor([x["prompt_len"] for x in items], dtype=torch.long),
        "instr_pos": torch.stack([pad1(x["instr_pos"], Li, -1) for x in items]),
        "n_images": torch.tensor([int(x["n_images"]) for x in items], dtype=torch.long),
        "pixel_values": torch.cat([x["pixel_values"] for x in items], dim=0),
        "image_grid_thw": torch.cat([x["image_grid_thw"] for x in items], dim=0),
        "image_grid_thw_llm": torch.cat([x["image_grid_thw_llm"] for x in items], dim=0),
        "image_is_hist": torch.cat([x["image_is_hist"] for x in items], dim=0),
    }
    return out
