# Copyright (C) 2026 Xiaomi Corporation.
# Modified by the Rex-M1-preview authors (2026): history-frame placeholder shortening, LLM-side
# position ids and image_pool_mask for pooled history frames.
import itertools
import os
from typing import Any, Dict, List

import torch
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor

SCORE_ID = 151669
STATE_ID = 151670
IM_START_ID = 151644
IMAGE_ID = 151655

# Tokens each pooled history frame occupies on the LLM side. Must equal HISTORY_TOKENS_PER_IMAGE in
# mibot/models/VLM/qwen3vl.py, which does the pooling; a mismatch raises in get_placeholder_mask.
HISTORY_TOKENS_PER_IMAGE = 16


class CustomCollate:
    def __init__(self) -> None:
        self.max_length = int(os.environ.get("MAX_LENGTH", 4096))
        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(60)})
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", use_fast=True, extra_special_tokens=special_tokens
        )
        token_ids = self.processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        if token_ids != [SCORE_ID, STATE_ID, STATE_ID + 1, STATE_ID + 60]:
            raise ValueError(f"unexpected action token ids: {token_ids}")

    def _llm_grids(self, inputs: Dict[str, torch.Tensor], n_hist: int):
        """LLM-side grid (t, h, w) of each image, h/w already divided by merge_size.

        The first n_hist images are history frames pooled to 4x4 tokens, so their LLM-side grid is
        (t, 4, 4) rather than the pixel grid in image_grid_thw. The vision tower uses the pixel grid;
        placeholder expansion and position ids use the grids returned here.
        """
        grids = inputs.get("image_grid_thw")
        if grids is None:
            return []
        merge = self.processor.image_processor.merge_size
        side = int(HISTORY_TOKENS_PER_IMAGE**0.5)
        out = []
        for index, grid in enumerate(grids):
            time, height, width = (int(value) for value in grid)
            if index < n_hist:
                out.append((time, side, side))
            else:
                out.append((time, height // merge, width // merge))
        return out

    def _shrink_history_pads(self, inputs: Dict[str, torch.Tensor], llm_grids, n_hist: int) -> None:
        """Shorten the <|image_pad|> run of each of the first n_hist images to
        HISTORY_TOKENS_PER_IMAGE, in place. The processor expands placeholders from the pixel grid;
        the vision tower still receives the full pixel_values and image_grid_thw.
        """
        if n_hist <= 0 or inputs.get("image_grid_thw") is None:
            return
        merge = self.processor.image_processor.merge_size
        tokens = inputs["input_ids"][0].tolist()
        kept, start = [], 0
        for index, grid in enumerate(inputs["image_grid_thw"]):
            time, height, width = (int(value) for value in grid)
            pads = time * (height // merge) * (width // merge)
            image_start = tokens.index(IMAGE_ID, start)
            kept.extend(tokens[start:image_start])
            wanted = llm_grids[index][0] * llm_grids[index][1] * llm_grids[index][2]
            kept.extend([IMAGE_ID] * wanted)
            start = image_start + pads
        kept.extend(tokens[start:])
        inputs["input_ids"] = torch.tensor(kept, dtype=inputs["input_ids"].dtype).view(1, -1)
        if inputs.get("attention_mask") is not None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    def _position_ids(self, inputs: Dict[str, torch.Tensor], llm_grids) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        if not llm_grids:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)

        tokens = input_ids[0].tolist()
        positions, start = [], 0
        for time, height, width in llm_grids:
            image_start = tokens.index(IMAGE_ID, start)
            text_length = image_start - start
            base = positions[-1].max() + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            temporal = torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            positions.append(torch.stack([temporal, rows, columns]) + text_length + base)
            start = image_start + time * height * width

        if start < len(tokens):
            base = positions[-1].max() + 1
            positions.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + base)
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected = []
        total_length = 0
        for item in batch:
            inputs = self.processor.apply_chat_template(
                item["messages"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                do_resize=False,
            )
            pool = int(item.get("history_pool", 0) or 0)
            if pool and "n_hist" not in item:
                raise ValueError("history_pool is set but the sample has no n_hist: dataset and collator disagree")
            n_hist = int(item.get("n_hist", 0) or 0) if pool else 0
            if pool and pool != HISTORY_TOKENS_PER_IMAGE:
                raise ValueError(
                    f"history_pool={pool} does not match HISTORY_TOKENS_PER_IMAGE="
                    f"{HISTORY_TOKENS_PER_IMAGE}"
                )
            llm_grids = self._llm_grids(inputs, n_hist)
            self._shrink_history_pads(inputs, llm_grids, n_hist)

            length = inputs["input_ids"].shape[1]
            if total_length + length > self.max_length:
                continue

            state = (inputs["input_ids"][0] == STATE_ID).nonzero(as_tuple=False)
            im_starts = (inputs["input_ids"][0] == IM_START_ID).nonzero(as_tuple=False).flatten()
            im_starts = im_starts[im_starts < state[0, 0]] if state.numel() else im_starts[:0]
            if im_starts.numel() == 0:
                raise ValueError("cannot locate the action-conditioning turn")
            inputs["position_ids"] = self._position_ids(inputs, llm_grids)
            if pool:
                inputs["image_pool_mask"] = torch.tensor(
                    [index < n_hist for index in range(len(llm_grids))], dtype=torch.bool
                )
                detach = int(item.get("history_detach", 0) or 0)
            payload = {
                key: value
                for key, value in item.items()
                if key not in ("messages", "n_hist", "history_pool", "history_detach")
            }
            selected.append((inputs, payload, int(im_starts[-1])))
            total_length += length

        if not selected:
            raise ValueError(f"no training sample fits within MAX_LENGTH={self.max_length}")

        lengths = [inputs["input_ids"].shape[1] for inputs, _, _ in selected]
        offsets = [0] + list(itertools.accumulate(lengths))
        result = {
            "input_ids": torch.cat([inputs["input_ids"] for inputs, _, _ in selected], dim=1),
            "position_ids": torch.cat([inputs["position_ids"] for inputs, _, _ in selected], dim=2),
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "max_length_q": max(lengths),
            "max_length_k": max(lengths),
            "action_segments": torch.tensor(list(zip(offsets[:-1], offsets[1:])), dtype=torch.long),
            "action_vlm_condition_segments": torch.tensor(
                [[offsets[index], offsets[index] + end] for index, (_, _, end) in enumerate(selected)],
                dtype=torch.long,
            ),
        }
        if selected and selected[0][0].get("image_pool_mask") is not None:
            result["image_pool_detach"] = bool(detach)
        for key in ("pixel_values", "image_grid_thw", "image_pool_mask"):
            values = [inputs[key] for inputs, _, _ in selected if inputs.get(key) is not None]
            if values:
                if key == "image_pool_mask" and len(values) != len(selected):
                    raise ValueError(
                        "image_pool_mask is present on only some samples; history_pool must be "
                        "uniform within a batch"
                    )
                result[key] = torch.cat(values)

        payloads = [payload for _, payload, _ in selected]
        targets = [payload.pop("vlm_action_target") for payload in payloads]
        masks = [payload.pop("vlm_action_mask") for payload in payloads]
        result.update(default_collate(payloads))
        result["vlm_action_target"] = torch.cat(targets)
        result["vlm_action_mask"] = torch.cat(masks)
        return result
