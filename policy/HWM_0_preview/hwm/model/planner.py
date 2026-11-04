"""Greedy subtask planning and learned WORLD/ACTION query conditioning."""

import logging
import math
import re
from copy import deepcopy

import torch
from PIL import Image
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def planner_message(instruction, memory, subtask, current, history, cfg):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Planner requires a nonempty instruction")
    prompt = cfg.conditioning["planner_prompt"].format(
        instruction=instruction.strip(), memory=memory, subtask=subtask
    )
    content = [{"type": "text", "text": prompt}]
    if history:
        content.append(
            {
                "type": "text",
                "text": cfg.conditioning["history_prompt"].format(
                    interval_seconds=cfg.history["sample_interval_steps"]
                    / cfg.history["control_clock_hz"]
                ),
            }
        )
        content.extend({"type": "image", "image": Image.fromarray(x)} for x in history)
    content.extend(
        [
            {"type": "text", "text": cfg.conditioning["current_prompt"]},
            {"type": "image", "image": Image.fromarray(current)},
        ]
    )
    return {"role": "user", "content": content}


def parse_update(body: str) -> tuple[str, str]:
    match = re.search(
        r"(?:^|\n)\s*Memory Add:\s*(.*?)\s*\n\s*Current Subtask:\s*(.*?)\s*$",
        body.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if match is None or not all(x.strip() for x in match.groups()):
        raise ValueError(f"Malformed planner UPDATE: {body!r}")
    return tuple(x.strip() for x in match.groups())


def select_cache(cache, indices, device):
    selected = deepcopy(cache)
    # reorder_cache handles both attention KV and linear-attention conv/recurrent states.
    selected.reorder_cache(torch.tensor(indices, device=device, dtype=torch.long))
    return selected


class Planner:
    """Grouped GPU inference; episode state remains owned by the adapter."""

    def __init__(self, network, processor, *, vision_encode_batch_size: int = 128):
        if type(vision_encode_batch_size) is not int or vision_encode_batch_size <= 0:
            raise ValueError("vision_encode_batch_size must be positive")
        self.vision_encode_batch_size = vision_encode_batch_size
        self.config = network.config
        self.image_id = self.config.backbones["vlm"]["image_token_id"]
        self.merge_size = self.config.backbones["vlm"]["vision_config"][
            "spatial_merge_size"
        ]
        self.pool_grid = tuple(self.config.history["vlm_pool_grid"])
        self.pool_tokens = math.prod(self.pool_grid)
        self.network = network
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "left"
        tokens = ("<WORLD_PLAN>", "<ACTION_PLAN>", "<KEEP>", "<UPDATE>")
        ids = self.tokenizer.convert_tokens_to_ids(list(tokens))
        if (
            ids != [self.config.conditioning["special_tokens"][x] for x in tokens]
            or len(self.tokenizer)
            != self.config.backbones["vlm"]["text_config"]["vocab_size"]
        ):
            raise ValueError(f"Incompatible planner tokenizer: token IDs={ids}")
        self.world_id, self.action_id, self.keep_id, self.update_id = ids
        self.eos_ids = set(self.config.conditioning["eos_token_ids"])
        self.finished_id = self.config.conditioning["eos_token_ids"][-1]

    def _tokens(self, message, response=None):
        conversation = [message]
        if response is not None:
            conversation.append(
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": [{"type": "text", "text": response}],
                }
            )
        return self.processor.apply_chat_template(
            [conversation],
            tokenize=True,
            processor_kwargs={"padding": True},
            add_generation_prompt=response is None,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

    def _compact(self, inputs, history_count):
        """Remove excess history placeholders; keep real grids for vision separately."""
        grids = inputs["image_grid_thw"].clone()
        ids = inputs["input_ids"][0].tolist()
        lengths = (grids.prod(-1) // self.merge_size**2).tolist()
        if len(lengths) != history_count + 1:
            raise ValueError(
                "Expected history images followed by one current composite"
            )
        kept, cursor, image_index = [], 0, 0
        while cursor < len(ids):
            if ids[cursor] != self.image_id:
                kept.append(cursor)
                cursor += 1
                continue
            if image_index >= len(lengths):
                raise ValueError("Unexpected image placeholder in planner text")
            length = lengths[image_index]
            if image_index < history_count and self.pool_tokens > length:
                raise ValueError(
                    "History pooling grid exceeds available image placeholders"
                )
            if ids[cursor : cursor + length] != [self.image_id] * length:
                raise ValueError("Image placeholders do not match the encoding grid")
            kept.extend(
                range(
                    cursor,
                    cursor
                    + (self.pool_tokens if image_index < history_count else length),
                )
            )
            cursor += length
            image_index += 1
        if image_index != len(lengths):
            raise ValueError("Missing image placeholders")
        result = {
            key: inputs[key][:, kept]
            for key in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        grids[:history_count] = grids.new_tensor(
            [1, *(x * self.merge_size for x in self.pool_grid)]
        )
        result["image_grid_thw"] = grids
        return result

    @torch.no_grad()
    def __call__(self, instruction, memory, subtask, cache_valid, current, history):
        return self.batch(
            [
                {
                    "instruction": instruction,
                    "memory": memory,
                    "subtask": subtask,
                    "cache_valid": cache_valid,
                    "current": current,
                    "history": history,
                }
            ]
        )[0]

    def _images(self, inputs, history_count):
        return self._images_many([inputs], [history_count])[0]

    def _prefill(self, logical, features, *, use_cache=True):
        output, position = self._prefill_group(
            [logical], [features], use_cache=use_cache
        )
        return output, int(position[0])

    def _continue(self, embeddings, cache, position):
        positions = torch.tensor([position], device=embeddings.device)
        return self._continue_group(embeddings, cache, positions)

    def _generate(self, cache, position, prefix, budget, *, stop_at_newline=False):
        positions = torch.tensor([position], device=self.network.world_queries.device)
        return self._generate_group(
            cache, positions, [prefix], budget, stop_at_newline
        )[0]

    @torch.no_grad()
    def _images_many(self, inputs, history_counts):
        device = self.network.world_queries.device
        grids = torch.cat([x["image_grid_thw"] for x in inputs])
        pixels = torch.cat([x["pixel_values"] for x in inputs])
        lengths = grids.prod(-1).tolist()
        features, cursor = [], 0
        for start in range(0, len(grids), self.vision_encode_batch_size):
            end = cursor + sum(lengths[start : start + self.vision_encode_batch_size])
            output = self.network.vlm.model.get_image_features(
                pixels[cursor:end].to(device=device, dtype=torch.bfloat16),
                grids[start : start + self.vision_encode_batch_size].to(device),
                return_dict=True,
            ).pooler_output
            features.extend(output)
            cursor = end
        groups = {}
        first = 0
        for count in history_counts:
            for i in range(first, first + count):
                _, h, w = grids[i].tolist()
                groups.setdefault(
                    (h // self.merge_size, w // self.merge_size), []
                ).append(i)
            first += count + 1
        for (h, w), indices in groups.items():
            values = torch.stack([features[i].reshape(h, w, -1) for i in indices])
            values = F.adaptive_avg_pool2d(
                values.permute(0, 3, 1, 2).contiguous(), self.pool_grid
            )
            for i, value in zip(
                indices,
                values.permute(0, 2, 3, 1).reshape(len(indices), self.pool_tokens, -1),
                strict=True,
            ):
                features[i] = value
        result, cursor = [], 0
        for count in history_counts:
            result.append(torch.cat(features[cursor : cursor + count + 1]))
            cursor += count + 1
        return result

    def _prefill_group(self, logicals, features, *, use_cache=True, with_queries=False):
        device = self.network.world_queries.device
        logical = {
            key: torch.cat([row[key] for row in logicals]).to(device)
            for key in logicals[0]
        }
        positions, delta = self.network.vlm.model.get_rope_index(**logical)
        embeds = self.network.vlm.get_input_embeddings()(logical["input_ids"])
        image_mask = logical["input_ids"].eq(self.image_id)
        visual = torch.cat(features)
        if int(image_mask.sum()) != len(visual):
            raise ValueError("Batch image placeholders do not match visual features")
        embeds[image_mask] = visual
        if with_queries:
            queries = torch.cat(
                (self.network.world_queries, self.network.action_queries), dim=1
            )
            embeds[:, -queries.shape[1] :] = queries.to(embeds)
        output = self.network.vlm.model.language_model(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=logical["attention_mask"],
            use_cache=use_cache,
            return_dict=True,
        )
        return output, logical["input_ids"].shape[1] + delta.flatten()

    def _continue_group(self, embeds, cache, position):
        batch, length = embeds.shape[:2]
        positions = (
            position[:, None] + torch.arange(length, device=embeds.device)[None]
        )[None].expand(3, -1, -1)
        return self.network.vlm.model.language_model(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=torch.ones(
                batch,
                cache.get_seq_length() + length,
                device=embeds.device,
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    def _generate_group(self, cache, position, prefixes, budget, stop_at_newline=False):
        device = self.network.world_queries.device
        ids = torch.tensor(prefixes, device=device)
        output = self._continue_group(
            self.network.vlm.get_input_embeddings()(ids), cache, position
        )
        position = position + ids.shape[1]
        tokens = [[] for _ in prefixes]
        results = [None for _ in prefixes]
        for step in range(budget):
            selected = (
                self.network.vlm.lm_head(output.last_hidden_state[:, -1])
                .float()
                .argmax(-1)
                .tolist()
            )
            for i, token in enumerate(selected):
                if results[i] is not None:
                    continue
                tokens[i].append(token)
                text = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
                if token in self.eos_ids:
                    results[i] = (text.strip(), "eos")
                elif stop_at_newline and "\n" in text:
                    results[i] = (text, "newline")
            if all(value is not None for value in results):
                break
            if step + 1 < budget:
                selected = [
                    token if result is None else self.finished_id
                    for token, result in zip(selected, results, strict=True)
                ]
                ids = torch.tensor(selected, device=device)[:, None]
                output = self._continue_group(
                    self.network.vlm.get_input_embeddings()(ids),
                    output.past_key_values,
                    position,
                )
                position = position + 1
        return [
            (
                value
                if value is not None
                else (
                    self.tokenizer.decode(tokens[i], skip_special_tokens=True).strip(),
                    "length",
                )
            )
            for i, value in enumerate(results)
        ]

    @torch.no_grad()
    def batch(self, requests):
        if not requests:
            return []
        messages = [
            planner_message(
                x["instruction"],
                x["memory"],
                x["subtask"],
                x["current"],
                x["history"],
                self.config,
            )
            for x in requests
        ]
        inputs = [self._tokens(x) for x in messages]
        counts = [len(x["history"]) for x in requests]
        logical = [self._compact(x, n) for x, n in zip(inputs, counts, strict=True)]
        features = self._images_many(inputs, counts)
        groups = {}
        for i, x in enumerate(logical):
            groups.setdefault(x["input_ids"].shape[1], []).append(i)
        results = [None] * len(requests)
        device = self.network.world_queries.device
        for indices in groups.values():
            output, position = self._prefill_group(
                [logical[i] for i in indices], [features[i] for i in indices]
            )
            cache = output.past_key_values
            decisions = (
                self.network.vlm.lm_head(output.last_hidden_state[:, -1])[
                    :, [self.keep_id, self.update_id]
                ]
                .float()
                .argmax(-1)
                .tolist()
            )
            update = [
                not requests[i]["cache_valid"] or decision == 1
                for i, decision in zip(indices, decisions, strict=True)
            ]
            bodies = [""] * len(indices)
            deltas = ["None."] * len(indices)
            subtasks = [requests[i]["subtask"] for i in indices]
            selected = [j for j, value in enumerate(update) if value]
            retry = []
            if selected:
                generated = self._generate_group(
                    select_cache(cache, selected, device),
                    position[selected],
                    [[self.update_id] for _ in selected],
                    self.config.conditioning["max_update_tokens"],
                )
                for j, (body, reason) in zip(selected, generated, strict=True):
                    bodies[j] = body
                    try:
                        deltas[j], subtasks[j] = parse_update(body)
                        if reason == "length":
                            raise ValueError("Planner UPDATE reached the token limit")
                    except ValueError as error:
                        logger.warning(
                            "Planner retry: %s; original body=%r", error, body
                        )
                        retry.append(j)
            # Preserve the original two bounded constrained retries; batch compatible prefixes.
            prefixes = {j: "Memory Add: " for j in retry}
            fields = {j: [] for j in retry}
            for budget in self.config.conditioning["retry_token_budgets"]:
                retry_groups = {}
                for j in retry:
                    tokens = [
                        self.update_id,
                        *self.tokenizer.encode(prefixes[j], add_special_tokens=False),
                    ]
                    retry_groups.setdefault(len(tokens), []).append((j, tokens))
                for members in retry_groups.values():
                    js = [x[0] for x in members]
                    fresh, p = self._prefill_group(
                        [logical[indices[j]] for j in js],
                        [features[indices[j]] for j in js],
                    )
                    generated = self._generate_group(
                        fresh.past_key_values, p, [x[1] for x in members], budget, True
                    )
                    for j, (text, reason) in zip(js, generated, strict=True):
                        value = text.split("\n", 1)[0].strip()
                        if not value or reason == "length":
                            raise ValueError(
                                f"Planner UPDATE retry failed: original={bodies[j]!r}, field={text!r}, finish={reason}"
                            )
                        fields[j].append(value)
                        prefixes[j] = f"Memory Add: {value}\nCurrent Subtask: "
            for j in retry:
                bodies[j] = (
                    f"Memory Add: {fields[j][0]}\nCurrent Subtask: {fields[j][1]}"
                )
                deltas[j], subtasks[j] = parse_update(bodies[j])
            suffix_groups = {}
            for j, i in enumerate(indices):
                response = "<UPDATE>\n" + bodies[j].strip() if update[j] else "<KEEP>"
                full = self._compact(self._tokens(messages[i], response), counts[i])[
                    "input_ids"
                ][0]
                prefix = logical[i]["input_ids"][0]
                if not torch.equal(full[: len(prefix)], prefix):
                    raise ValueError("Assistant response changed prompt prefix")
                suffix = full[len(prefix) :].tolist()
                if update[j]:
                    stop = next(
                        (
                            k + 1
                            for k, token in enumerate(suffix)
                            if token in self.eos_ids
                        ),
                        None,
                    )
                    if stop is None:
                        raise ValueError("Assistant template did not terminate UPDATE")
                    suffix = suffix[:stop]
                else:
                    suffix = suffix[:1]
                    if suffix != [self.keep_id]:
                        raise ValueError("KEEP must be exactly one decision token")
                suffix_groups.setdefault(len(suffix), []).append((j, suffix))
            # Recompute prompt + response + learned queries as one sequence.
            # Qwen3.5's BF16 chunked recurrent attention is not bitwise invariant
            # to prefill/continuation boundaries. Keep generation caching above,
            # but match the full-prefix conditioning path used by the model.
            del output, cache
            for members in suffix_groups.values():
                js = [x[0] for x in members]
                full_inputs = []
                for j, suffix in members:
                    prefix = logical[indices[j]]
                    extra = prefix["input_ids"].new_tensor(
                        [
                            suffix
                            + [self.world_id] * self.config.m["world_queries"]
                            + [self.action_id] * self.config.m["action_horizon"]
                        ]
                    )
                    full_inputs.append(
                        {
                            **prefix,
                            "input_ids": torch.cat((prefix["input_ids"], extra), 1),
                            "attention_mask": torch.cat(
                                (prefix["attention_mask"], torch.ones_like(extra)), 1
                            ),
                            "mm_token_type_ids": torch.cat(
                                (prefix["mm_token_type_ids"], torch.zeros_like(extra)),
                                1,
                            ),
                        }
                    )
                query_output, _ = self._prefill_group(
                    full_inputs,
                    [features[indices[j]] for j in js],
                    use_cache=False,
                    with_queries=True,
                )
                plans = query_output.last_hidden_state
                for row, j in enumerate(js):
                    results[indices[j]] = (
                        plans[
                            row : row + 1,
                            -(
                                self.config.m["world_queries"]
                                + self.config.m["action_horizon"]
                            ) : -self.config.m["action_horizon"],
                        ].clone(),
                        plans[
                            row : row + 1, -self.config.m["action_horizon"] :
                        ].clone(),
                        "UPDATE" if update[j] else "KEEP",
                        deltas[j],
                        subtasks[j],
                    )
        return results
