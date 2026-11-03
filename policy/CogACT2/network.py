"""Inference-only Qwen3.5 feature encoder and two-stream flow transformer."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from .attention import AttentionBackend
from .memory import INITIAL_TYPES, HISTORY_TYPES


class FloatLayerNorm(nn.LayerNorm):
    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1_g = nn.Linear(dim, hidden)
        self.fc1_x = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1_g(x)) * self.fc1_x(x))


class LoRALinear(nn.Module):
    """Evaluation of the checkpoint's unmerged PEFT default adapter."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base.in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, base.out_features, bias=False)})
        self.scale = alpha / rank

    def forward(self, x):
        result = self.base_layer(x)
        update = self.lora_B["default"](self.lora_A["default"](x.to(self.lora_A["default"].weight.dtype)))
        return (result + update * self.scale).to(result.dtype)


class Projector(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.projector(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(t.device)
        angles = t[:, None].float() * frequencies[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), -1).to(self.mlp[0].weight.dtype))


class ActionOutput(nn.Module):
    def __init__(self, dim, action_dim):
        super().__init__()
        self.norm_final = FloatLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, action_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t[:, None]).chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class Stream(nn.Module):
    def __init__(self, input_dim, dim, action_dim=None):
        super().__init__()
        self.embed = nn.Linear(input_dim, dim)
        if action_dim is not None:
            self.time_embed = TimeEmbedding(dim)
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            self.head = ActionOutput(dim, action_dim)


class StreamBlock(nn.Module):
    def __init__(self, dim, heads, expansion, qk_norm, action):
        super().__init__()
        self.heads = heads
        self.norm1 = FloatLayerNorm(dim, elementwise_affine=not action, eps=1e-6)
        self.norm2 = FloatLayerNorm(dim, elementwise_affine=not action, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim // heads) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.mlp = SwiGLU(dim, expansion)
        self.mod_bias = nn.Parameter(torch.zeros(1, 6, dim)) if action else None

    def prepare(self, x, rope, modulation=None):
        h = self.norm1(x)
        mods = None
        if modulation is not None:
            mods = (modulation + self.mod_bias).unsqueeze(2).unbind(1)
            h = h * (1 + mods[1]) + mods[0]
        q, k, v = self.qkv(h).reshape(*h.shape[:2], 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        return apply_rope(q, rope), apply_rope(k, rope), v, mods

    def finish(self, x, attention, mods):
        h = self.proj(attention.transpose(1, 2).flatten(2))
        if mods is None:
            x = x + h
            return x + self.mlp(self.norm2(x))
        x = x + mods[2] * h
        return x + mods[5] * self.mlp(self.norm2(x) * (1 + mods[4]) + mods[3])


def image_time_positions(types, first_t, history_t, relative=False):
    initial = torch.isin(types, types.new_tensor(INITIAL_TYPES))
    positions = torch.where(initial, first_t, 0)
    history_t = [history_t] * 3 if isinstance(history_t, int) else history_t
    if len(history_t) != 3 or any(type(t) is not int for t in history_t):
        raise ValueError("rope.history_t must be an integer or three integers")
    for group, value in zip(HISTORY_TYPES, history_t):
        if value != 0:
            value = max(first_t, value) if relative else value
            positions = torch.where(torch.isin(types, types.new_tensor(group)), value, positions)
    return positions


def rope_coefficients(types, patches, head_dim, config, first_t, relative=False):
    image = patches[..., 0] >= 0
    running = ((~image).long().cumsum(1) - 1).clamp(min=0)
    action = types == -2
    running = torch.where(action, (action.long().cumsum(1) - 1).clamp(min=0) + config["action_start"], running)
    running = torch.where(types == 8, config["action_start"] - 1, running)
    image_t = image_time_positions(types, first_t, config.get("history_t", 0), relative)
    positions = torch.stack((torch.where(image, image_t, running),
                             torch.where(image, patches[..., 0], running),
                             torch.where(image, patches[..., 1], running)), -1).float()
    half = head_dim // 2
    spatial = {64: 10, 128: 20}.get(head_dim, half // 3)
    axes = torch.arange(half, device=types.device) % 3
    axes[3 * spatial:] = 0
    frequencies = 1.0 / config.get("base", 10000) ** (
        torch.arange(half, device=types.device, dtype=torch.float32) / half)
    angles = positions[..., axes] * frequencies
    return angles.cos(), angles.sin()


def apply_rope(x, coefficients):
    # Round only the completed rotation, preserving the FP32 frequencies.
    c, s = (v[:, None].float() for v in coefficients)
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2).to(x.dtype)


class MoTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.streams = nn.ModuleDict({name: StreamBlock(
            config["hidden_size"], config["num_heads"], config["mlp_ratio"], config["qk_norm"], name == "action")
            for name in ("condition", "action")})
        for branch, cfg in config.get("lora", {}).items():
            if not cfg.get("enabled", False):
                continue
            if branch not in self.streams:
                raise ValueError(f"Unknown MoT LoRA branch: {branch}")
            targets = cfg["target_modules"]
            if type(cfg["r"]) is not int or cfg["r"] < 1 or not targets or len(set(targets)) != len(targets):
                raise ValueError("LoRA requires a positive rank and distinct target modules")
            stream = self.streams[branch]
            for target in targets:
                module = stream.get_submodule(target)
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"LoRA target {branch}.{target} must be linear")
                parent, _, name = target.rpartition(".")
                setattr(stream.get_submodule(parent) if parent else stream, name,
                        LoRALinear(module, cfg["r"], cfg["alpha"]))


class MoT(nn.Module):
    def __init__(self, config, token_dim, action_dim, attention=None):
        super().__init__()
        self.attention = attention if attention is not None else AttentionBackend(config.get("attn_backend", "fa4"))
        dim = config["hidden_size"]
        self.blocks = nn.ModuleList([MoTLayer(config) for _ in range(config["depth"])])
        self.streams = nn.ModuleDict({"condition": Stream(token_dim, dim),
                                      "action": Stream(2 * action_dim, dim, action_dim)})

    def condition_cache(self, features, mask, rope):
        x = self.streams["condition"].embed(features).masked_fill(~mask[..., None], 0)
        cache = []
        layout = self.attention.layout(mask, mask)
        for layer in self.blocks:
            block = layer.streams["condition"]
            q, k, v, _ = block.prepare(x, rope)
            cache.append((k, v))
            attention = self.attention(q, k, v, mask, layout)
            x = block.finish(x, attention, None).masked_fill(~mask[..., None], 0)
        return cache

    def predict(self, x, x_mask, t, auxiliary, cache, condition_mask, rope):
        stream = self.streams["action"]
        valid = x_mask.bool().any(-1)
        x = stream.embed(torch.cat((x * x_mask, x_mask), -1).to(stream.embed.weight.dtype))
        x = x.masked_fill(~valid[..., None], 0)
        t = stream.time_embed(t) + auxiliary
        modulation = stream.modulation(t).reshape(len(x), 6, -1)
        mask = torch.cat((condition_mask, valid), 1)
        layout = self.attention.layout(valid, mask)
        for layer, (ck, cv) in zip(self.blocks, cache):
            block = layer.streams["action"]
            q, k, v, mods = block.prepare(x, rope, modulation)
            attention = self.attention(q, torch.cat((ck, k), 2), torch.cat((cv, v), 2), mask, layout)
            x = block.finish(x, attention, mods).masked_fill(~valid[..., None], 0)
        return stream.head(x, t).masked_fill(~valid[..., None], 0)


class ActionHead(nn.Module):
    def __init__(self, config, token_dim, action_dim, state_dim, attention=None):
        super().__init__()
        dim = config["hidden_size"]
        self.null_aux_embedding = nn.Parameter(torch.zeros(1, dim))
        self.aux_embedder = nn.Sequential(nn.Linear(2 * state_dim + 6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.mot = MoT(config, token_dim, action_dim, attention)


class CogACT2(nn.Module):
    def __init__(self, checkpoint, backbone_config, attention=None):
        super().__init__()
        self.attention = attention if attention is not None else AttentionBackend()
        self.checkpoint = checkpoint
        m = checkpoint.model
        self.layers = m["vlm_feature_layer"]
        depth = max(self.layers)
        if depth >= backbone_config.text_config.num_hidden_layers:
            raise ValueError("This inference path requires intermediate Qwen feature layers")
        backbone_config.text_config.num_hidden_layers = depth
        backbone_config.text_config.layer_types = backbone_config.text_config.layer_types[:depth]
        self.backbone = AutoModelForImageTextToText.from_config(
            backbone_config, attn_implementation=self.attention.hf_implementation, dtype=torch.float32)
        dim = backbone_config.text_config.hidden_size
        self.vlm_feature_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.layers])
        self.vlm_feature_proj = nn.Linear(dim * len(self.layers), dim)
        self.vlm_feature_post_norm = nn.LayerNorm(dim)
        self.state_encoder = Projector(2 * checkpoint.dim("state"), dim)
        self.extra_encoders = nn.ModuleDict({"fov": Projector(2, dim)})
        self.action_head = ActionHead(m["act_head"]["flow_matching_config"]["net"], dim,
                                      checkpoint.dim("action"), checkpoint.dim("state"), self.attention)

    @torch.inference_mode()
    def sample(self, tokens, types, patches, system_length, state, state_mask, fov, seed, steps,
               first_frame_offset=0):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        tokens = {k: v.to(device) for k, v in tokens.items() if k in
                  ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        types, patches = types.to(device), patches.to(device)
        tokens["pixel_values"] = tokens["pixel_values"].to(dtype)
        ids = tokens.pop("input_ids")
        tokens["attention_mask"] = tokens["attention_mask"].bool()
        positions, _ = self.backbone.model.get_rope_index(
            ids, (ids == self.backbone.config.image_token_id).int(), tokens["image_grid_thw"], None,
            attention_mask=tokens["attention_mask"].long())
        captured = []
        handle = self.backbone.model.language_model.layers[-1].register_forward_hook(
            lambda module, args, output: captured.append(output))
        try:
            outputs = self.backbone.model(inputs_embeds=self.backbone.get_input_embeddings()(ids),
                                           position_ids=positions, **tokens, output_hidden_states=True, use_cache=False)
        finally:
            handle.remove()
        hidden = outputs.hidden_states[:-1] + (captured[0],)
        features = self.vlm_feature_post_norm(self.vlm_feature_proj(torch.cat([
            norm(hidden[layer][:, system_length:]) for norm, layer in zip(self.vlm_feature_norms, self.layers)], -1)))
        types, patches = types[:, system_length:], patches[:, system_length:]
        budget = self.checkpoint.model["act_head"]["action_feature_length"]
        if features.shape[1] > budget:
            raise ValueError(f"Observation needs {features.shape[1]} condition tokens; checkpoint budget is {budget}")
        length = features.shape[1]
        features = F.pad(features, (0, 0, 0, budget - length))
        types = F.pad(types, (0, budget - length), value=-1)
        patches = F.pad(patches, (0, 0, 0, budget - length), value=-1)
        state, state_mask, fov = [torch.as_tensor(v, device=device, dtype=dtype).unsqueeze(0)
                                  for v in (state, state_mask, fov)]
        state_pair = torch.cat((state * state_mask, state_mask), -1)
        features = torch.cat((self.state_encoder(state_pair)[:, None], self.extra_encoders["fov"](fov), features), 1)
        types = torch.cat((types.new_tensor([[8, 9, 9, 9]]), types), 1)
        patches = torch.cat((patches.new_full((1, 4, 2), -1), patches), 1)
        mask = torch.arange(budget + 4, device=device)[None] < length + 4
        m = self.checkpoint.model
        net = m["act_head"]["flow_matching_config"]["net"]
        rope_cfg = m["act_head"]["position_embedding"]["rope"]
        head_dim = net["hidden_size"] // net["num_heads"]
        first_t = self.checkpoint.memory.first_t
        relative = first_t == "relative"
        if relative:
            if type(first_frame_offset) is not int or first_frame_offset > 0:
                raise ValueError("Relative first-frame time requires a nonpositive integer frame offset")
            first_t = first_frame_offset / 25.0
        cond_rope = rope_coefficients(types, patches, head_dim, rope_cfg, first_t, relative)
        cache = self.action_head.mot.condition_cache(features, mask, cond_rope)
        auxiliary = self.action_head.aux_embedder(torch.cat((state_pair, fov.flatten(1)), -1))
        chunk, dim = self.checkpoint.chunk_size, self.checkpoint.dim("action")
        # Keep noise, predictions and Euler arithmetic in FP32 under autocast.
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((1, chunk, dim), generator=generator, device=device, dtype=torch.float32)
        x_mask = torch.as_tensor(self.checkpoint.action_mask(), device=device, dtype=torch.float32)[None]
        action_rope = rope_coefficients(types.new_full((1, chunk), -2), patches.new_full((1, chunk, 2), -1),
                                        head_dim, rope_cfg, first_t, relative)
        for i in range(steps):
            t, previous = 1 - i / steps, 1 - (i + 1) / steps
            prediction = self.action_head.mot.predict(x, x_mask, torch.full((1,), t * 1000, device=device),
                                                       auxiliary, cache, mask, action_rope).float()
            x = x - (t - previous) * ((x - prediction) / t)
        return x[0].float().cpu().numpy()


class ImageTextProcessor:
    def __init__(self, name, memory):
        self.processor = AutoProcessor.from_pretrained(name)
        self.memory = memory

    def __call__(self, images, kinds, instruction):
        tokens = self.processor.apply_chat_template(
            self.memory.messages(images, kinds, instruction),
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        ids = tokens["input_ids"]
        tokenizer = self.processor.tokenizer
        system_length = int((ids[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")).nonzero()[1])
        image_ids = (ids[0] == tokenizer.convert_tokens_to_ids("<|image_pad|>")).nonzero().flatten()
        types = torch.full_like(ids, 4)
        patches = torch.full((*ids.shape, 2), -1, dtype=torch.long)
        offset = 0
        merge = self.processor.image_processor.merge_size
        for kind, (t, h, w) in zip(kinds, tokens["image_grid_thw"].tolist()):
            if t != 1:
                raise ValueError("Only still-image observations are supported")
            h, w = h // merge, w // merge
            indices = image_ids[offset:offset + h * w]
            types[0, indices] = kind
            patches[0, indices] = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)
            offset += h * w
        if offset != len(image_ids):
            raise ValueError("Processor image grid does not match image tokens")
        return tokens, types, patches, system_length


def load_network(checkpoint, backbone_path=None, use_ema=False, device="cuda",
                 attn_backend="fa4", auto_cast_bf16=True):
    from accelerate import init_empty_weights

    name = backbone_path or checkpoint.model["vl_encoders"][0]["model_name_or_path"]
    config = AutoConfig.from_pretrained(name)
    processor = ImageTextProcessor(name, checkpoint.memory)
    compute_dtype = torch.bfloat16 if auto_cast_bf16 else torch.float32
    attention = AttentionBackend(attn_backend, device, compute_dtype)
    with init_empty_weights():
        model = CogACT2(checkpoint, config, attention)
    filename = "weights_ema.pt" if use_ema else "weights.pt"
    weights = torch.load(checkpoint.path / filename, map_location="cpu", weights_only=True, mmap=True)
    renamed = {}
    for key, value in weights.items():
        if key.startswith("backbone_list.0.backbone."):
            key = "backbone." + key.removeprefix("backbone_list.0.backbone.")
        elif key.startswith("act_head.net."):
            key = "action_head." + key.removeprefix("act_head.net.")
        renamed[key] = value
    model.load_state_dict(renamed, strict=True, assign=True)
    # Preserve checkpoint values: never round weights or buffers through BF16.
    # Autocast controls compute precision independently of parameter storage.
    model.to(device=device, dtype=torch.float32)
    return model.eval().requires_grad_(False), processor
