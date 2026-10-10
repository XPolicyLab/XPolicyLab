"""Cache only non-learned vision geometry; recompute learned embeddings each call."""
from collections import OrderedDict
from types import MethodType

import torch


def build_vision_geometry(grids, num_grid_per_side, merge_size):
    indices, weights, permutations, rotary_ids, lengths = [], [], [], [], []
    offset = 0
    for frames, height, width in grids:
        h = torch.linspace(0, num_grid_per_side - 1, height)
        w = torch.linspace(0, num_grid_per_side - 1, width)
        h0, w0 = h.int(), w.int()
        h1 = (h0 + 1).clip(max=num_grid_per_side - 1)
        w1 = (w0 + 1).clip(max=num_grid_per_side - 1)
        dh, dw = h - h0, w - w0
        indices.append(torch.stack([
            (hi[:, None] * num_grid_per_side + wi[None, :]).flatten()
            for hi, wi in ((h0, w0), (h0, w1), (h1, w0), (h1, w1))
        ]))
        weights.append(torch.stack([
            (wh[:, None] * ww[None, :]).flatten()
            for wh, ww in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))
        ]))
        permutation = torch.arange(height * width).repeat(frames).view(
            frames, height // merge_size, merge_size, width // merge_size, merge_size
        ).permute(0, 1, 3, 2, 4).flatten()
        permutations.append(permutation + offset)
        rotary_ids.append(torch.stack((permutation // width, permutation % width), dim=-1))
        lengths.extend([height * width] * frames)
        offset += height * width
    return dict(indices=torch.cat(indices, dim=1).long(), weights=torch.cat(weights, dim=1),
                permutation=torch.cat(permutations), rotary_ids=torch.cat(rotary_ids),
                lengths=tuple(lengths), max_hw=max(max(h, w) for _, h, w in grids))


def _geometry(visual, grid_thw):
    if grid_thw.device.type != 'cpu':
        raise ValueError('Cached vision geometry requires the CPU grid prepared at the vision entrypoint.')
    grids = tuple(tuple(row) for row in grid_thw.tolist())
    weight = visual.pos_embed.weight
    key = (grids, visual.num_grid_per_side, visual.spatial_merge_size, weight.device, weight.dtype)
    cache = visual._o1_geometry_cache
    if key not in cache:
        layout = build_vision_geometry(grids, visual.num_grid_per_side, visual.spatial_merge_size)
        for name in ('indices', 'permutation', 'rotary_ids'):
            layout[name] = layout[name].to(weight.device)
        layout['weights'] = layout['weights'].to(device=weight.device, dtype=weight.dtype)
        cache[key] = layout
        if len(cache) > 16:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def _interpolate(self, grid_thw):
    layout = _geometry(self, grid_thw)
    # Preserve the pretrained interpolation's BF16 multiply/add order.
    values = self.pos_embed(layout['indices']) * layout['weights'][:, :, None]
    positions = values[0] + values[1] + values[2] + values[3]
    return positions.index_select(0, layout['permutation'])


def _rotary(self, grid_thw):
    layout = _geometry(self, grid_thw)
    return self.rotary_pos_emb(layout['max_hw'])[layout['rotary_ids']].flatten(1)


def install_cached_vision_geometry(visual):
    original_forward = visual.forward
    visual._o1_geometry_cache = OrderedDict()
    visual.fast_pos_embed_interpolate = MethodType(_interpolate, visual)
    visual.rot_pos_emb = MethodType(_rotary, visual)

    def forward(hidden_states, grid_thw, **kwargs):
        cpu_grid = grid_thw.detach().cpu()
        layout = _geometry(visual, cpu_grid)
        return original_forward(hidden_states, cpu_grid, o1_image_lengths=layout['lengths'], **kwargs)

    visual.forward = forward
