import torch

def get_mesh_id(f, h, w, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)

    return grid_id

def print_model_info(model):
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")