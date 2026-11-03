import logging

logger = logging.getLogger(__name__)


def register_fuser_grad_hooks(fuser, grad_cache):
    hook_targets = [
        ("gate.0.weight", fuser.gate[0].weight),
        ("gate.0.bias",   fuser.gate[0].bias),
        ("gate.2.weight", fuser.gate[2].weight),
        ("gate.2.bias",   fuser.gate[2].bias),
        ("norm.weight",   fuser.norm.weight),
        ("norm.bias",     fuser.norm.bias),
    ]
    registered = 0
    skipped = 0
    for name, param in hook_targets:
        if not param.requires_grad:
            skipped += 1
            continue

        def _make_hook(n):
            def _hook(grad):
                g = grad.float()
                grad_cache[n] = {
                    "norm": g.norm(2).item(),
                    "abs_mean": g.abs().mean().item(),
                    "max": g.abs().max().item(),
                }
            return _hook
        param.register_hook(_make_hook(name))
        registered += 1
    logger.info(
        "Registered gradient hooks on %d fuser parameters; skipped %d frozen parameters",
        registered,
        skipped,
    )


def log_fuser_gradients(fuser, grad_cache, global_step, cotrain_phase):
    if not grad_cache:
        return

    fuser_params = {
        "gate.0.weight": fuser.gate[0].weight,
        "gate.0.bias":   fuser.gate[0].bias,
        "gate.2.weight": fuser.gate[2].weight,
        "gate.2.bias":   fuser.gate[2].bias,
        "norm.weight":   fuser.norm.weight,
        "norm.bias":     fuser.norm.bias,
    }
    for fname, fparam in fuser_params.items():
        gc = grad_cache.get(fname, None)
        if gc is not None:
            fg_norm = gc["norm"]
            fg_abs_mean = gc["abs_mean"]
            fg_max = gc["max"]
        else:
            fg_norm = fg_abs_mean = fg_max = -1.0
        logger.info(
            "[step %d][ph=%s] fuser.%s  grad_norm=%.6e  grad_abs_mean=%.6e  grad_max=%.6e  "
            "param_norm=%.6e  requires_grad=%s",
            global_step, cotrain_phase, fname,
            fg_norm, fg_abs_mean, fg_max,
            fparam.data.float().norm(2).item(),
            fparam.requires_grad,
        )
    grad_cache.clear()


def log_vlm_token_composition(vlm_inputs, tokenizer, global_step, captions=None):
    ids = vlm_inputs.get("input_ids", None)
    if ids is None:
        return

    pad_id = tokenizer.pad_token_id
    img_ids = {151655, 248056}
    vid_ids = {151656, 248057}
    total = ids.numel()
    n_pad = (ids == pad_id).sum().item() if pad_id is not None else 0
    n_img = sum((ids == tid).sum().item() for tid in img_ids)
    n_vid = sum((ids == tid).sum().item() for tid in vid_ids)
    n_text = total - n_pad - n_img - n_vid
    logger.info(
        "[step %d] VLM input_ids: shape=%s  pad=%d  img=%d  vid=%d  text/special=%d  caption=%r",
        global_step, list(ids.shape), n_pad, n_img, n_vid, n_text,
        captions[0] if captions else "N/A",
    )
