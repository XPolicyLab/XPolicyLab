def _resolve_model_path(cfg: dict) -> str:
    """Resolve the hf_model/step-N directory from deploy.yml + eval args.

    Priority: explicit ``model_path`` > ``ckpt_name`` as a path >
    ``<exp_dir>/hf_model/<ckpt_name>`` (``last`` picks the highest step).
    """
    explicit = cfg.get('model_path')
    if explicit:
        return str(explicit)
    ckpt = str(cfg.get('ckpt_name') or 'last')
    if os.path.isdir(ckpt):
        return ckpt
    exp_dir = cfg.get('exp_dir')
    if not exp_dir:
        raise ValueError('deploy.yml must set model_path, or exp_dir (+ ckpt_name/step)')
    hf_dir = os.path.join(str(exp_dir), 'hf_model')
    if not os.path.isdir(hf_dir):
        hf_dir = str(exp_dir)
    if ckpt in ('last', 'latest'):
        steps = [d for d in os.listdir(hf_dir) if d.startswith('step-') and os.path.isdir(os.path.join(hf_dir, d))]
        if not steps:
            raise FileNotFoundError(f'no step-* checkpoints in {hf_dir}')
        ckpt = max(steps, key=lambda s: int(s.split('-')[-1]))
    elif not ckpt.startswith('step-'):
        ckpt = f'step-{ckpt}'
    path = os.path.join(hf_dir, ckpt)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'checkpoint dir not found: {path}')
    return path
