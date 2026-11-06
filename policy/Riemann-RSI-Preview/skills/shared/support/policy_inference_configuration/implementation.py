def _load_train_config(model_path: str, exp_dir: str) -> dict:
    """Load the toml describing how this checkpoint was trained.

    ``skyrobot.eval.utils.load_train_config`` globs exactly two directories
    above the checkpoint, which only holds for the trainer's own
    ``<exp>/hf_model/step-N`` layout. A dump copied out for deployment usually
    has ``<exp>/step-N``, where that path lands one level ABOVE exp_dir, finds
    no toml and silently returns {} - num_latents, video size and (worst)
    norm_mode then fall back to the embodiment defaults and the policy
    denormalizes actions with the wrong math, which looks like a bad
    checkpoint rather than a config bug. So search exp_dir first, and treat a
    missing toml as fatal instead of degrading silently.
    """
    import glob
    import tomllib
    seen = []
    for cand in (exp_dir, os.path.dirname(os.path.normpath(str(model_path)))):
        if not cand or cand in seen:
            continue
        seen.append(cand)
        tomls = sorted(glob.glob(os.path.join(str(cand), '*.toml')))
        if tomls:
            with open(tomls[0], 'rb') as f:
                cfg = tomllib.load(f)
            print(f'[SkyRobot_WAM] training config: {tomls[0]}')
            return cfg
    raise FileNotFoundError(f'no training *.toml found in {seen!r}. It carries num_latents, video size and norm_mode; without it they silently fall back to the embodiment defaults and the policy emits garbage actions. Point deploy.yml train_config_path at the toml, or fix exp_dir.')
