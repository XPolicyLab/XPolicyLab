def __init__(self, model_cfg):
    self.model_cfg = model_cfg
    skyrobot_root = str(model_cfg.get('skyrobot_root') or '')
    if not os.path.isdir(skyrobot_root):
        raise ValueError(f'deploy.yml skyrobot_root not found: {skyrobot_root!r}')
    if skyrobot_root not in sys.path:
        sys.path.insert(0, skyrobot_root)
    import torch
    from skyrobot.data.robot_config.robodojo import RoboDojo as RoboDojoCfg
    from skyrobot.deploy.policy import SkyrobotWAMPolicy
    model_path = _resolve_model_path(model_cfg)
    exp_dir = model_cfg.get('exp_dir') or os.path.normpath(os.path.join(model_path, '..', '..'))
    norm_stats_path = model_cfg.get('norm_stats_path') or os.path.join(str(exp_dir), 'action_norm_stats.jsonl')
    _verify_norm_stats(norm_stats_path, model_cfg['norm_stats_sha256'])
    encoder_path = model_cfg.get('encoder_path')
    if not encoder_path or not os.path.isdir(str(encoder_path)):
        raise ValueError(f'deploy.yml encoder_path invalid: {encoder_path!r} (needs Wan2.2_VAE.pth + T5 weights)')
    explicit_toml = model_cfg.get('train_config_path')
    if explicit_toml:
        import tomllib
        with open(str(explicit_toml), 'rb') as _f:
            train_cfg = tomllib.load(_f)
        print(f'[SkyRobot_WAM] training config: {explicit_toml}')
    else:
        train_cfg = _load_train_config(model_path, str(exp_dir))
    train = train_cfg.get('training', {})
    data = train_cfg.get('data', {})
    if data.get('norm_mode') != model_cfg['norm_mode']:
        raise ValueError('Normalization mode differs from the training TOML')
    data_cfg = RoboDojoCfg()
    data_cfg.video_height = data.get('video_height', data_cfg.video_height)
    data_cfg.video_width = data.get('video_width', data_cfg.video_width)
    data_cfg.frame_compression = train.get('frame_compression', data_cfg.frame_compression)
    data_cfg.num_latents = int(model_cfg.get('num_latents') or train.get('num_latents', data_cfg.num_latents))
    data_cfg.frame_per_latent = train.get('vae_stride', [data_cfg.frame_per_latent])[0]
    self.data_cfg = data_cfg
    device = str(model_cfg.get('device') or 'cuda')
    t5_device = model_cfg.get('t5_device')
    t5_device = str(t5_device) if t5_device else None
    _n_gpu = torch.cuda.device_count()
    for (_name, _dev) in (('device', device), ('t5_device', t5_device)):
        if _dev and _dev.startswith('cuda:') and (int(_dev.split(':')[1]) >= _n_gpu):
            raise ValueError(f"deploy.yml {_name}={_dev!r} but only {_n_gpu} GPU(s) are visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}). The policy server must expose both the policy GPU and the T5 GPU — see setup_eval_policy_server.sh.")
    apf = data_cfg.frame_per_latent * data_cfg.frame_compression
    print(f'[SkyRobot_WAM] model_path={model_path}')
    print(f"[SkyRobot_WAM] device={device} t5_device={t5_device or device} visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[SkyRobot_WAM] apf={apf} num_latents={data_cfg.num_latents} video_hw=({data_cfg.video_height},{data_cfg.video_width}) norm_mode={data.get('norm_mode', 'zscore')}")
    self.policy = SkyrobotWAMPolicy(model_path=model_path, encoder_path=str(encoder_path), norm_stats_path=norm_stats_path, embodiment_cfg=data_cfg, num_inference_steps=int(model_cfg.get('num_inference_steps', 16)), action_num_train_timesteps=train.get('action_num_train_timesteps', 1000), action_snr_shift=train.get('action_snr_shift', 1.0), norm_mode=data['norm_mode'], cfg_scale=float(model_cfg.get('cfg_scale', 1.0)), cfg_uncond_prompt=train.get('cfg_uncond_prompt', 'Perform the default behavior.'), device=device, t5_cpu=bool(model_cfg.get('t5_cpu', False)), t5_device=t5_device, t5_lazy=bool(model_cfg.get('t5_lazy', False)), t5_unload_after_encode=bool(model_cfg.get('t5_unload_after_encode', False)), verbose=bool(model_cfg.get('verbose', False)))
    warmup_prompts = model_cfg.get('t5_warmup_prompts_by_task', {}).get(model_cfg.get('task_name'), [])
    if warmup_prompts:
        for prompt in warmup_prompts:
            self.policy._encode_text_cached(self.policy._wrap_prompt(prompt))
        if self.policy.cfg_scale != 1.0:
            self.policy._encode_text_cached(self.policy.cfg_uncond_prompt)
        self.policy.infer_model.unload_text_encoder()
        print('[SkyRobot_WAM] Task text cached; T5 unloaded before simulator startup')
    self._pending_reset = True
    self._latest = None
    print('[SkyRobot_WAM] Model successfully initialized (action_type=ee)')
