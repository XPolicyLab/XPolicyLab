def _stitch_image(self, obs: dict) -> np.ndarray:
    """3 runtime cams -> inverted-T combined frame [Hc, Wc, 3] uint8.

        Uses the SAME combine_views as the training dataset (RoboDojo config)
        so the eval frame layout can never drift from training.
        """
    import torch
    vision = obs.get('vision') or {}
    video_dict = {}
    for (env_key, ds_key) in _CAM_ENV_TO_DS.items():
        cam = vision.get(env_key)
        if cam is None:
            raise KeyError(f'obs missing camera {env_key!r}; have {list(vision)}')
        img = np.asarray(cam['color'] if isinstance(cam, dict) else cam)
        if img.ndim != 3 or img.shape[-1] < 3:
            raise ValueError(f'cam {env_key!r}: expected HWC RGB, got {img.shape}')
        img = img[:, :, :3]
        if img.dtype != np.uint8:
            arr = img.astype(np.float32)
            if arr.max() <= 1.5:
                arr = arr * 255.0
            img = arr.clip(0, 255).astype(np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        video_dict[ds_key] = t
    combined = self.data_cfg.combine_views(video_dict)
    (vh, vw) = (self.data_cfg.video_height, self.data_cfg.video_width)
    if combined.shape[-2:] != (vh, vw):
        combined = self.data_cfg.resize(combined, vh, vw)
    return combined[0].clamp(0, 1).mul_(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
