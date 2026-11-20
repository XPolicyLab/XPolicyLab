# Benchmark-agnostic SkyrobotWAM online policy


from __future__ import annotations

import json
import time
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    # type-only import (no runtime cost / cycle); embodiment_cfg is one of the
    # DATA_CONFIG entries, all base_data_config subclasses (e.g. RiEMann_Dynamics).
    from skyrobot.data.robot_config.base_config import base_data_config


def _load_norm_stats(path: str, embodiment: str | None = None) -> dict:
    """Read action_norm_stats.jsonl and return the entry for `embodiment`.

    The trainer writes one JSON per line, one per embodiment.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    if embodiment is None:
        raise ValueError(f"The provided embodiment is {embodiment}, please specify a valid embodiment.")
    matches = [r for r in rows if r.get("embodiment") == embodiment]
    if not matches:
        embs = [r.get("embodiment") for r in rows]
        raise ValueError(
            f"embodiment={embodiment!r} not found in {path}; "
            f"have {embs!r}")
    return matches[0]


class SkyrobotWAMPolicy:
    """Online video-conditioned action policy for SkyrobotWAM.

    Decoupled from any specific benchmark — caller provides:

      reset(image, state, task_text):
          image: [H, W, 3] uint8 — conditioning frame.
          state: [active_dim]    raw float32 (un-normalized).

      act(image, state) -> action:
          image: [H, W, 3] uint8 — current env frame.
          state: [active_dim]    raw float32.
          action: [active_dim]   raw float32 (z-score-denormalized; ready
                                  for env consumption).

    Internally manages:
      - Wan2.2 VAE encode of conditioning + observation frames
      - T5 text encode for cross-attention
      - OnlineSession KV cache (auto resets every num_latents frames)
      - State z-score normalize / action denormalize
      - Frame queue: every `action_per_latent` env steps -> 1 observe call
      - Action queue: each diffusion call returns `apf` actions; we hand
        them out one per env step.
    """

    def __init__(
        self,
        model_path: str,
        encoder_path: str,
        norm_stats_path: str,
        embodiment_cfg: "base_data_config",
        embodiment_id: Optional[int] = None,
        num_inference_steps: int = 16,
        action_num_train_timesteps: int = 1000,
        action_snr_shift: float = 1.0,
        norm_mode: str = "zscore",            # "zscore" | "minmax_quantile" | "minmax"
        norm_clamp_abs: float = 5,
        relative_action: Optional[bool] = None,  # None → read from norm stats
        cfg_scale: float = 1.0,
        cfg_uncond_prompt: str = "Perform the default behavior.",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        t5_cpu: bool = False,
        t5_device: Optional[str] = None,
        t5_lazy: bool = False,
        t5_unload_after_encode: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: hf_model/step-N dir (config.json + safetensors).
            encoder_path: dir with Wan2.2_VAE.pth + T5 weights/tokenizer.
            norm_stats_path: action_norm_stats.jsonl produced at training.
            embodiment_cfg: a `base_data_config` instance (e.g. `RoboCasa_GR1()`).
                Provides video_height/_width, num_latents, frame_per_latent,
                frame_compression, embodiment_id, action_dim (active dim).
            embodiment_id: override raw EmbodimentId; defaults to
                embodiment_cfg.embodiment_id.
            num_inference_steps: diffusion steps per action chunk.
            action_num_train_timesteps / action_snr_shift: must match training.
            device: device for the 5B backbone + VAE (the hot path).
            t5_cpu: legacy offload flag — park umt5-xxl (~11.4GB) on CPU. Slow;
                prefer t5_device.
            t5_device: explicit device for umt5-xxl, e.g. "cuda:1". Lets a 24GB
                card hold the backbone while the text encoder sits on a second
                card (typically the one running the simulator).
            t5_lazy: defer reading the T5 weights until the first encode.
            t5_unload_after_encode: free the T5 weights right after each encode
                and rebuild on the next cache miss. Combined with the per-prompt
                embedding cache this means T5 VRAM is transient: a single-task
                eval server pays it once. Use when the T5 card is tight.
            verbose: print per-call timing.
        """
        self.cfg = embodiment_cfg
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        if norm_mode not in ("zscore", "minmax_quantile", "minmax"):
            raise ValueError(f"norm_mode must be zscore|minmax_quantile|minmax, got {norm_mode!r}")
        self.norm_mode = norm_mode
        self.norm_clamp_abs = float(norm_clamp_abs)
        # Classifier-free guidance (inference). cfg_scale==1.0 -> disabled.
        self.cfg_scale = float(cfg_scale)
        self.cfg_uncond_prompt = str(cfg_uncond_prompt)

        # Derived from embodiment config (all match training).
        self.apf = embodiment_cfg.frame_per_latent * embodiment_cfg.frame_compression
        self.frame_compression = embodiment_cfg.frame_compression
        self.num_latents = embodiment_cfg.num_latents
        self.embodiment_id = (embodiment_id if embodiment_id is not None
                              else int(embodiment_cfg.embodiment_id))
        self.active_action_dim = embodiment_cfg.action_dim
        self.num_inference_steps = num_inference_steps

        # Norm stats are stored in active-dim space (post action_indices /
        # state_indices selection). The trainer writes one JSON-line per
        # embodiment in action_norm_stats.jsonl, so we look up by name.
        emb_name = getattr(embodiment_cfg, "embodiment", None)
        stats = _load_norm_stats(norm_stats_path, embodiment=emb_name)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(stats["state_std"], dtype=np.float32)
        # Optional quantile / min-max stats; required when norm_mode is
        # 'minmax_quantile' or 'minmax' respectively.
        def _opt(name):
            return np.asarray(stats[name], dtype=np.float32) if name in stats else None
        self.action_q01 = _opt("action_q01")
        self.action_q99 = _opt("action_q99")
        self.state_q01  = _opt("state_q01")
        self.state_q99  = _opt("state_q99")
        self.action_min = _opt("action_min")
        self.action_max = _opt("action_max")
        self.state_min  = _opt("state_min")
        self.state_max  = _opt("state_max")
        if self.norm_mode == "minmax_quantile":
            missing = [k for k in ("action_q01","action_q99","state_q01","state_q99")
                       if stats.get(k) is None]
            if missing:
                raise RuntimeError(
                    f"norm_mode='minmax_quantile' but {missing} missing in "
                    f"{norm_stats_path}. Re-train (will save quantiles) or "
                    f"set norm_mode='zscore'."
                )
        if self.norm_mode == "minmax":
            missing = [k for k in ("action_min","action_max","state_min","state_max")
                       if stats.get(k) is None]
            if missing:
                raise RuntimeError(
                    f"norm_mode='minmax' but {missing} missing in "
                    f"{norm_stats_path}. Re-train (will save min/max) or "
                    f"set norm_mode='zscore'."
                )
        # Relative-action: actions are stored relative to the per-latent anchor
        # state (see lerobot_datasets / base_config). At inference we denormalize
        # then ADD the current (last-observed) state back on the relative dims —
        # all handled inside cfg.action_denormalize. The flag + dims are recorded
        # in the norm stats jsonl by the trainer; the ctor arg can override.
        # Action norm stats are already the relative (hybrid) stats.
        self.relative_action = (bool(stats.get("relative_action", False))
                                if relative_action is None else bool(relative_action))
        rel_dims = stats.get("relative_action_dims")
        if rel_dims is None and self.relative_action:
            rel_dims = self.cfg.relative_dims()   # fall back to embodiment config
        self.rel_idx = (np.asarray(rel_dims, dtype=np.int64)
                        if self.relative_action and rel_dims else None)
        if self.relative_action and self.rel_idx is None:
            raise RuntimeError(
                "relative_action enabled but no relative dims found in "
                f"{norm_stats_path} or embodiment config.")
        # Make cfg the single source of truth for normalization AND the relative
        # add-back: align its norm_mode / clamp / relative_action / dims with what
        # we resolved so cfg.{action,state}_{normalize,denormalize} use exactly
        # the same math as training (the config's own defaults — e.g. zscore —
        # would otherwise silently override the policy's norm_mode).
        self.cfg.norm_mode = self.norm_mode
        self.cfg.norm_clamp_abs = self.norm_clamp_abs
        self.cfg.relative_action = self.relative_action
        if self.relative_action:
            self.cfg.relative_action_dims = self.rel_idx.tolist()
        # anchor state (raw, active-dim) the current action block is relative to;
        # set at reset() and refreshed on every observe() in act().
        self._anchor_state = None

        # Per-step relative-action stats [H, active_dim] — used (and required)
        # only when relative_action is on. The trainer writes them when the
        # embodiment defines relative_action_dims; action at within-block offset
        # i is denormalized by rel_stat[i] (see _denormalize_action), matching
        # training. Non-relative dims already carry the absolute stats here.
        def _optv(name):
            v = stats.get(name)
            return np.asarray(v, dtype=np.float32) if v is not None else None
        self.rel_action_mean = _optv("rel_action_mean")
        self.rel_action_std  = _optv("rel_action_std")
        self.rel_action_q01  = _optv("rel_action_q01")
        self.rel_action_q99  = _optv("rel_action_q99")
        self.rel_action_min  = _optv("rel_action_min")
        self.rel_action_max  = _optv("rel_action_max")
        if self.relative_action:
            _need = {"zscore": ("rel_action_mean", "rel_action_std"),
                     "minmax_quantile": ("rel_action_q01", "rel_action_q99"),
                     "minmax": ("rel_action_min", "rel_action_max")}[self.norm_mode]
            _miss = [k for k in _need if stats.get(k) is None]
            if _miss:
                raise RuntimeError(
                    f"relative_action on but per-step {_miss} missing in "
                    f"{norm_stats_path}. Recompute stats (compute_action_stats.py) "
                    f"and re-train so the per-step rel stats are saved.")

        # Torch copies of the stats (cfg.* ops are torch); built once. Per-step
        # state/action vectors are tiny, so the numpy<->torch hop is negligible.
        def _t(a):
            return torch.from_numpy(a) if a is not None else None
        self._ts = dict(
            action_mean=_t(self.action_mean), action_std=_t(self.action_std),
            action_q01=_t(self.action_q01),   action_q99=_t(self.action_q99),
            action_min=_t(self.action_min),   action_max=_t(self.action_max),
            state_mean=_t(self.state_mean),   state_std=_t(self.state_std),
            state_q01=_t(self.state_q01),     state_q99=_t(self.state_q99),
            state_min=_t(self.state_min),     state_max=_t(self.state_max),
            rel_action_mean=_t(self.rel_action_mean), rel_action_std=_t(self.rel_action_std),
            rel_action_q01=_t(self.rel_action_q01),   rel_action_q99=_t(self.rel_action_q99),
            rel_action_min=_t(self.rel_action_min),   rel_action_max=_t(self.rel_action_max),
        )

        if self.norm_mode == "minmax":
            bounds_loaded = "min/max"
        elif self.norm_mode == "minmax_quantile":
            bounds_loaded = "q01/q99"
        else:
            bounds_loaded = "mean/std"
        print(f"[policy] norm stats for {emb_name!r}: "
              f"action_dim={len(self.action_mean)} state_dim={len(self.state_mean)} "
              f"norm_mode={self.norm_mode} "
              f"bounds={bounds_loaded} "
              f"relative_action={self.relative_action}"
              + (f"(dims={self.rel_idx.tolist()})" if self.relative_action else ""))

        # Lazy-import inference deps.
        from skyrobot.models.model import SkyrobotWAM
        from skyrobot.deploy.model_dp import SkyrobotWAMForInference
        from skyrobot.utils.scheduler import FlowMatchScheduler

        model = SkyrobotWAM.from_pretrained(model_path, torch_dtype=dtype)
        model = model.to(device).to(dtype).eval()
        self.infer_model = SkyrobotWAMForInference.from_trained_model(
            model, device=device, dtype=dtype,
            video_height=embodiment_cfg.video_height,
            video_width=embodiment_cfg.video_width,
        )
        self.t5_device = self.infer_model.resolve_t5_device(
            device, t5_cpu, t5_device)
        self.t5_unload_after_encode = bool(t5_unload_after_encode)
        self.infer_model.load_encoders(encoder_path, device=device,
                                       t5_cpu=t5_cpu, t5_device=t5_device,
                                       t5_lazy=t5_lazy)
        print(f"[policy] backbone/VAE on {device}, T5 on {self.t5_device}"
              + (" (unloaded after each encode)"
                 if self.t5_unload_after_encode else ""))
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_snr_shift,
            sigma_min=0.0, extra_one_step=True,
        )

        # Per-episode runtime state (initialized in reset).
        self.session = None
        self._action_queue = []
        self._frame_buffer = []
        self._step_count = 0
        # Encoded task text, cached at reset() so reset_context() can re-anchor
        # the session on a fresh frame without re-running T5 (used by num_latents
        # ==1 replan, where every prediction is anchored to one context frame).
        self._text_emb = None
        self._cond_kpm = None
        self._uncond_emb = None
        self._uncond_kpm = None
        # Wrapped-prompt -> (emb, kpm), so T5 runs ONCE per distinct prompt
        # instead of once per episode. An eval server is pinned to one task, so
        # in practice this is a single T5 call for the whole run. ~4MB/entry
        # ([1, 512, 4096] bf16), so an unbounded dict is fine here.
        self._text_cache = {}

    # ------------------------------------------------------------------
    # Normalization — delegated to base_config (single source of truth shared
    # with the training dataset). The relative-action subtract/add lives in
    # cfg.action_(de)normalize, so the two paths can never diverge.
    # ------------------------------------------------------------------

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        out = self.cfg.state_normalize(
            torch.from_numpy(np.array(state, np.float32)),
            mean=self._ts["state_mean"], std=self._ts["state_std"],
            q01=self._ts["state_q01"], q99=self._ts["state_q99"],
            x_min=self._ts["state_min"], x_max=self._ts["state_max"])
        return out.numpy()

    def _denormalize_action(self, actions_norm: np.ndarray) -> np.ndarray:
        """Denormalize a full predicted block -> raw actions.

        actions_norm: [apf, active_dim]. We work in [active_dim, apf] layout so
        the chunk offset is the step axis (step_axis=1): action at within-block
        offset i is denormalized by rel_stat[i] (per-step), then the block's
        anchor state is added back on the relative dims. All apf actions share
        self._anchor_state (the last-observed raw state). Math lives on
        cfg.action_denormalize — identical to training. Returns [apf, active_dim].
        """
        x = torch.from_numpy(np.asarray(actions_norm, np.float32)).transpose(0, 1)  # [A, apf]
        anchor = None
        if self.relative_action and self._anchor_state is not None:
            # [active_dim] -> [active_dim, 1] so it broadcasts over the apf offsets
            anchor = torch.from_numpy(self._anchor_state).unsqueeze(1)
        out = self.cfg.action_denormalize(
            x, anchor, step_axis=1,
            mean=self._ts["action_mean"], std=self._ts["action_std"],
            q01=self._ts["action_q01"], q99=self._ts["action_q99"],
            x_min=self._ts["action_min"], x_max=self._ts["action_max"],
            rel_mean=self._ts["rel_action_mean"], rel_std=self._ts["rel_action_std"],
            rel_q01=self._ts["rel_action_q01"], rel_q99=self._ts["rel_action_q99"],
            rel_min=self._ts["rel_action_min"], rel_max=self._ts["rel_action_max"])
        return out.transpose(0, 1).contiguous().numpy()   # [apf, active_dim]

    @staticmethod
    def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
        """[H, W, 3] uint8 -> [1, 3, H, W] float32 in [0, 1]."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return (torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div_(255.0))

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _wrap_prompt(self, task_text: str) -> str:
        """Wrap a raw instruction in DEFAULT_PROMPT, mirroring the training
        dataset (lerobot_datasets.__getitem__): embodiment name + camera views +
        the task. Keeps the conditional prompt identical to what the model saw
        during training. The uncond/CFG prompt stays raw (training dropout target)."""
        from skyrobot.data.data_config import DEFAULT_PROMPT
        embodiment = str(getattr(self.cfg, "embodiment", "")).replace("_", " ")
        cam_keys = getattr(self.cfg, "obs_cam_keys", None) or []
        views = ", ".join(k.split(".")[-1].replace("_", " ") for k in cam_keys)
        return DEFAULT_PROMPT.format(embodiment=embodiment, views=views,
                                     task=str(task_text))

    def reset(self, image: np.ndarray, state: np.ndarray, task_text: str):
        """Initialize a new episode session.

        Args:
            image: [H, W, 3] uint8 — conditioning frame at t=0.
            state: [active_dim] raw float32 — robot state at t=0.
            task_text: raw instruction; wrapped in DEFAULT_PROMPT (see
                `_wrap_prompt`) before encoding to match training.
        """
        # Conditional text (real instruction) + its key-padding mask. Wrap the
        # raw instruction in DEFAULT_PROMPT exactly as training does (the dataset
        # emits the wrapped string; CFG dropout swaps in the RAW cfg_uncond_prompt
        # — see wam_trainer / lerobot_datasets). Encoding the bare instruction
        # here would be a train/inference prompt mismatch.
        task_text = self._wrap_prompt(task_text)
        self._text_emb, self._cond_kpm = self._encode_text_cached(task_text)

        # Unconditional text for classifier-free guidance: the same fixed
        # neutral instruction used as the training dropout target. Goes
        # through the identical encode path -> its mask is just its normal
        # tokenizer mask (no special handling).
        self._uncond_emb = self._uncond_kpm = None
        if self.cfg_scale != 1.0:
            self._uncond_emb, self._uncond_kpm = self._encode_text_cached(
                self.cfg_uncond_prompt)

        # All prompts for this episode are encoded and cached — drop the T5
        # weights now (no-op if already unloaded / flag off). Deferred to here
        # so CFG's two encodes share one residency.
        if self.t5_unload_after_encode:
            self.infer_model.unload_text_encoder()

        self._build_session(image, state)

    def _encode_text_cached(self, prompt: str):
        """T5-encode `prompt` once and memoize (emb, key-padding mask).

        Episode reset is on the eval critical path and umt5-xxl is the single
        most expensive call in it (seconds on a second GPU, minutes on CPU).
        The prompt is constant for a whole task, so every reset after the first
        is served from this dict.
        """
        hit = self._text_cache.get(prompt)
        if hit is not None:
            return hit

        t0 = time.perf_counter()
        emb, kpm = self.infer_model.encode_text_with_mask([prompt], self.device)
        entry = (emb.to(self.dtype), kpm)
        self._text_cache[prompt] = entry
        if self.verbose or len(self._text_cache) == 1:
            print(f"[policy] T5 encode on {self.t5_device} took "
                  f"{(time.perf_counter() - t0)*1000:.0f}ms "
                  f"(cache size {len(self._text_cache)})")
        return entry

    def reset_context(self, image: np.ndarray, state: np.ndarray):
        """Re-anchor the session on a new context frame, reusing the task text
        already encoded by reset() (no T5 re-run).

        This is the replan primitive for `num_latents == 1`: in that regime the
        windowed KV cache is reset to a fresh single context frame on every
        observe() (model_dp.OnlineSession.observe, window-full branch), so each
        prediction is anchored to exactly one context frame and the model behaves
        like a stateless chunk policy. The caller runs the first `replan_steps`
        of each `apf`-action block, then calls this with the latest observation
        to predict a fresh block. Re-anchoring this way is NOT valid for
        num_latents > 1, where the model relies on the streaming multi-latent
        history — drive those with act() alone.
        """
        if self._text_emb is None:
            raise RuntimeError("Call reset(...) before reset_context(...).")
        self._build_session(image, state)

    def _build_session(self, image: np.ndarray, state: np.ndarray):
        """(Re)build the OnlineSession from a context frame + state, using the
        cached text embeddings. Shared by reset() and reset_context()."""
        self._action_queue = []
        self._frame_buffer = []
        self._step_count = 0

        first_frame = self._image_to_tensor(image)               # [1, 3, H, W]
        state_raw = np.asarray(state, np.float32)
        # first action block is relative to this state (session is built from
        # it); refreshed on each observe() in act().
        self._anchor_state = state_raw
        state_0 = self._normalize_state(state_raw)

        # OnlineSession handles state padding (active_dim -> MAX_STATE_DIM)
        # and active_action_dim slicing internally.
        self.session = self.infer_model.create_session(
            first_frame=first_frame,
            state_0=state_0,
            text_emb=self._text_emb,
            scheduler=self.scheduler,
            num_inference_steps=self.num_inference_steps,
            max_frames=self.num_latents,
            embodiment_id=self.embodiment_id,
            active_action_dim=self.active_action_dim,
            cond_text_kpm=self._cond_kpm,
            uncond_text_emb=self._uncond_emb,
            uncond_text_kpm=self._uncond_kpm,
            cfg_scale=self.cfg_scale,
        )

    def act(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Return next action.

        Args:
            image: [H, W, 3] uint8 — current env frame.
            state: [active_dim]   raw float32.

        Returns:
            action: [active_dim] float32 — denormalized, ready for env.
        """
        if self.session is None:
            raise RuntimeError("Call reset(...) before act(...).")

        # Buffer the current frame (every step).
        self._frame_buffer.append(np.asarray(image))
        self._step_count += 1

        # If a previous diffusion call still has queued actions, pop one.
        if self._action_queue:
            return self._action_queue.pop(0)

        # Otherwise: feed the last `apf` frames + current state to KV cache,
        # then run a new diffusion call to get `apf` future actions.
        t_start = time.perf_counter()
        t_observe = 0.0
        if self._step_count > 1 and len(self._frame_buffer) >= self.apf:
            # Match training latent[N] = VAE(o[16N-12], o[16N-8], o[16N-4], o[16N]).
            # The Wan2.2 VAE has 1+4 causal time compression and the dataset
            # subsamples the raw stream by frame_compression starting at position
            # 0 (the context frame). So per chunk N, we want the LAST four
            # subsampled frames of the apf-frame window (positions fc-1, 2fc-1,
            # 3fc-1, 4fc-1 in the buffer), not the first four.
            obs_frames_full = self._frame_buffer[-self.apf:]
            obs_frames = obs_frames_full[self.frame_compression - 1::self.frame_compression]
            frames_tensor = (torch.from_numpy(np.stack(obs_frames, 0))
                             .permute(0, 3, 1, 2)
                             .float()
                             .div_(255.0))   # [apf/fc, 3, H, W]
            state_raw = np.asarray(state, np.float32)
            # this newly observed state anchors the action block predicted below
            # (relative actions add it back after denormalizing).
            self._anchor_state = state_raw
            state_norm = self._normalize_state(state_raw)
            torch.cuda.synchronize()
            t_obs_start = time.perf_counter()
            self.session.observe(frames_tensor, state_norm)
            torch.cuda.synchronize()
            t_observe = time.perf_counter() - t_obs_start
            self._frame_buffer = []

        torch.cuda.synchronize()
        t_pred_start = time.perf_counter()
        actions_norm = self.session.predict_next_action()  # [apf, active_dim]
        torch.cuda.synchronize()
        t_predict = time.perf_counter() - t_pred_start

        if self.verbose:
            print(f"  [timing] observe={t_observe*1000:.0f}ms "
                  f"predict={t_predict*1000:.0f}ms "
                  f"total={(time.perf_counter()-t_start)*1000:.0f}ms "
                  f"(frame_idx={self.session.frame_idx})")

        # Denormalize the whole block per-step (offset i uses rel_stat[i]) and
        # queue. _denormalize_action also adds the block's anchor state back on
        # the relative dims when relative_action is on (all apf actions share
        # self._anchor_state, the last-observed state).
        actions_raw = self._denormalize_action(actions_norm)   # [apf, active_dim]
        self._action_queue = list(actions_raw[1:])
        return actions_raw[0]
