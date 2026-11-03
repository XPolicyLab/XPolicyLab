"""Appearance perturbations for PICA-WAM's paired training views."""

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import torchvision.transforms as TF
import torchvision.transforms.functional as F


class VideoAugmentation(nn.Module):
    """Weak train-time augmentation for a camera clip in float [0, 1].

    The same sampled parameters are applied to every frame in the clip to keep
    temporal consistency within a camera view.
    """

    def __init__(
        self,
        p: float = 1.0,
        augment_types: list[str] | tuple[str, ...] = ("corrupt_only", "color_only", "both"),
        color_jitter: dict | None = None,
        gamma: dict | None = None,
        exposure: dict | None = None,
        gaussian_noise: dict | None = None,
        random_resized_crop: dict | None = None,
        rotate: dict | None = None,
        optical_corruption: dict | None = None,
        randconv: dict | None = None,
        canonical_geometry: str = "shared",
        canonical_random_resized_crop: dict | None = None,
        canonical_rotate: dict | None = None,
        geometric_p: float | None = None,
        allow_ungated_identity_pairs: bool = False,
    ):
        super().__init__()
        self.p = float(p)
        # SECOND, INDEPENDENT GATE FOR THE GEOMETRIC STAGE ONLY. `None` (the default) means
        # "there is no second gate": `p` keeps deciding the whole augmentation exactly as
        # before, no extra RNG draw is taken, and a run trains bit-identically to one
        # predating this parameter.
        #
        # WHY IT EXISTS. `p` is the master gate on EVERYTHING, but `_augment_frames_impl`
        # runs the geometric stage UNCONDITIONALLY once that gate fires. So `p` couples two
        # things that need different rates:
        #
        #   * the PHOTOMETRIC perturbation, which is what a canonical-teacher alignment pair
        #     is built from. At p=0.25 the `"shared"` pair is bit-identical on 75% of every
        #     batch (`_forward_impl`'s ungated branch returns the input twice), so the
        #     appearance gap the term is supposed to learn from is present on a quarter of
        #     the data and the objective degenerates to the layer/sigma pair elsewhere.
        #   * the GEOMETRIC displacement, which must stay BELOW 1.0. `R-OWN-032` priced
        #     p=1.0 at -17.33pp (p<1e-16, past the pre-registered -8pp gate) and the cause is
        #     precisely this coupling: at p=1.0 every sample is cropped and rotated, so the
        #     training set contains ZERO nominal-viewpoint clips while every eval axis but
        #     Camera Viewpoints is nominal.
        #
        # Setting `p=1.0, geometric_p=0.25` separates them: the pair carries an appearance
        # gap on 100% of samples while 75% keep the nominal viewpoint. Neither rate is
        # reachable with a single gate, which is the whole reason this is a parameter and not
        # a config value.
        self.geometric_p = None if geometric_p is None else float(geometric_p)
        self.augment_types = tuple(augment_types)
        self.color_jitter = color_jitter or {}
        self.gamma = gamma or {}
        self.exposure = exposure or {}
        self.gaussian_noise = gaussian_noise or {}
        self.random_resized_crop = random_resized_crop or {}
        self.rotate = rotate or {}
        self.optical_corruption = optical_corruption or {}
        self.randconv = randconv or {}
        # L4 guard: a config block present but with p unset (or 0) is a silent no-op --
        # the resolved yaml would document a treatment that never ran, which is exactly
        # how r1 trained a full run without its `weight_min_ratio`.
        for name, cfg in (("optical_corruption", self.optical_corruption),
                          ("randconv", self.randconv)):
            if cfg and float(cfg.get("p", 0.0) or 0.0) <= 0.0:
                raise ValueError(
                    f"`{name}` is configured but its `p` is unset or 0, so it would never "
                    f"fire. Set `{name}.p` or remove the block.")
        # How the canonical (second) clip's geometric stage is drawn. See
        # `_augment_frames_impl` for the full contract.
        #   "shared"      -- reuse the augmented branch's crop/rotation, so the pair
        #                    differs by photometry ONLY. Required by `appearance_path`
        #                    and `lambda_deaug`, which read the pair as an appearance
        #                    residual in latent space.
        #   "independent" -- the canonical branch draws its own crop/rotation. The pair
        #                    then differs by photometry AND viewpoint, which is what a
        #                    cross-view invariance term needs and what the ADF residual
        #                    cannot represent. `runtime._assert_canonical_clip_contract`
        #                    refuses to run ADF on top of this.
        self.canonical_geometry = str(canonical_geometry)
        # Explicit escape hatch for controlled, diluted canonical-teacher experiments such
        # as `p2_geo`: when the master gate does not fire, both branches intentionally see
        # the raw clip and the alignment target reduces to the ordinary layer/sigma pair.
        # Keeping this false by default prevents an accidental p<1 + pristine-teacher config
        # from silently claiming full cross-view coverage.
        self.allow_ungated_identity_pairs = bool(allow_ungated_identity_pairs)
        # OPTIONAL per-branch geometric distributions for `canonical_geometry="independent"`.
        # `None` (the default) means "reuse the augmented branch's blocks", which is the
        # behaviour every arm through ROUND 13 trained under and is bit-for-bit unchanged:
        # `_sample_canonical_geometry` only swaps when an override is actually supplied.
        #
        # Why this exists: the two branches feed DIFFERENT consumers. The augmented branch
        # drives `L_video` and `L_action`, so widening its geometry costs task accuracy; the
        # canonical branch is a no-grad anchor read only by the cross-view term. Letting the
        # anchor sit at (or near) the nominal viewpoint while the augmented branch is
        # displaced turns the objective from "align two random displacements" into "map a
        # displaced view onto a fixed reference", which is the direction inference actually
        # needs -- the filter only ever receives gradient through the augmented branch.
        #
        # ⚠ Keep `ratio: preserve` on both blocks. `_sample_random_resized_crop` takes a
        # different number of RNG draws on the non-preserve path, so mixing the two would
        # desynchronise the augmented branch's stream from a run without the override.
        self.canonical_random_resized_crop = canonical_random_resized_crop or {}
        self.canonical_rotate = canonical_rotate or {}
        if self.canonical_geometry not in {"shared", "independent", "none"}:
            raise ValueError(
                f"unknown `canonical_geometry` {self.canonical_geometry!r}; "
                "expected 'shared', 'independent' or 'none'")
        if self.allow_ungated_identity_pairs and not (
                self.canonical_geometry == "none" and self.p < 1.0):
            raise ValueError(
                "`allow_ungated_identity_pairs=True` is only meaningful with "
                "`canonical_geometry='none'` and p<1.0; remove the inert opt-in or fix "
                "the canonical-pair configuration.")
        if (self.canonical_geometry == "none" and self.p < 1.0
                and not self.allow_ungated_identity_pairs):
            # "none" makes the canonical clip the RAW input. That is only safe when the
            # augmented branch is always perturbed: `_forward_impl`'s gate returns the
            # untouched input when it does not fire, so at p<1 the ungated samples would
            # be a pair of BIT-IDENTICAL clips. The cross-view component then vanishes on
            # (1-p) of every batch and the objective reduces to the ordinary layer/sigma
            # alignment there. That dilution is allowed only through the explicit opt-in
            # above, so it cannot happen by typo.
            #
            # NOTE the gate this guards is `p`, the MASTER gate -- which under
            # `geometric_p` is the PHOTOMETRIC rate, not the geometric one. That is the
            # correct thing to require: `p=1.0` makes the photometric stage always run, so
            # the pair is never bit-identical, while `geometric_p < 1.0` still keeps
            # nominal-viewpoint clips in the training set. Do not relax this to look at
            # `geometric_p`.
            raise ValueError(
                f"`canonical_geometry='none'` requires p=1.0, got p={self.p}. Below 1.0 the "
                "ungated samples would return two identical clips. Set "
                "`allow_ungated_identity_pairs=True` only for a deliberately diluted "
                "canonical-teacher experiment.")
        if self.geometric_p is not None:
            if not 0.0 <= self.geometric_p <= 1.0:
                raise ValueError(
                    f"`geometric_p` must be in [0, 1], got {self.geometric_p}")
            if not (self.random_resized_crop or self.rotate):
                # Same L4 guard as `optical_corruption`/`randconv` above: a rate for a stage
                # that is not configured documents a treatment that never ran.
                raise ValueError(
                    "`geometric_p` is set but neither `random_resized_crop` nor `rotate` is "
                    "configured, so it would gate nothing. Remove it or configure a "
                    "geometric op.")
            if self.canonical_geometry == "independent":
                # `independent` draws the canonical branch's OWN displacement on every
                # sample (that is how it keeps the pair non-degenerate at p<1). Gating the
                # augmented branch's geometry separately would leave the two branches
                # displaced at different RATES as well as by different amounts, and the
                # arm would no longer be readable as either "shared geometry" or "two
                # independent viewpoints". Refused rather than documented.
                raise ValueError(
                    "`geometric_p` cannot be combined with `canonical_geometry="
                    "'independent'`: that mode displaces the canonical branch on every "
                    "sample by design, so a separate augmented-branch geometry rate makes "
                    "the pair's asymmetry uninterpretable. Use 'shared' or 'none'.")

        if (self.canonical_random_resized_crop or self.canonical_rotate) and \
                self.canonical_geometry != "independent":
            raise ValueError(
                "`canonical_random_resized_crop` / `canonical_rotate` only have meaning "
                f"under canonical_geometry='independent', got {self.canonical_geometry!r}; "
                "in 'shared' mode the canonical branch reuses the augmented draw by design.")
        for _name, _cfg in (("canonical_random_resized_crop", self.canonical_random_resized_crop),
                            ("canonical_rotate", self.canonical_rotate)):
            if _cfg and str(_cfg.get("ratio", "preserve")) != "preserve":
                raise ValueError(
                    f"`{_name}.ratio` must stay 'preserve' -- the other path consumes a "
                    "different number of RNG draws and would desynchronise the augmented "
                    "branch's stream from an un-overridden run.")
        if self.canonical_geometry == "independent" and not (
                self.random_resized_crop or self.rotate
                or self.canonical_random_resized_crop or self.canonical_rotate):
            raise ValueError(
                "`canonical_geometry='independent'` needs at least one geometric op "
                "(`random_resized_crop` or `rotate`) configured; without one the "
                "canonical branch would redraw nothing and the pair would collapse back "
                "to the 'shared' case while claiming otherwise.")
        unknown = sorted(set(self.augment_types) - {"corrupt_only", "color_only", "both"})
        if unknown:
            raise ValueError(f"unknown augment_types {unknown}")
        if self.optical_corruption:
            declared = list(self.optical_corruption.get("families", ()))
            if not declared:
                raise ValueError("`optical_corruption` needs a non-empty `families` list")
            bad = sorted(set(declared) - set(self.OPTICAL_FAMILIES))
            if bad:
                raise ValueError(f"unknown optical_corruption families {bad}; expected "
                                 f"a subset of {list(self.OPTICAL_FAMILIES)}")
            if "glass_blur" in declared:
                raise NotImplementedError(
                    "`glass_blur` is deliberately not implemented: it is a random, "
                    "unobservable per-pixel permutation, so the clean future ADF asks the "
                    "model to predict is not recoverable from the corrupted present "
                    "(R-OWN-007c measures -4.0pp there, the only negative family). It is "
                    "held out as a pre-registered falsifier -- see "
                    "`_apply_optical_corruption`.")

    @staticmethod
    def _rand(device: torch.device) -> torch.Tensor:
        return torch.rand((), device=device)

    @staticmethod
    def _uniform(device: torch.device, low: float, high: float) -> float:
        return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())

    def _sample_augment_type(self) -> str:
        if not self.augment_types:
            return "both"
        index = int(torch.randint(0, len(self.augment_types), (), device=torch.device("cpu")).item())
        return str(self.augment_types[index])

    def _apply_color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.color_jitter
        if not cfg:
            return frames

        ops = []
        brightness = float(cfg.get("brightness", 0.0))
        if brightness > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - brightness), 1.0 + brightness)
            ops.append(lambda img, factor=factor: F.adjust_brightness(img, factor))

        contrast = float(cfg.get("contrast", 0.0))
        if contrast > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - contrast), 1.0 + contrast)
            ops.append(lambda img, factor=factor: F.adjust_contrast(img, factor))

        saturation = float(cfg.get("saturation", 0.0))
        if saturation > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - saturation), 1.0 + saturation)
            ops.append(lambda img, factor=factor: F.adjust_saturation(img, factor))

        hue = float(cfg.get("hue", 0.0))
        if hue > 0:
            factor = self._uniform(frames.device, -min(hue, 0.5), min(hue, 0.5))
            ops.append(lambda img, factor=factor: F.adjust_hue(img, factor))

        if len(ops) > 1:
            order = torch.randperm(len(ops)).tolist()
            ops = [ops[i] for i in order]
        for op in ops:
            frames = op(frames)
        return frames

    def _apply_gamma(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.gamma
        if not cfg:
            return frames
        gamma_range = cfg.get("range", [0.9, 1.1])
        gamma = self._uniform(frames.device, float(gamma_range[0]), float(gamma_range[1]))
        return F.adjust_gamma(frames.clamp(0.0, 1.0), gamma=gamma)

    def _apply_exposure(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.exposure
        if not cfg:
            return frames
        ev_range = cfg.get("ev_range", cfg.get("range", [-0.1, 0.1]))
        ev = self._uniform(frames.device, float(ev_range[0]), float(ev_range[1]))
        return frames * (2.0 ** ev)

    def _sample_random_resized_crop(self, frames: torch.Tensor):
        cfg = self.random_resized_crop
        if not cfg:
            return None

        _, _, height, width = frames.shape
        scale = tuple(float(v) for v in cfg.get("scale", [0.95, 1.0]))
        ratio = cfg.get("ratio", "preserve")
        if ratio == "preserve":
            side_scale = self._uniform(frames.device, scale[0], scale[1])
            crop_h = max(1, min(height, int(round(height * side_scale))))
            crop_w = max(1, min(width, int(round(width * side_scale))))
            top = int(torch.randint(0, height - crop_h + 1, (), device=frames.device).item())
            left = int(torch.randint(0, width - crop_w + 1, (), device=frames.device).item())
            return (top, left, crop_h, crop_w, height, width)

        ratio = tuple(float(v) for v in ratio)
        top, left, crop_h, crop_w = TF.RandomResizedCrop.get_params(frames[0], scale=scale, ratio=ratio)
        return (top, left, crop_h, crop_w, height, width)

    @staticmethod
    def _apply_random_resized_crop_params(frames: torch.Tensor, params) -> torch.Tensor:
        if params is None:
            return frames
        top, left, crop_h, crop_w, height, width = params
        return F.resized_crop(
            frames,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[height, width],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _sample_rotate(self, frames: torch.Tensor):
        cfg = self.rotate
        if not cfg:
            return None
        degrees = cfg.get("degrees", 5.0)
        if isinstance(degrees, (list, tuple)):
            angle = self._uniform(frames.device, float(degrees[0]), float(degrees[1]))
        else:
            max_degrees = float(degrees)
            angle = self._uniform(frames.device, -max_degrees, max_degrees)
        return angle

    def _apply_rotate_angle(self, frames: torch.Tensor, angle) -> torch.Tensor:
        if angle is None:
            return frames
        fill = self.rotate.get("fill", "mean")
        if fill == "mean":
            # Computed per branch on purpose: the rotation *angle* is the geometric
            # part and must be shared, while the padding colour is an appearance
            # detail that belongs on whichever branch it is applied to.
            fill = float(frames.mean().item())
        return F.rotate(
            frames,
            angle=angle,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=fill,
        )

    def _apply_photometric(self, frames: torch.Tensor, aug_type: str) -> torch.Tensor:
        if aug_type in {"color_only", "both"}:
            frames = self._apply_color_jitter(frames)
            frames = self._apply_gamma(frames)
            frames = self._apply_exposure(frames)
        if aug_type in {"corrupt_only", "both"}:
            frames = self._apply_gaussian_noise(frames)
        # RandConv is gated by its OWN probability rather than by `augment_types`, so that
        # enabling it is a single-variable change against the existing recipe. Folding it
        # into `augment_types` instead would silently displace the types it replaced --
        # [color_only, texture_only, ...] drops `corrupt_only`/`both`, the only path that
        # fires `gaussian_noise`, so the arm would differ by two changes at once.
        #
        # It stays in the scene stage (before geometry) because it stands in for material
        # and texture changes, which are properties of the scene. `optical_corruption` is
        # the opposite -- see `_apply_sensor_stage`.
        if self.randconv and self._rand(frames.device).item() < float(
                self.randconv.get("p", 0.0) or 0.0):
            frames = self._apply_randconv(frames)
        return frames

    OPTICAL_FAMILIES = ("isotropic_blur", "directional_blur", "zoom_blur", "fog",
                        "glass_blur")

    def _apply_optical_corruption(self, frames: torch.Tensor) -> torch.Tensor:
        """One optical corruption from LIBERO-plus's `Sensor Noise` axis.

        That axis is not pixel noise: `LIBERO-plus/libero/libero/envs/env_wrapper.py`
        dispatches five ImageNet-C corruptions off the task name's `_noise_<N>` tag --
        motion / gaussian / zoom blur, fog, glass blur. Nothing in this class produced
        any of them, which is why `M1-E020` found the family's learned invariance did not
        transfer (r6 suppressed propagation of the *training* perturbation by 21.3% and of
        the benchmark's by 0.09%).

        Which families to enable is a config choice, and `glass_blur` is left out by
        default on a mechanistic argument with data behind it. ADF does not remove the
        corruption from the input -- frame 0 stays pinned to the corrupted observation at
        every step -- it makes the *predicted future* canonical so the action branch can
        read that instead. That requires the clean future to be recoverable from the
        corrupted present. Fog is an additive smooth veil and blur is a convolution, both
        ill-posed but learnable with scene priors; glass blur is a random, unobservable
        per-pixel permutation, which is not invertible even in principle. `R-OWN-007c`
        agrees: the mask (the same "read the predicted future instead of frame 0"
        mechanism) reads +19.7 / +18.2 / +12.8 / +8.5pp on gaussian / motion / fog / zoom
        and **-4.0pp on glass blur**, the only negative. Holding it out costs 50 tasks and
        buys a pre-registered falsifier: if a run trained on the other four improves glass
        blur too, this mechanistic story is wrong.

        Deliberately NOT the benchmark's implementations -- every parameter here is drawn
        continuously where theirs walks an integer 1..10 ladder, the directional kernel is
        a uniform line where theirs is ImageMagick's gaussian-decaying one, and the fog
        field is octave noise where theirs is a diamond-square plasma fractal. Parameters
        are sampled once per clip so the corruption is temporally consistent, matching
        every other op here.
        """
        cfg = self.optical_corruption
        if not cfg:
            return frames
        families = list(cfg.get("families", ("isotropic_blur", "directional_blur")))
        idx = int(torch.randint(0, len(families), (), device=torch.device("cpu")).item())
        family = families[idx]
        params = dict(cfg.get(family, {}) or {})

        if family == "isotropic_blur":
            lo, hi = params.get("sigma", (0.4, 5.0))
            sigma = self._uniform(frames.device, float(lo), float(hi))
            if sigma <= 1e-3:
                return frames
            ksize = int(2 * round(3.0 * sigma) + 1)
            return F.gaussian_blur(frames, kernel_size=[ksize, ksize], sigma=[sigma, sigma])

        if family == "directional_blur":
            lo, hi = params.get("length", (3, 21))
            length = int(round(self._uniform(frames.device, float(lo), float(hi))))
            if length < 2:
                return frames
            angle = self._uniform(frames.device, 0.0, 180.0)
            return self._apply_directional_blur(frames, length, angle)

        if family == "zoom_blur":
            lo, hi = params.get("max_zoom", (1.02, 1.20))
            steps = int(params.get("steps", 5))
            max_zoom = self._uniform(frames.device, float(lo), float(hi))
            return self._apply_zoom_blur(frames, max_zoom, steps)

        if family == "fog":
            lo, hi = params.get("strength", (0.3, 2.5))
            dlo, dhi = params.get("decay", (1.3, 3.0))
            strength = self._uniform(frames.device, float(lo), float(hi))
            decay = self._uniform(frames.device, float(dlo), float(dhi))
            return self._apply_fog(frames, strength, decay)

        raise ValueError(f"unknown optical corruption family {family!r}; expected one of "
                         f"{list(self.OPTICAL_FAMILIES)}")

    @staticmethod
    def _apply_zoom_blur(frames: torch.Tensor, max_zoom: float, steps: int) -> torch.Tensor:
        """Radial smear: the mean of several centre-zoomed copies.

        LIBERO-plus builds this from repeated `scipy.ndimage.zoom` calls over a fixed
        `arange(1, c[0], c[1])` ladder; this samples `max_zoom` continuously and averages
        `steps` bilinear rescales instead.
        """
        if max_zoom <= 1.0 or steps < 1:
            return frames
        h, w = frames.shape[-2:]
        acc = frames.clone()
        for i in range(1, steps + 1):
            factor = 1.0 + (max_zoom - 1.0) * i / steps
            zh, zw = max(int(round(h * factor)), h + 1), max(int(round(w * factor)), w + 1)
            scaled = torch_F.interpolate(frames, size=(zh, zw), mode="bilinear",
                                         align_corners=False)
            top, left = (zh - h) // 2, (zw - w) // 2
            acc = acc + scaled[..., top:top + h, left:left + w]
        return (acc / (steps + 1)).clamp(0.0, 1.0)

    @staticmethod
    def _apply_fog(frames: torch.Tensor, strength: float, decay: float) -> torch.Tensor:
        """Additive haze: a smooth random field veiling the scene, then renormalised.

        The combination `x = (x + s*field) * max/(max + s)` is the standard haze model and
        is kept because it is what makes the result a *veil* rather than a brightness
        shift. The field itself is octave noise (successively upsampled uniform noise)
        rather than LIBERO-plus's diamond-square plasma fractal.
        """
        if strength <= 0.0:
            return frames
        h, w = frames.shape[-2:]
        field = torch.zeros((1, 1, h, w), device=frames.device, dtype=frames.dtype)
        amplitude, size = 1.0, 2
        while size <= max(h, w) * 2:
            octave = torch.rand((1, 1, size, size), device=frames.device, dtype=frames.dtype)
            field = field + amplitude * torch_F.interpolate(
                octave, size=(h, w), mode="bilinear", align_corners=False)
            amplitude /= decay
            size *= 2
        field = field - field.min()
        peak = field.max()
        if peak <= 0:
            return frames
        field = field / peak
        max_val = frames.max()
        out = (frames + strength * field) * max_val / (max_val + strength)
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _apply_directional_blur(frames: torch.Tensor, length: int, angle: float) -> torch.Tensor:
        """Uniform-weight line kernel at `angle` degrees, applied per channel."""
        import math

        ksize = length if length % 2 == 1 else length + 1
        kernel = torch.zeros((ksize, ksize), device=frames.device, dtype=frames.dtype)
        centre = (ksize - 1) / 2.0
        rad = math.radians(angle)
        dx, dy = math.cos(rad), math.sin(rad)
        for step in range(ksize):
            offset = step - centre
            r = int(round(centre + offset * dy))
            c = int(round(centre + offset * dx))
            if 0 <= r < ksize and 0 <= c < ksize:
                kernel[r, c] = 1.0
        total = kernel.sum()
        if total <= 0:
            return frames
        kernel = kernel / total

        channels = frames.shape[1]
        weight = kernel.expand(channels, 1, ksize, ksize)
        pad = ksize // 2
        padded = torch_F.pad(frames, (pad, pad, pad, pad), mode="reflect")
        return torch_F.conv2d(padded, weight, groups=channels)

    def _apply_randconv(self, frames: torch.Tensor) -> torch.Tensor:
        """Random-convolution texture randomisation (Xu et al., RandConv).

        A random-weight convolution scrambles local texture and colour while leaving
        global shape intact, which is the one thing that targets LIBERO-plus's
        `Background Textures` axis: those are bddl texture swaps (`envs/textures.py`),
        a LOCAL semantic change that no global photometric op can imitate. That axis
        has the third-largest headroom in the ledger (121/185 for F-JA25, 64 tasks) and
        `R-PREL-005` shows geometric augmentation makes it *worse* (52.4 vs 58.9 with no
        augmentation at all), so it is currently served by nothing.

        Unlike the blur family this is NOT one of the benchmark's perturbations, so a
        gain here is a clean generalisation result rather than a number that has to be
        caveated as trained-on-the-test-distribution.

        The output is renormalised to the input's mean/std before mixing, otherwise a
        random kernel walks the clip straight out of [0, 1] and the caller's clamp
        would silently flatten it.
        """
        cfg = self.randconv
        if not cfg:
            return frames
        sizes = list(cfg.get("kernel_sizes", (1, 3, 5, 7)))
        ksize = int(sizes[int(torch.randint(0, len(sizes), (), device=torch.device("cpu")).item())])
        channels = frames.shape[1]
        weight = torch.randn((channels, channels, ksize, ksize),
                             device=frames.device, dtype=frames.dtype)
        weight = weight / (channels * ksize * ksize) ** 0.5
        pad = ksize // 2
        padded = torch_F.pad(frames, (pad, pad, pad, pad), mode="reflect")
        out = torch_F.conv2d(padded, weight)
        # Match the clip's statistics so the mix stays in range and the randomisation
        # changes texture rather than overall exposure.
        out = (out - out.mean()) / (out.std() + 1e-6) * frames.std() + frames.mean()
        lo, hi = cfg.get("mix", (0.0, 0.5))
        alpha = self._uniform(frames.device, float(lo), float(hi))
        return (alpha * frames + (1.0 - alpha) * out).clamp(0.0, 1.0)

    def _apply_sensor_stage(self, frames: torch.Tensor) -> torch.Tensor:
        """Corruptions that happen at the sensor, i.e. AFTER the camera geometry.

        LIBERO-plus corrupts the *rendered* image (`env_wrapper.step` post-processes
        `agentview_image`), while our geometric augmentation stands in for a camera pose
        change, which happens upstream of the sensor. Applying the corruption before the
        crop/rotation would blur an image that the camera had not taken yet, and would let
        the rotation's `fill=mean` border leak into a field that should have been laid
        down over the final frame.

        Ordering it here also keeps the random draws for `augment_types`, the photometric
        ops, the crop and the rotation in exactly the same sequence as an arm without this
        block, so the geometry sampled for a given RNG state is unchanged.
        """
        if self.optical_corruption and self._rand(frames.device).item() < float(
                self.optical_corruption.get("p", 0.0) or 0.0):
            frames = self._apply_optical_corruption(frames)
        return frames

    def _apply_geometric(self, frames: torch.Tensor, crop_params, angle) -> torch.Tensor:
        frames = self._apply_random_resized_crop_params(frames, crop_params)
        return self._apply_rotate_angle(frames, angle)

    def _apply_gaussian_noise(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.gaussian_noise
        if not cfg:
            return frames
        std = float(cfg.get("std", 0.01))
        noise = torch.randn((1,) + tuple(frames.shape[1:]), device=frames.device, dtype=frames.dtype) * std
        return frames + noise

    @staticmethod
    def _validate_input_range(frames: torch.Tensor) -> None:
        min_value = float(frames.min().item())
        max_value = float(frames.max().item())
        eps = 1e-6
        if min_value < -eps or max_value > 1.0 + eps:
            raise ValueError(
                "`VideoAugmentation` expects input frames in [0, 1] before augmentation, "
                f"got min={min_value:.6f}, max={max_value:.6f}."
            )

    def _sample_canonical_geometry(self, frames: torch.Tensor):
        """Draw the canonical branch's own crop/rotation (`canonical_geometry='independent'`).

        Deliberately drawn from the same distributions as the augmented branch and
        deliberately drawn LAST, after the augmented clip is fully constructed. That
        ordering is the point: every draw the augmented branch makes happens in exactly
        the same sequence as in `"shared"` mode, so switching this flag changes the
        canonical branch's distribution and nothing else about the augmented one.
        (The extra draws do advance the global RNG, so downstream samples are not
        bit-identical to a `"shared"` run -- the augmented *distribution* is preserved,
        not the realization.)
        """
        # `canonical_random_resized_crop` / `canonical_rotate` override the SAMPLING
        # distribution only (scale / degrees). `fill` and the interpolation mode are read
        # later by `_apply_rotate_angle`, outside this swap, and stay shared on purpose --
        # those are appearance details, not part of the viewpoint distribution.
        # With no override the swap is a no-op assignment of the same objects, so the draw
        # sequence is byte-identical to a run predating this parameter.
        prev_crop, prev_rot = self.random_resized_crop, self.rotate
        if self.canonical_random_resized_crop:
            self.random_resized_crop = self.canonical_random_resized_crop
        if self.canonical_rotate:
            self.rotate = self.canonical_rotate
        try:
            return self._sample_random_resized_crop(frames), self._sample_rotate(frames)
        finally:
            self.random_resized_crop, self.rotate = prev_crop, prev_rot

    def _canonical_geometry_only(self, frames: torch.Tensor) -> torch.Tensor:
        """Canonical clip for a sample whose `p` gate did not fire.

        Only reachable under `canonical_geometry="independent"`. The augmented clip is
        the untouched input in that case, so the pair's entire difference is the
        canonical branch's viewpoint draw. Keeping the pair non-degenerate here matters:
        with `p=0.25`, letting the ungated 75% return two identical clips would make the
        cross-view term trivially satisfiable on three quarters of every batch.
        """
        crop_params, angle = self._sample_canonical_geometry(frames)
        return self._apply_geometric(frames, crop_params, angle).clamp(0.0, 1.0)

    def _augment_frames_impl(self, frames: torch.Tensor, split: bool):
        """Shared body for the single-output and paired augmentation paths.

        The geometric stage is the expensive part, so it runs on the canonical
        branch only when that branch is actually requested.
        """
        self._validate_input_range(frames)
        aug_type = self._sample_augment_type()
        photometric = self._apply_photometric(frames, aug_type)
        # The geometric gate. `geometric_p is None` takes NO draw and samples the crop and
        # angle exactly as before, so the RNG stream is untouched for every pre-existing arm.
        # When it is set, skipping the stage also skips its two draws -- a fired sample and
        # an unfired one consume different amounts of randomness, which is fine within an arm
        # and is why this is not bit-comparable against a run with a different rate.
        geometry_fires = True
        if self.geometric_p is not None:
            geometry_fires = self._rand(frames.device).item() < self.geometric_p
        crop_params = self._sample_random_resized_crop(frames) if geometry_fires else None
        angle = self._sample_rotate(frames) if geometry_fires else None
        augmented = self._apply_geometric(photometric, crop_params, angle).clamp(0.0, 1.0)
        augmented = self._apply_sensor_stage(augmented).clamp(0.0, 1.0)
        if not split:
            return augmented
        if self.canonical_geometry == "none":
            # The canonical clip IS the nominal view -- no geometry, no photometry. The
            # objective becomes "map a perturbed view onto a fixed reference" instead of
            # "align two random displacements", and that fixed reference is exactly what
            # inference sees. Requires p=1.0, asserted in __init__.
            return augmented, frames.clone()
        if self.canonical_geometry == "independent":
            canon_crop, canon_angle = self._sample_canonical_geometry(frames)
        else:
            # "shared": the canonical branch reuses the augmented branch's displacement, so
            # the pair's entire difference is photometric. Under `geometric_p` an unfired
            # sample carries `(None, None)` here, which `_apply_geometric` passes through --
            # both branches then sit at the nominal viewpoint and the pair is still purely
            # photometric. That is the intended behaviour, not a special case.
            canon_crop, canon_angle = crop_params, angle
        canonical = self._apply_geometric(frames, canon_crop, canon_angle).clamp(0.0, 1.0)
        return augmented, canonical

    def _augment_frames_split(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(augmented, canonical)``; ``canonical_geometry`` sets how they relate.

        ``"shared"`` (default): ``canonical = Geo(I)`` and ``augmented = Geo(Photo(I))``
        run the geometric stage with the *same* sampled crop and rotation, so their
        difference is purely photometric. That is what lets `appearance_path` /
        `lambda_deaug` treat the residual as an appearance offset while leaving camera
        geometry alone -- those terms regress a latent difference, and a viewpoint change
        is not expressible as one.

        ``"independent"``: ``canonical = Geo'(I)`` with its own crop and rotation, so the
        pair differs by viewpoint as well. Nothing that reads the pair as a residual may
        run in this mode (`runtime._assert_canonical_clip_contract` refuses); it exists
        for the cross-view invariance term, which only needs the two clips to be
        action-equivalent, not latent-subtractable.

        Random draws for the *augmented* branch happen in the same order in both modes
        (augment type, photometric ops, crop, rotation, sensor stage), so ``_augment_frames``
        reproduces the previous behaviour bit for bit and the augmented branch's
        distribution is identical across modes.
        """
        return self._augment_frames_impl(frames, split=True)

    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._augment_frames_impl(frames, split=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(frames, split=False)

    def forward_split(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Like :meth:`forward` but also returns the geometry-matched canonical clip.

        When the ``p`` gate does not fire, both returned clips are the untouched
        input, so the appearance residual is exactly zero for that sample.
        """
        return self._forward_impl(frames, split=True)

    def _forward_impl(self, frames: torch.Tensor, split: bool):
        if frames.ndim == 5:
            if not torch.is_floating_point(frames):
                raise TypeError(f"`VideoAugmentation` expects floating frames, got {frames.dtype}")
            if self._rand(torch.device("cpu")).item() >= self.p:
                if not split:
                    return frames
                if self.canonical_geometry != "independent":
                    return frames, frames.clone()
                canonical = frames.clone()
                for cam_idx in range(frames.shape[0]):
                    canonical[cam_idx] = self._canonical_geometry_only(frames[cam_idx])
                return frames, canonical
            augmented = frames.clone()
            canonical = frames.clone() if split else None
            for cam_idx in range(frames.shape[0]):
                if split:
                    aug_cam, canon_cam = self._augment_frames_split(frames[cam_idx])
                    augmented[cam_idx] = aug_cam
                    canonical[cam_idx] = canon_cam
                else:
                    augmented[cam_idx] = self._augment_frames(frames[cam_idx])
            return (augmented, canonical) if split else augmented

        squeeze_time = False
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
            squeeze_time = True
        if frames.ndim != 4:
            raise ValueError(f"`VideoAugmentation` expects [T,C,H,W], got {tuple(frames.shape)}")
        if not torch.is_floating_point(frames):
            raise TypeError(f"`VideoAugmentation` expects floating frames, got {frames.dtype}")

        def _maybe_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0) if squeeze_time else x

        if self._rand(torch.device("cpu")).item() >= self.p:
            if split:
                canonical = (
                    self._canonical_geometry_only(frames)
                    if self.canonical_geometry == "independent"
                    else frames.clone()
                )
                return _maybe_squeeze(frames), _maybe_squeeze(canonical)
            return _maybe_squeeze(frames)

        if split:
            augmented, canonical = self._augment_frames_split(frames)
            return _maybe_squeeze(augmented), _maybe_squeeze(canonical)
        return _maybe_squeeze(self._augment_frames(frames))
