"""Training loop.

Step-based throughout: there is no epoch, because the mixture sampler draws from
a weighted distribution over 26 profiles indefinitely. `max_steps` is the only
notion of training length.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mmabc.data import Collator, MixtureDataset, to_device
from mmabc.models.attention import set_sdpa_debug
from mmabc.paths import home_path
from mmabc.models.mmabc import MMABCPolicy, load_model_config
from mmabc.train.checkpoint import CheckpointConfig, CheckpointManager
from mmabc.train.fsdp import (
    DistInfo,
    all_reduce_mean,
    all_reduce_min,
    barrier,
    build_mesh,
    clip_grad_norm,
    grad_diagnostics,
    grad_group_norms,
    init_distributed,
    shard_model,
)
from mmabc.utils.logging import MetricLogger, setup_logging


@dataclass
class Timings:
    """Wall-clock split of a step."""

    data: float = 0.0
    compute: float = 0.0
    sync: float = 0.0

    def reset(self) -> None:
        self.data = 0.0
        self.compute = 0.0
        self.sync = 0.0

    @property
    def total(self) -> float:
        return self.data + self.compute + self.sync


def _warm_page_cache(roots: list[Path], log) -> None:
    t0, total = time.time(), 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                with open(path, "rb") as fh:
                    while chunk := fh.read(16 << 20):
                        total += len(chunk)
            except OSError:
                continue
    log.info("page cache warmed: %.1f GB in %.0fs", total / 2**30, time.time() - t0)


def cosine_with_warmup(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, cfg: DictConfig, repo_root: Path) -> None:
        self.cfg = cfg
        self.repo_root = repo_root
        self.info: DistInfo = init_distributed()
        self.log = setup_logging(rank=self.info.rank)
        if bool(cfg.get("allow_tf32", True)):
            # The vision tower runs in fp32 (fp32_visual). TF32 keeps fp32's
            # exponent range, which is what protects that backward from
            # overflowing, while putting its matmuls on tensor cores.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_cfg = load_model_config(str(cfg.model), repo_root=str(repo_root))
        # `model` is a path, so it cannot be overridden field-by-field from the
        # command line. `model_overrides` is a nested block merged on top, which
        # is what makes sweeps possible without copying the whole model config.
        overrides = cfg.get("model_overrides")
        if overrides:
            model_cfg = OmegaConf.merge(model_cfg, overrides)
            if self.info.is_master:
                self.log.info(
                    "model overrides: %s", OmegaConf.to_container(overrides, resolve=True)
                )
        self.model_cfg = model_cfg

        # Initialise identical weights across ranks before HSDP shards and replicates them.
        torch.manual_seed(int(cfg.seed))
        self.model = MMABCPolicy(model_cfg, repo_root=str(repo_root))
        self.model.to(self.info.device)
        # Now decorrelate the per-rank RNG so dropout masks and any sampling
        # differ across ranks; this must come after construction, not before.
        torch.manual_seed(int(cfg.seed) + self.info.rank)

        # Freeze parameters before sharding to exclude them from gradient collectives.
        for target in cfg.get("freeze", []) or []:
            try:
                module = self.model.get_submodule(str(target))
            except AttributeError as exc:
                raise ValueError(f"cannot freeze unknown module {target!r}") from exc
            n = 0
            for p in module.parameters():
                p.requires_grad_(False)
                n += 1
            if self.info.is_master:
                self.log.info("froze %s (%d tensors)", target, n)

        self.mesh = build_mesh(
            self.info,
            strategy=str(cfg.get("shard_strategy", "hsdp")),
            shard_size=cfg.get("shard_size"),
        )
        self.model = shard_model(
            self.model,
            self.mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            reshard_after_forward=bool(cfg.get("reshard_after_forward", True)),
            visual_param_dtype=(
                torch.float32 if bool(cfg.get("fp32_visual", True)) else None
            ),
        )
        if self.mesh is not None and self.info.is_master:
            self.log.info(
                "sharding: %s over mesh %s", cfg.get("shard_strategy", "hsdp"), tuple(self.mesh.shape)
            )

        if bool(cfg.get("compile_expert", False)):
            # Only the expert. The backbone sees variable image-token counts as
            # views come and go, which would trigger repeated recompilation.
            self.model.expert = torch.compile(self.model.expert, dynamic=True)

        self.optimizer = torch.optim.AdamW(
            self.model.param_groups(cfg.optim),
            betas=tuple(float(b) for b in cfg.optim.betas),
            eps=float(cfg.optim.eps),
        )
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        self.ckpt_root = home_path(cfg.checkpoint.root)
        self.dumped_bad_batch = False
        self.ckpt = CheckpointManager(
            CheckpointConfig(
                root=str(self.ckpt_root),
                keep_resume=int(cfg.checkpoint.keep_resume),
                milestone_every=int(cfg.checkpoint.milestone_every),
                resume_every=int(cfg.checkpoint.resume_every),
                min_free_gb=float(cfg.checkpoint.min_free_gb),
            ),
            is_master=self.info.is_master,
        )
        self.ckpt.check_space()

        self.loader = self._build_loader()
        if self.info.is_master:
            self._write_run_meta()
        self.metrics = MetricLogger(
            log_every=int(cfg.log_every),
            wandb_project=cfg.get("wandb_project"),
            wandb_entity=cfg.get("wandb_entity"),
            run_name=str(cfg.get("run_name", "mmabc")),
            enabled=self.info.is_master,
            config=OmegaConf.to_container(cfg, resolve=True),
            id_file=self.ckpt_root / "wandb_run_id.txt",
        )
        self.step = 0
        opt = cfg.optim
        self.skip_factor = float(opt.get("grad_norm_spike_factor", 8.0))
        self.skip_min_history = int(opt.get("grad_norm_min_history", 25))
        self.grad_norm_ceiling = float(opt.get("grad_norm_ceiling", 1.0e4))
        # A norm the clipper accepts verbatim is never an outlier; without this
        # floor a warm-started run whose healthy norm rises during LR warmup
        # rejects every step (it aborted FT50's first launch at step 157).
        self.grad_norm_floor = float(opt.get("max_grad_norm", 1.0))
        self.max_skip_frac = float(opt.get("max_skip_frac", 0.25))
        self.grad_norm_history: deque[float] = deque(
            maxlen=int(opt.get("grad_norm_window", 200))
        )
        self.skip_window: deque[int] = deque(maxlen=int(opt.get("skip_rate_window", 200)))
        self.skips: dict[str, int] = {}

    def _resolve(self, rel) -> Path:
        p = Path(str(rel))
        return p if p.is_absolute() else self.repo_root / p

    def _write_run_meta(self) -> None:
        """Everything deployment needs, frozen next to the checkpoints.

        Norm stats, the layout and the model config are copied rather than
        referenced: re-running the converter rewrites configs/norm_stats, and a
        checkpoint must keep denormalising with the statistics it trained on.
        """
        import json
        import shutil

        meta = self.ckpt_root / "run_meta"
        meta.mkdir(parents=True, exist_ok=True)
        canonical = self._resolve(self.model_cfg.canonical)
        shutil.copy(canonical, meta / "canonical.yaml")
        model_cfg = OmegaConf.create(OmegaConf.to_container(self.model_cfg, resolve=True))
        model_cfg.canonical = str(meta / "canonical.yaml")
        OmegaConf.save(model_cfg, meta / "model_config.yaml")
        OmegaConf.save(self.cfg, meta / "train_config.yaml")

        profiles = {}
        for pc in self.dataset.profile_configs:
            tag = pc.embodiment_tag
            stats = self._resolve(self.cfg.data.norm_stats_dir) / f"{tag}.json"
            if stats.exists():
                shutil.copy(stats, meta / f"norm_stats_{tag}.json")
            root = Path(pc.path) / "meta"
            shutil.copy(root / "embodiment.json", meta / f"embodiment_{tag}.json")
            info = json.loads((root / "info.json").read_text())
            if "layout" in info:
                (meta / f"contract_{tag}.json").write_text(json.dumps(info["layout"], indent=1))
            profiles[pc.name] = {"embodiment_tag": tag, "path": pc.path, "views": pc.views,
                                 "fps": pc.fps, "reference_frame": pc.reference_frame}
        (meta / "profiles.json").write_text(json.dumps(profiles, indent=1))
        self.log.info("wrote run metadata to %s", meta)

    def _build_loader(self) -> DataLoader:
        cfg = self.cfg
        dataset = MixtureDataset(
            self._resolve(cfg.data.mixture),
            chunk_size=int(self.model_cfg.flow.chunk_size),
            image_size=int(self.model_cfg.obs.image_size),
            future_offsets=self.model.future_offsets
            if isinstance(self.model, MMABCPolicy)
            else int(self.model_cfg.flow.chunk_size),
            prompt_dropout=float(cfg.data.prompt_dropout),
            prompt_header=bool(self.model_cfg.get("prompt", {}).get("header", False)),
            norm_stats_dir=self._resolve(cfg.data.norm_stats_dir),
            seed=int(cfg.seed) + self.info.rank * 10_007,
            repo_root=str(self.repo_root),
        )
        if self.info.is_master:
            self.log.info("\n%s", dataset.describe())
        if self.info.local_rank == 0 and bool(cfg.data.get("warm_page_cache", True)):
            # One reader per node pulls every profile file through the page
            # cache once; afterwards all of the node's workers read from RAM.
            import threading

            roots = [Path(pc.path) for pc in dataset.profile_configs]
            threading.Thread(target=_warm_page_cache, args=(roots, self.log), daemon=True).start()
        workers = int(cfg.data.num_workers)
        # DataLoader timeout bounds stalls that the per-decode watchdog cannot handle.
        loader_timeout = float(cfg.data.get("loader_timeout", 600.0)) if workers > 0 else 0.0
        in_workers = workers > 0 and bool(cfg.data.get("processor_in_workers", True))
        collate_fn = Collator(str(home_path(self.model_cfg.backbone.path)) if in_workers else None)
        self.dataset = dataset
        return DataLoader(
            dataset,
            batch_size=int(cfg.micro_batch_size),
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=int(cfg.data.prefetch_factor) if workers > 0 else None,
            timeout=loader_timeout,
        )

    def _autocast(self):
        """bf16 compute.

        With FSDP2 the mixed-precision policy already casts parameters, so this
        is a no-op there. Without a mesh (single GPU) there is no policy, so
        autocast is what keeps fp32 master weights from running fp32 matmuls.
        """
        if self.mesh is not None:
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _set_lr(self, step: int) -> float:
        scale = cosine_with_warmup(
            step,
            warmup=int(self.cfg.optim.warmup_steps),
            total=int(self.cfg.max_steps),
            min_ratio=float(self.cfg.optim.min_lr_ratio),
        )
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale
        return scale

    def _dump_bad_batch(self, batch: dict, micro: int) -> None:
        """Write the inputs behind a non-finite loss, once per rank.

        Which rank produces the NaN is decided by its data stream, and replaying
        that stream offline against the initial weights does not reproduce it,
        so the batch has to be captured where it actually fails. One file per
        rank is enough to identify the sample, and keeps a persistent failure
        from filling the disk.
        """
        if self.dumped_bad_batch:
            return
        self.dumped_bad_batch = True
        out = self.ckpt_root / "bad_batches"
        try:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"rank{self.info.rank}_step{self.step}_micro{micro}.pt"
            payload = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            payload["_step"] = self.step
            payload["_rank"] = self.info.rank
            torch.save(payload, path)
            # Say which tensors are already corrupt on arrival: that separates a
            # bad sample on disk from a model that overflowed on a valid one.
            offenders = {
                k: f"{int((~torch.isfinite(v)).sum())}/{v.numel()}"
                for k, v in batch.items()
                if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
            }
            self.log.warning(
                "dumped the offending batch to %s; non-finite inputs: %s",
                path,
                offenders or "none, the inputs are clean",
            )
        except Exception as exc:  # diagnostics must never end a run
            self.log.warning("could not dump the offending batch: %r", exc)

    def _diagnose_nonfinite_forward(self, batch: dict, *, mine: bool) -> None:
        """Separate a corrupt parameter from a corrupt computation."""
        try:
            bad = []
            biggest, where = 0.0, "-"
            n_params = 0
            for name, p in self.model.named_parameters():
                n_params += 1
                local = p.to_local() if hasattr(p, "to_local") else p
                if local.numel() == 0:
                    continue
                if not torch.isfinite(local).all():
                    bad.append(name)
                    continue
                # "Finite" is too weak a question on its own: a weight of 1e30
                # passes it and still overflows the next forward, which looks
                # identical to a NaN weight from the loss's point of view.
                peak = float(local.abs().max())
                if peak > biggest:
                    biggest, where = peak, name
            with torch.no_grad(), self._autocast():
                again = float(self.model(batch).loss)
            birth = self._first_nonfinite_module(batch)
            if mine:
                self.log.warning("rank %d probe%s", self.info.rank, birth)
                self.log.warning(
                    "rank %d non-finite forward: %d/%d parameter shards non-finite%s; "
                    "largest finite weight %.4g at %s; "
                    "re-running the same batch gives %s",
                    self.info.rank,
                    len(bad),
                    n_params,
                    f" (first: {bad[:5]})" if bad else "",
                    biggest,
                    where,
                    again,
                )
        except Exception as exc:
            self.log.warning("could not diagnose the non-finite forward: %r", exc)

    def _watch_report(self) -> str:
        """Trace named parameters step by step, weight and gradient together."""
        wanted = list(self.cfg.get("debug_watch", []) or [])
        if not wanted:
            return ""
        bits = []
        for name, p in self.model.named_parameters():
            if not any(w in name for w in wanted):
                continue
            local = p.to_local() if hasattr(p, "to_local") else p
            if local.numel() == 0:
                continue
            w = float(local.abs().max())
            g = "-"
            if p.grad is not None:
                gl = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                if gl.numel():
                    g = f"{float(gl.abs().max()):.4g}"
            bits.append(f"{name.split('.')[-2]}.{name.split('.')[-1]}: |w|={w:.4g} |g|={g}")
        return ("; watch: " + ", ".join(bits)) if bits else ""

    def _first_nonfinite_module(self, batch: dict) -> str:
        """Name the module that *creates* the non-finite value."""
        found: list[tuple[int, str, bool, float, float]] = []
        order = [0]

        def clean(x):
            if torch.is_tensor(x) and x.is_floating_point():
                return bool(torch.isfinite(x).all())
            if isinstance(x, (tuple, list)):
                return all(clean(i) for i in x)
            return True

        def peak(x):
            """Largest finite magnitude anywhere in a nest of tensors.

            Finiteness alone is not the right question for the *input* side.
            bf16 saturates just above 3e38, so an input of 1e20 is perfectly
            finite and still overflows the moment a matmul sums four thousand
            of them. Without the magnitude, such a module looks like it
            invented the NaN when it merely finished an overflow that started
            further upstream.
            """
            if torch.is_tensor(x) and x.is_floating_point():
                f = x[torch.isfinite(x)]
                return float(f.abs().max()) if f.numel() else 0.0
            if isinstance(x, (tuple, list)):
                vals = [peak(i) for i in x]
                return max(vals) if vals else 0.0
            return 0.0

        def check(name):
            def hook(module, args, output):
                idx = order[0]
                order[0] += 1
                if not clean(output):
                    # Inspect the materialised forward weight when FSDP has replaced its parameter view.
                    w = getattr(module, "weight", None)
                    wmax = float(w.detach().abs().max()) if torch.is_tensor(w) and w.numel() else -1.0
                    found.append((idx, name, clean(args), peak(args), peak(output), wmax))

            return hook

        handles = [m.register_forward_hook(check(n)) for n, m in self.model.named_modules()]
        try:
            with torch.no_grad(), self._autocast():
                self.model(batch)
        except Exception as exc:
            return f"; hook probe failed: {exc!r}"
        finally:
            for h in handles:
                h.remove()

        if not found:
            return "; hook probe saw no non-finite module output"
        found.sort()
        births = [
            f"{n} (idx {i}, max|in|={pin:.4g}, max finite |out|={pout:.4g}, "
            f"max|w as used|={wmax:.4g})"
            for i, n, inputs_clean, pin, pout, wmax in found
            if inputs_clean
        ]
        return (
            f"; {len(found)} modules emitted non-finite output, first={found[0][1]}"
            f"; born in: {births[:3] if births else 'nowhere -- inputs were already bad'}"
        )

    def _grad_group_report(self) -> str:
        """Attribute a rejected gradient to a part of the model.

        Printed on the two paths that throw a step away, because those are the
        only moments when the gradient that caused the problem still exists.
        Both paths are rare by construction, so the extra collective costs
        nothing measurable.
        """
        norms = getattr(self, "pending_grad_groups", None)
        if not norms:
            return ""
        ranked = sorted(norms.items(), key=lambda kv: -kv[1][0])
        return "; by group: " + ", ".join(
            f"{k}={norm:.4g}" + (f"(bad={n_bad})" if n_bad else "")
            for k, (norm, n_bad) in ranked
        )

    def _grad_norm_threshold(self) -> float:
        """Above what gradient norm this step should be thrown away."""
        history = self.grad_norm_history
        if len(history) < self.skip_min_history:
            # Apply the absolute norm ceiling until enough finite steps establish a rolling baseline.
            return self.grad_norm_ceiling
        tracking = max(self.skip_factor * float(median(history)), self.grad_norm_floor)
        return min(tracking, self.grad_norm_ceiling)

    def _record_skip(self, reason: str) -> None:
        """Count a dropped step and fail the run if dropping becomes the norm.

        Isolated skips are the mechanism working as intended. A sustained rate
        means the data or the configuration is wrong, and silently training on a
        fraction of the batches for days is worse than stopping.
        """
        self.skips[reason] = self.skips.get(reason, 0) + 1
        self.skip_window.append(1)
        # Judge on a partial window too, capped so a couple of early skips cannot
        # end the run: waiting for a full window lets a completely broken run
        # burn through hundreds of steps before anyone is told.
        if len(self.skip_window) >= min(self.skip_window.maxlen, 40):
            frac = sum(self.skip_window) / len(self.skip_window)
            if frac > self.max_skip_frac:
                raise RuntimeError(
                    f"aborting: {frac:.0%} of the last {len(self.skip_window)} steps were "
                    f"skipped (limit {self.max_skip_frac:.0%}); counts={self.skips}"
                )

    def _nonfinite_report(self, batch: dict, accumulated: dict) -> str:
        """Name the local cause of a non-finite gradient, if there is one.

        The clipped norm is global, so every rank sees the failure no matter
        which rank produced it. Reporting only what is locally non-finite is
        what separates "this rank had a bad batch" from "this rank was told
        about someone else's".
        """
        if not bool(self.cfg.get("debug_nonfinite", True)):
            return ""
        bad_params = []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            g = g.to_local() if hasattr(g, "to_local") else g
            if not torch.isfinite(g).all():
                bad_params.append(name)
                if len(bad_params) >= 3:
                    break
        if not bad_params:
            return " (grads finite here; another rank produced it)"
        loss_val = accumulated.get("loss", float("nan"))
        profiles = sorted(set(batch.get("profile", []) or []))
        return (
            f" local grads non-finite in {bad_params}; loss={loss_val:.4g}; "
            f"profiles={profiles}"
        )

    def train(self) -> None:
        cfg = self.cfg
        accum = int(cfg.grad_accum_steps)
        max_steps = int(cfg.max_steps)

        resume = self.ckpt.latest_resume() if bool(cfg.get("auto_resume", True)) else None
        if resume is not None:
            self.step = self.ckpt.load_resume(resume, self.model, self.optimizer, None)
            self.log.info("resumed from %s at step %d", resume, self.step)
        elif cfg.get("init_from"):
            # Warm-start model weights, then allow a resume checkpoint to restore full training state.
            self.ckpt.load_pretrained(str(home_path(cfg.init_from)), self.model, log=self.log)

        it = iter(self.loader)
        timings = Timings()
        self.model.train()
        window_start = time.time()

        # Names the operation that produced a NaN in backward, which is the only
        # practical way to find one that leaves the forward pass finite. Several
        # times slower, so it is opt-in and meant for short diagnostic runs.
        if bool(cfg.get("debug_anomaly", False)):
            torch.autograd.set_detect_anomaly(True)
            self.log.warning("autograd anomaly detection is on; expect a slow run")
        if bool(cfg.get("debug_sdpa", False)):
            set_sdpa_debug(True)
            self.log.warning("sdpa NaN reporting is on")

        while self.step < max_steps:
            scale = self._set_lr(self.step)
            accumulated: dict[str, float] = {}
            did_backward = False

            for micro in range(accum):
                t0 = time.time()
                batch = next(it)
                batch = to_device(batch, self.info.device)
                timings.data += time.time() - t0

                t0 = time.time()
                with self._autocast():
                    out = self.model(batch)
                # A NaN loss on even one rank must skip backward on every rank,
                # otherwise FSDP's gradient all-reduce desynchronises.
                loss_ok = all_reduce_min(
                    1.0 if torch.isfinite(out.loss) else 0.0, self.info.device
                )
                if loss_ok < 1.0:
                    # The vote is global, so say whether *this* rank is the one
                    # that voted no; otherwise every rank looks equally guilty.
                    local = float(out.loss)
                    mine = not math.isfinite(local)
                    # Which term went first decides where to look, and the terms
                    # are gone by the time the step-level handler runs.
                    terms = (
                        " ".join(f"{k}={v:.6g}" for k, v in sorted(out.metrics.items()))
                        if mine
                        else ""
                    )
                    self.log.warning(
                        "step %d micro %d: skipped, local loss=%.6g (%s) %s",
                        self.step,
                        micro,
                        local,
                        "this rank is non-finite" if mine else "another rank is",
                        terms,
                    )
                    if mine:
                        self._dump_bad_batch(batch, micro)
                    if bool(self.cfg.get("debug_nonfinite", False)):
                        self._diagnose_nonfinite_forward(batch, mine=mine)
                    torch.cuda.synchronize()
                    timings.compute += time.time() - t0
                    continue
                (out.loss / accum).backward()
                did_backward = True
                torch.cuda.synchronize()
                timings.compute += time.time() - t0

                for k, v in out.metrics.items():
                    accumulated[k] = accumulated.get(k, 0.0) + v / accum

            t0 = time.time()
            if not did_backward:
                self.optimizer.zero_grad(set_to_none=True)
                self.log.warning(
                    "step %d: non-finite loss on every micro-batch, update skipped",
                    self.step,
                )
                self._record_skip("nonfinite_loss")
                self.step += 1
                continue
            params = list(self.model.parameters())
            if bool(cfg.get("debug_grad_norm", False)):
                d = grad_diagnostics(self.model, self.mesh)
                self.log.info(
                    "step %d: finite grad norm %.6g (max |g| %.6g) bad=%d "
                    "(inf=%d nan=%d) %s",
                    self.step,
                    d["norm"],
                    d["max"],
                    d["n_bad"],
                    d["n_inf"],
                    d["n_nan"],
                    d["worst"][:6],
                )
            # Sampled before clipping, which is destructive: a non-finite norm
            # gives a scale of max_norm/inf = 0, so every gradient is zeroed and
            # a report taken afterwards shows nothing anywhere.
            self.pending_grad_groups = (
                grad_group_norms(self.model, self.mesh)
                if bool(cfg.get("debug_grad_groups", False))
                else None
            )
            watched = self._watch_report()
            if watched:
                self.log.info("step %d pre-step%s", self.step, watched)
            grad_norm = clip_grad_norm(params, float(cfg.optim.max_grad_norm))
            threshold = self._grad_norm_threshold()

            # Skip non-finite updates on every rank before they can corrupt model or optimiser state.
            if not torch.isfinite(grad_norm):
                self.log.warning(
                    "step %d: non-finite gradient norm, update skipped%s%s",
                    self.step,
                    self._nonfinite_report(batch, accumulated),
                    self._grad_group_report(),
                )
                self.optimizer.zero_grad(set_to_none=True)
                self._record_skip("nonfinite_grad")
                self.step += 1
                continue

            if float(grad_norm) > threshold:
                self.log.warning(
                    "step %d: gradient norm %.4g over the %.4g spike threshold "
                    "(%.1fx the median of the last %d steps), update skipped%s",
                    self.step,
                    float(grad_norm),
                    threshold,
                    self.skip_factor,
                    len(self.grad_norm_history),
                    self._grad_group_report(),
                )
                self.optimizer.zero_grad(set_to_none=True)
                self._record_skip("spike")
                self.step += 1
                continue

            self.grad_norm_history.append(float(grad_norm))
            self.skip_window.append(0)
            self.optimizer.step()
            torch.cuda.synchronize()
            timings.sync += time.time() - t0
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if self.step % int(cfg.log_every) == 0:
                elapsed = time.time() - window_start
                n = int(cfg.log_every)
                samples = n * accum * int(cfg.micro_batch_size)
                stats = dict(accumulated)
                stats["grad_norm"] = float(grad_norm)
                stats["grad_norm_limit"] = threshold
                stats["skip_frac"] = (
                    sum(self.skip_window) / len(self.skip_window) if self.skip_window else 0.0
                )
                stats["lr_scale"] = scale
                stats["samples_per_s"] = samples * self.info.world_size / max(elapsed, 1e-6)
                stats["s_per_step"] = elapsed / n
                stats["t_data_ms"] = timings.data / n * 1000.0
                stats["t_compute_ms"] = timings.compute / n * 1000.0
                stats["t_sync_ms"] = timings.sync / n * 1000.0
                stats["data_frac"] = timings.data / max(timings.total, 1e-6)
                stats["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                stats["loss"] = all_reduce_mean(stats.get("loss", 0.0), self.info.device)
                self.metrics.log(self.step, stats)
                timings.reset()
                window_start = time.time()

            if self.step % int(cfg.checkpoint.resume_every) == 0:
                barrier()
                self.ckpt.save_resume(self.step, self.model, self.optimizer, None)
                self.log.info("saved resume checkpoint at step %d", self.step)
            if self.step % int(cfg.checkpoint.milestone_every) == 0:
                barrier()
                self.ckpt.save_milestone(self.step, self.model)
                self.log.info("saved milestone at step %d", self.step)

        barrier()
        # The loop already wrote this step if it fell on an interval.
        save_final = bool(cfg.checkpoint.get("save_final", True))
        if save_final and self.step % int(cfg.checkpoint.resume_every) != 0:
            self.ckpt.save_resume(self.step, self.model, self.optimizer, None)
        if save_final and self.step % int(cfg.checkpoint.milestone_every) != 0:
            self.ckpt.save_milestone(self.step, self.model)
        self.metrics.close()
        self.log.info("training finished at step %d", self.step)
