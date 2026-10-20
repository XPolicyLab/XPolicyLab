"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from pathlib import Path
from typing import Tuple
import os
import re
import json
import numpy as np
import torch
import torch.distributed as dist
from transformers import get_scheduler

from accelerate.logging import get_logger

from starVLA.training.trainer_utils.nan_fuse import assert_finite_training_tensors

logger = get_logger(__name__)


def _dist_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


CHECKPOINT_FILENAME_RE = re.compile(r"^steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def resolve_resume_checkpoint(spec) -> Tuple[str, int]:
    """`trainer.resume_from` -> (checkpoint file, completed optimizer steps).

    Accepts a `steps_<N>_*` checkpoint file, a run directory, or a run's
    `checkpoints/` directory (highest step wins). The step count is parsed from
    the filename; it is what the LR schedule is fast-forwarded to, so a renamed
    checkpoint is rejected rather than silently restarting the schedule.

    Lives here rather than in ``train_starvla.py`` so ``train_starvla_cotrain``
    can share it: both entrypoints build a module-level ``Accelerator``, so
    importing one from the other would construct a second one.
    """
    path = Path(spec).expanduser()

    if path.is_dir():
        found = []
        for candidate_dir in (path / "checkpoints", path):
            if not candidate_dir.is_dir():
                continue
            found = [
                (int(match.group(1)), entry)
                for entry in candidate_dir.iterdir()
                if entry.is_file() and (match := CHECKPOINT_FILENAME_RE.match(entry.name))
            ]
            if found:
                break
        if not found:
            raise FileNotFoundError(
                f"trainer.resume_from=`{spec}` holds no steps_<N>_pytorch_model.pt / "
                "steps_<N>_model.safetensors checkpoint."
            )
        completed_steps, path = max(found)
        return str(path), completed_steps

    if not path.is_file():
        # train.sh absolutizes the YAML / SPARK_0_RESUME_FROM value, but a CLI
        # dotlist override is merged after that and stays relative to the
        # training cwd (source_starvla), so report it.
        raise FileNotFoundError(
            f"trainer.resume_from=`{spec}` does not exist (resolved from cwd {os.getcwd()}). "
            "Use an absolute path when overriding it on the command line."
        )

    match = CHECKPOINT_FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"trainer.resume_from=`{spec}` must be named steps_<N>_pytorch_model.pt or "
            "steps_<N>_model.safetensors: the step the LR schedule resumes at is parsed "
            "from the filename."
        )
    return str(path), int(match.group(1))


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        elif "=" in arg:
            # Bare OmegaConf dotlist override `x.y=val` — the form the wrapper
            # scripts document (`bash scripts/... trainer.max_train_steps=20000`).
            # Upstream dropped these silently.
            normalized.append(arg)
        # anything else: orphaned value, skip
    return normalized


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build AdamW optimizer + LR scheduler from cfg.trainer.

    Supports:
    - Per-module learning rates via cfg.trainer.learning_rate
    - cosine_with_min_lr (or any transformers scheduler) via cfg.trainer.lr_scheduler_type
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    fused_available = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused_available,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = cfg.trainer.scheduler_specific_kwargs
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # Resolve the module under the model by (possibly nested) path.
        #   - a MISSING attribute means the lr key names a module that does not
        #     exist -> hard error (typo / stale config), so mis-set learning
        #     rates never get silently dropped into the base group.
        #   - an attribute that RESOLVES TO None is a disabled optional module
        #     (e.g. spatial_align when framework.spatial_alignment.enable=false);
        #     that is legitimate, so we skip it quietly.
        module = model
        disabled_optional = False
        for attr in module_name.split("."):
            if not hasattr(module, attr):
                raise ValueError(
                    f"trainer.learning_rate names module path `{module_name}`, "
                    f"but `{attr}` does not exist on the model. Fix the config "
                    f"key or remove it (available top-level modules: "
                    f"{[n for n, _ in model.named_children()]})."
                )
            module = getattr(module, attr)
            if module is None:
                disabled_optional = True
                break
        if disabled_optional:
            continue
        # filter out frozen parameters
        params = [p for p in module.parameters() if id(p) not in frozen_params]
        if params:  # only add param group if there are trainable parameters
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def _is_single_hw(size) -> bool:
    """True for one PIL resize target ``(W, H)`` / ``[W, H]`` / int."""
    if isinstance(size, bool):
        return False
    if isinstance(size, int):
        return True
    return (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and not isinstance(size[0], (list, tuple))
    )


def expand_obs_image_sizes(spec, video_keys, default=(384, 384)):
    """Expand ``obs_image_size`` into one ``(W, H)`` per video key.

    Mirrors the training dataloader / deploy resolver: int / ``[W, H]``
    (uniform) or a per-camera mapping (short name or ``video.*`` key, optional
    ``default`` entry).
    """
    if not video_keys:
        raise ValueError("expand_obs_image_sizes requires a non-empty video_keys list.")

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
    except Exception:
        pass

    def _parse_hw(value):
        if isinstance(value, bool):
            raise ValueError(f"obs_image_size entry must be [W, H], got {value!r}.")
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"obs_image_size entry must be [W, H], got {value!r}.")

    def _short(name: str) -> str:
        text = str(name)
        return text.split(".", 1)[-1] if "." in text else text

    if spec is None:
        return [default for _ in video_keys]
    if _is_single_hw(spec):
        size = _parse_hw(spec)
        return [size for _ in video_keys]
    if not isinstance(spec, dict):
        raise ValueError(
            "obs_image_size must be an int, [W, H], or a per-camera mapping, "
            f"got {spec!r}."
        )

    by_short = {}
    mapping_default = None
    for raw_key, raw_size in spec.items():
        key = str(raw_key)
        if key in {"default", "_default"}:
            mapping_default = _parse_hw(raw_size)
            continue
        by_short[_short(key)] = _parse_hw(raw_size)

    sizes = []
    missing = []
    for video_key in video_keys:
        short = _short(video_key)
        if short in by_short:
            sizes.append(by_short[short])
        elif mapping_default is not None:
            sizes.append(mapping_default)
        else:
            missing.append(str(video_key))
    if missing:
        raise ValueError(
            f"obs_image_size mapping is missing camera(s) {missing}; "
            f"known keys={sorted(by_short)}."
        )
    return sizes


def resolve_predict_resize_targets(config, n_views, view_mask=None):
    """Training-resolution resize target(s) for a framework's predict path.

    Reads ``datasets.vla_data.obs_image_size`` (alias ``image_size``) from the
    run config. Returns ``None`` when no size is configured, one ``(W, H)``
    for uniform sizes, or a per-view list aligned with
    ``embodiment_spec.video_keys`` for per-camera mappings.

    Per-camera mappings REQUIRE ``embodiment_spec.video_keys`` in the config
    (train.sh always writes it): without the key order there is no safe way
    to assign sizes to positional eval views.

    ``view_mask`` (optional, length = len(video_keys)): when flexible_views
    omits absent cameras from the image list, pass the mask so sizes follow
    the present subsequence rather than the first ``n_views`` catalog keys.
    """
    vla_data_cfg = getattr(getattr(config, "datasets", None), "vla_data", None)
    if vla_data_cfg is None:
        return None
    spec = getattr(vla_data_cfg, "obs_image_size", None)
    if not spec:
        spec = getattr(vla_data_cfg, "image_size", None)
    if not spec:
        return None

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
    except Exception:
        pass

    if not isinstance(spec, dict):
        return spec  # uniform int / [W, H]

    video_keys = None
    embodiment_spec = getattr(config, "embodiment_spec", None)
    if embodiment_spec is not None:
        try:
            video_keys = list(embodiment_spec.get("video_keys") or [])
        except Exception:
            video_keys = None
    if video_keys and view_mask is not None:
        if len(view_mask) != len(video_keys):
            raise ValueError(
                f"view_mask length {len(view_mask)} != embodiment_spec.video_keys "
                f"length {len(video_keys)}."
            )
        present_keys = [key for key, present in zip(video_keys, view_mask) if present]
        if len(present_keys) != n_views:
            raise ValueError(
                f"view_mask selects {len(present_keys)} camera(s) but the example "
                f"carries {n_views} image(s)."
            )
        return expand_obs_image_sizes(spec, present_keys)
    if video_keys and len(video_keys) == n_views:
        return expand_obs_image_sizes(spec, video_keys)
    if any(str(k) not in {"default", "_default"} for k in spec):
        raise ValueError(
            "per-camera obs_image_size requires embodiment_spec.video_keys in the "
            f"run config matching the {n_views} eval view(s) so predict_action can "
            "resize views in training order."
        )
    # default-only mapping: uniform after all
    return expand_obs_image_sizes(spec, [f"view_{i}" for i in range(max(n_views, 1))])


def resize_images(images, target_size=(384, 384)):
    """
    Recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: one ``(W, H)`` applied to every image, OR a list of
        ``(W, H)`` sizes aligned with the views of each sample
        (``List[List[Image]]`` -> resize view i with ``target_size[i]``).
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        if not _is_single_hw(target_size):
            raise ValueError(
                f"Resizing a single image requires one (W, H) target_size, got {target_size!r}."
            )
        size = (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
        return images.resize(size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        # Per-view sizes for one sample: [img0, img1, ...] + [(W,H), ...]
        if (
            images
            and isinstance(images[0], Image.Image)
            and isinstance(target_size, (list, tuple))
            and target_size
            and not _is_single_hw(target_size)
        ):
            if len(images) != len(target_size):
                raise ValueError(
                    f"Got {len(images)} view image(s) but {len(target_size)} "
                    f"target size(s): {target_size!r}."
                )
            return [
                img.resize((sz, sz) if isinstance(sz, int) else tuple(sz))
                for img, sz in zip(images, target_size)
            ]
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


import torch.distributed as dist


class TrainerUtils:
    def _sync_train_step_counter(self) -> None:
        """Publish the optimizer-step counter to step-dependent schedules.

        Today only the action head's shortcut-mask ramp reads it
        (``LayerwiseFlowmatchingActionHead.set_train_step``). Pushed from the
        trainer rather than self-incremented inside the head because gradient
        accumulation and ``repeated_diffusion_steps`` decouple forward calls
        from optimizer steps, and ``resume_from`` must resume the schedule
        instead of restarting it. A head without the hook is a no-op, so
        every other framework / action model is unaffected.
        """
        model = getattr(self, "model", None)
        while model is not None and not hasattr(model, "action_model"):
            model = getattr(model, "module", None)
        setter = getattr(getattr(model, "action_model", None), "set_train_step", None)
        if callable(setter):
            setter(int(getattr(self, "completed_steps", 0)))

    def _nan_fuse(self, parts, *, grads_ok: bool = True) -> None:
        """Trip the NaN fuse for this optimizer-step counter (DDP-safe)."""
        assert_finite_training_tensors(
            parts,
            step=int(getattr(self, "completed_steps", 0)),
            extra_ok=grads_ok,
            extra_name="param_grad",
        )

    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if _dist_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if _dist_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if _dist_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if _dist_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                incompatible = model.load_state_dict(checkpoint, strict=False)
                if _dist_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                    # strict=False silently skips mismatches; surface them so a
                    # warm start (e.g. baseline ckpt into a model with extra
                    # heads like spatial_align) is visible instead of silent.
                    for label, keys in (
                        ("missing (kept at init values)", incompatible.missing_keys),
                        ("unexpected (ignored)", incompatible.unexpected_keys),
                    ):
                        if keys:
                            shown = ", ".join(keys[:8]) + (" ..." if len(keys) > 8 else "")
                            print(f"⚠️ {len(keys)} {label} keys: {shown}")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _propagate_epoch(dataloader, epoch: int) -> None:
        """Forward ``set_epoch`` to every component that drives per-epoch sampling.

        accelerate's prepared dataloader nests the pieces differently depending
        on ``dispatch_batches`` / shuffle (a DistributedSampler may sit inside a
        BatchSampler inside a BatchSamplerShard, or be reachable straight from
        ``dataloader.sampler``), so walk the known layouts instead of assuming
        one. Dedup by identity — the same sampler is often reachable twice.

        The dataset itself is included for the num_workers=0 case; with
        persistent workers only the sampler path (shuffled indices) reaches
        the worker processes, which is why the train dataloader shuffles.
        """
        seen: set[int] = set()
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        inner_batch_sampler = getattr(batch_sampler, "batch_sampler", None)
        candidates = [
            getattr(dataloader, "sampler", None),
            batch_sampler,
            getattr(batch_sampler, "sampler", None),
            inner_batch_sampler,
            getattr(inner_batch_sampler, "sampler", None),
            getattr(dataloader, "dataset", None),
        ]
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            set_epoch = getattr(candidate, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        TrainerUtils._propagate_epoch(dataloader, epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        # Find all checkpoints matching the naming convention, supports .pt and .safetensors
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) 
            if re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))  # ensure it is a file
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Extract step numbers and sort
        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        # Sort by step number and get the latest checkpoint
        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")
