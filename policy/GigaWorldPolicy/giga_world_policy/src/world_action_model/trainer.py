import json
import os
from datetime import timedelta

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm


class DictConfig(dict):
    """Dict that also supports attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


class ModuleDict(nn.ModuleDict):
    """nn.ModuleDict with forward dispatching by key."""

    def forward(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)


def _install_process_group_timeout(timeout_sec) -> None:
    if not timeout_sec:
        return

    timeout = timedelta(seconds=int(timeout_sec))
    os.environ.setdefault("DEEPSPEED_TIMEOUT", str(max(1, int(timeout.total_seconds() // 60))))

    try:
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d

        dist.constants.default_pg_timeout = timeout
        c10d.default_pg_timeout = timeout

        if not hasattr(dist, "_gwp_original_new_group"):
            dist._gwp_original_new_group = dist.new_group

            def _new_group_with_default_timeout(*args, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = getattr(dist, "_gwp_new_group_timeout", timeout)
                return dist._gwp_original_new_group(*args, **kwargs)

            dist.new_group = _new_group_with_default_timeout
        dist._gwp_new_group_timeout = timeout
    except Exception as exc:
        print(f"[WARN] failed to install distributed timeout patch: {exc}", flush=True)


class EMA:
    """FP32 EMA for trainable floating point parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.995, device: str | torch.device | None = None):
        self.decay = float(decay)
        self.device = torch.device(device) if device and str(device) != "model" else None
        self.updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _target_device(self, param: torch.Tensor) -> torch.device:
        return self.device or param.device

    def _init_shadow(self, model: nn.Module):
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                self.shadow[name] = param.detach().float().to(self._target_device(param)).clone()

    @property
    def tracked_names(self) -> set[str]:
        return set(self.shadow.keys())

    def update(self, model: nn.Module):
        if not self.shadow:
            self._init_shadow(model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                shadow.mul_(self.decay).add_(
                    param.detach().float().to(device=shadow.device),
                    alpha=1.0 - self.decay,
                )
        self.updates += 1

    def state_dict(self, cpu: bool = True):
        shadow = {
            name: tensor.detach().cpu().clone() if cpu else tensor.detach().clone()
            for name, tensor in self.shadow.items()
        }
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": shadow,
        }

    def load_state_dict(self, state_dict: dict):
        if "shadow" in state_dict:
            shadow_state = state_dict.get("shadow", {})
            self.decay = float(state_dict.get("decay", self.decay))
            self.updates = int(state_dict.get("updates", self.updates))
        else:
            shadow_state = state_dict

        for name, tensor in shadow_state.items():
            if name not in self.shadow or not torch.is_tensor(tensor):
                continue
            self.shadow[name] = tensor.detach().float().to(device=self.shadow[name].device).clone()

    def load_from_model_state_dict(self, model_state: dict[str, torch.Tensor]):
        for name, shadow in list(self.shadow.items()):
            tensor = model_state.get(name)
            if torch.is_tensor(tensor):
                self.shadow[name] = tensor.detach().float().to(device=shadow.device).clone()

    def apply_shadow(self, model: nn.Module):
        self._backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                self._backup[name] = param.detach().clone()
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                backup = self._backup.get(name)
                if backup is not None:
                    param.copy_(backup)
        self._backup.clear()


class Trainer:
    """Base trainer using HuggingFace Accelerate + DeepSpeed.

    Subclasses must implement:
        get_models(model_config) -> ModuleDict
        forward_step(batch_dict) -> dict[str, Tensor]
    """

    def __init__(self, config: dict):
        self.config = config
        train_cfg = config.get("train", {})

        mixed_precision = train_cfg.get("mixed_precision", "no")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            mixed_precision, torch.float32
        )

        gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        timeout_sec = (
            train_cfg.get("process_group_timeout_sec")
            or train_cfg.get("distributed_timeout")
            or os.environ.get("GIGAWORLD_DISTRIBUTED_TIMEOUT")
            or os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SEC")
            or os.environ.get("TORCH_NCCL_TIMEOUT_SEC")
            or 3600
        )
        _install_process_group_timeout(timeout_sec)
        accelerator_kwargs = {"gradient_accumulation_steps": gradient_accumulation_steps}
        if timeout_sec:
            accelerator_kwargs["kwargs_handlers"] = [
                InitProcessGroupKwargs(timeout=timedelta(seconds=int(timeout_sec)))
            ]
        self.accelerator = Accelerator(**accelerator_kwargs)

        self.device = self.accelerator.device
        self.process_index = self.accelerator.process_index
        self.cur_step = 0
        self._outputs: list = []
        self.model: ModuleDict | None = None

    def get_models(self, model_config: DictConfig) -> ModuleDict:
        raise NotImplementedError

    def forward_step(self, batch_dict: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint, models, strict=True):
        if checkpoint is None:
            return
        if isinstance(checkpoint, (list, tuple)):
            for ckpt in checkpoint:
                self.load_checkpoint(ckpt, models, strict=strict)
            return

        if self.process_index == 0:
            print(f"Loading checkpoint: {checkpoint}")

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        for m in models:
            # Try loading as-is first; if all keys are unexpected, strip common prefix and retry
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            if unexpected and missing:
                # Detect a shared prefix in unexpected keys (e.g. "transformer.")
                prefixes = set(k.split(".")[0] + "." for k in unexpected)
                if len(prefixes) == 1:
                    prefix = prefixes.pop()
                    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                    if stripped:
                        missing, unexpected = m.load_state_dict(stripped, strict=strict)
                        if self.process_index == 0:
                            print(f"  Stripped prefix '{prefix}' from checkpoint keys")
            if self.process_index == 0:
                if missing:
                    print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    def _build_optimizer(self, model: nn.Module):
        opt_cfg = self.config.get("optimizers", {})
        opt_type = opt_cfg.get("type", "AdamW")
        lr = opt_cfg.get("lr", 1e-4)
        weight_decay = opt_cfg.get("weight_decay", 0.01)
        action_lr_mult = opt_cfg.get("action_lr_mult", 1.0)

        # Split params: action branch vs backbone
        action_keywords = ("action_encoder", "action_decoder", "state_encoder", "action_rope")
        action_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in action_keywords):
                action_params.append(param)
            else:
                backbone_params.append(param)

        if action_lr_mult != 1.0 and action_params:
            param_groups = [
                {"params": backbone_params, "lr": lr},
                {"params": action_params, "lr": lr * action_lr_mult},
            ]
            if self.process_index == 0:
                print(f"Optimizer: backbone lr={lr:.2e}, action lr={lr * action_lr_mult:.2e} "
                      f"(×{action_lr_mult}), {len(backbone_params)} + {len(action_params)} params")
        else:
            param_groups = [{"params": backbone_params + action_params, "lr": lr}]

        if opt_type in ("CAME", "CAME8Bit"):
            try:
                from came_pytorch import CAME
                return CAME(param_groups, lr=lr, weight_decay=weight_decay)
            except ImportError:
                if self.process_index == 0:
                    print("came_pytorch not installed, falling back to AdamW")
                return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        if opt_type == "Adam8Bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.Adam8bit(param_groups, lr=lr, weight_decay=weight_decay)
            except ImportError:
                if self.process_index == 0:
                    print("bitsandbytes not installed, falling back to AdamW")
                return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self, optimizer):
        sch_cfg = self.config.get("schedulers", {})
        sch_type = sch_cfg.get("type", "ConstantScheduler")
        max_steps = self._resolved_max_steps  # use the resolved value (epoch-aware)
        warmup_steps = sch_cfg.get("warmup_steps", 0)
        min_lr_ratio = sch_cfg.get("min_lr_ratio", 0.0)  # cosine decays to lr * min_lr_ratio

        if sch_type == "CosineScheduler":
            base_lr = optimizer.param_groups[0]["lr"]
            # support decay_epochs: convert to steps using the same pre-estimate logic
            decay_epochs = sch_cfg.get("decay_epochs", None)
            if decay_epochs is not None:
                train_cfg = self.config.get("train", {})
                max_epochs = train_cfg.get("max_epochs", 0)
                decay_steps = int(round(max_steps * decay_epochs / max_epochs)) if max_epochs > 0 else None
            else:
                decay_steps = sch_cfg.get("decay_steps", None)
            decay_lr = sch_cfg.get("decay_lr", None)
            # decay_lr takes priority over min_lr_ratio if both are set
            if decay_lr is not None:
                eta_min = decay_lr
            else:
                eta_min = base_lr * min_lr_ratio
            cosine_steps = (decay_steps if decay_steps is not None else max_steps) - warmup_steps

            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
            schedulers = []
            milestones = []
            if warmup_steps > 0:
                schedulers.append(LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps))
                milestones.append(warmup_steps)
            schedulers.append(CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min))
            # if decay_steps < max_steps, hold at eta_min for the remaining steps
            if decay_steps is not None and decay_steps < max_steps:
                schedulers.append(ConstantLR(optimizer, factor=eta_min / base_lr, total_iters=max_steps - decay_steps))
                milestones.append((decay_steps if warmup_steps == 0 else decay_steps))
            if len(schedulers) == 1:
                return schedulers[0]
            return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

        # ConstantScheduler (with optional warmup)
        if warmup_steps > 0:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, total_iters=warmup_steps
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def _build_dataset(self):
        import world_action_model as _wam
        from .datasets.lerobot_dataset import LeRobotDataset

        dl_cfg = self.config["dataloaders"]["train"]

        transform_cfg = dict(dl_cfg.get("transform", {}))
        transform_type = transform_cfg.pop("type")
        transform_cls = getattr(_wam, transform_type)
        transform = transform_cls(**transform_cfg)

        data_configs = dl_cfg.get("data_or_config", [])
        is_main = int(os.environ.get("RANK", "0")) == 0

        def _load_one(dc):
            return LeRobotDataset(
                data_path=dc.get("data_path"),
                delta_info=dc.get("delta_info"),
                delta_frames=dc.get("delta_frames"),
                video_backend=dc.get("video_backend", "pyav"),
                transform=transform,
                t5_embed_path=dc.get("t5_embed_path"),
                robotype=dc.get("robotype", "aloha"),
            )

        if len(data_configs) > 5:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            datasets = [None] * len(data_configs)
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {pool.submit(_load_one, dc): i for i, dc in enumerate(data_configs)}
                pbar = tqdm(total=len(data_configs), desc="Loading datasets", disable=not is_main)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    datasets[idx] = fut.result()
                    pbar.update(1)
                pbar.close()
        else:
            datasets = [_load_one(dc) for dc in data_configs]

        if len(datasets) == 1:
            return datasets[0]
        return torch.utils.data.ConcatDataset(datasets)

    def _build_dataloader(self, dataset):
        dl_cfg = self.config["dataloaders"]["train"]
        return DataLoader(
            dataset,
            batch_size=dl_cfg.get("batch_size_per_gpu", 1),
            shuffle=True,
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

    def run(self):
        project_dir = self.config.get("project_dir", "./output")
        os.makedirs(project_dir, exist_ok=True)
        train_cfg = self.config.get("train", {})

        model_config = DictConfig(self.config["models"])
        self.model = self.get_models(model_config)

        dataset = self._build_dataset()
        dataloader = self._build_dataloader(dataset)
        if self.process_index == 0:
            print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

        optimizer = self._build_optimizer(self.model)

        if self.process_index == 0:
            print("Preparing model with DeepSpeed (this may take a few minutes)...")
        self.model, optimizer, dataloader = self.accelerator.prepare(
            self.model, optimizer, dataloader,
        )
        if self.process_index == 0:
            print("DeepSpeed ready.")

        # Compute max_steps from actual post-prepare dataloader length
        max_steps = train_cfg.get("max_steps", 100000)
        max_epochs = train_cfg.get("max_epochs", 0)
        steps_per_epoch = None
        if max_epochs > 0:
            steps_per_epoch = len(dataloader)
            max_steps = max_epochs * steps_per_epoch
            if self.process_index == 0:
                print(f"Epoch mode: max_epochs={max_epochs}, steps_per_epoch={steps_per_epoch}, max_steps={max_steps}")
        self._resolved_max_steps = max_steps

        # Build scheduler AFTER prepare so it uses the correct max_steps
        scheduler = self._build_scheduler(optimizer)

        max_steps = self._resolved_max_steps
        checkpoint_interval = train_cfg.get("checkpoint_interval", 5000)
        checkpoint_epoch_interval = int(train_cfg.get("checkpoint_epoch_interval", 0) or 0)
        log_interval = train_cfg.get("log_interval", 1)
        ema_cfg = train_cfg.get("ema", {}) or {}
        with_ema = bool(ema_cfg.get("enabled", train_cfg.get("with_ema", False)))
        ema_decay = float(ema_cfg.get("decay", train_cfg.get("ema_decay", 0.995)))
        ema_device = ema_cfg.get("device", train_cfg.get("ema_device", "model"))

        unwrapped = self.accelerator.unwrap_model(self.model)
        ema = EMA(unwrapped, decay=ema_decay, device=ema_device) if with_ema else None
        if self.process_index == 0 and ema is not None:
            print(
                f"EMA enabled: decay={ema.decay}, device={ema_device}, "
                f"tracked={len(ema.shadow)} trainable floating tensors"
            )

        if self.process_index == 0:
            wandb_cfg = self.config.get("wandb", {})
            wandb_settings = wandb.Settings(
                init_timeout=int(wandb_cfg.get("init_timeout", os.environ.get("WANDB_INIT_TIMEOUT", 300)))
            )
            wandb.init(
                project=wandb_cfg.get("project", "gwp-xpolicylab"),
                name=wandb_cfg.get("name", os.path.basename(project_dir)),
                config=self.config,
                dir=project_dir,
                resume="allow",
                mode=wandb_cfg.get("mode", "online"),
                settings=wandb_settings,
            )

        if train_cfg.get("resume", False):
            self.cur_step = self._try_resume(project_dir, ema)

        self.model.train()
        if self.process_index == 0:
            print(f"Starting training from step {self.cur_step}, max_steps={max_steps}")

        pbar = None
        if self.process_index == 0:
            pbar = tqdm(initial=self.cur_step, total=max_steps, desc="Training", unit="step")

        while self.cur_step < max_steps:
            for batch in dataloader:
                if self.cur_step >= max_steps:
                    break

                with self.accelerator.accumulate(self.model):
                    losses = self.forward_step(batch)
                    total_loss = sum(losses.values())
                    self.accelerator.backward(total_loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                self.cur_step += 1

                if ema is not None:
                    ema.update(unwrapped)

                if self.process_index == 0 and self.cur_step % log_interval == 0:
                    log_dict = {}
                    postfix = {}
                    for k, v in losses.items():
                        val = v.item() if hasattr(v, "item") else v
                        postfix[k] = f"{val:.4f}"
                        log_dict[f"train/{k}"] = val
                    log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
                    if ema is not None:
                        log_dict["ema/updates"] = ema.updates
                    postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    wandb.log(log_dict, step=self.cur_step)
                    if pbar is not None:
                        pbar.set_postfix(postfix)

                if pbar is not None:
                    pbar.update(1)

                should_save_step_checkpoint = checkpoint_interval > 0 and self.cur_step % checkpoint_interval == 0
                should_save_epoch_checkpoint = (
                    checkpoint_epoch_interval > 0
                    and steps_per_epoch
                    and self.cur_step < max_steps
                    and self.cur_step % (steps_per_epoch * checkpoint_epoch_interval) == 0
                )
                if should_save_step_checkpoint or should_save_epoch_checkpoint:
                    self._save_checkpoint(project_dir, ema)

                self._outputs.clear()

        if pbar is not None:
            pbar.close()

        self._save_checkpoint(project_dir, ema)
        self.accelerator.end_training()
        if self.process_index == 0:
            wandb.finish()
            print("Training finished.")

    @staticmethod
    def _trainable_floating_param_names(model: nn.Module) -> set[str]:
        return {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.is_floating_point()
        }

    @staticmethod
    def _state_dict_for_save(
        model: nn.Module,
        fp32_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        state = {}
        for name, tensor in model.state_dict().items():
            if torch.is_tensor(tensor):
                out = tensor.detach().cpu()
                if fp32_names is not None and name in fp32_names and tensor.is_floating_point():
                    out = out.float()
                state[name] = out
            else:
                state[name] = tensor
        return state

    @staticmethod
    def _state_dict_with_ema(
        model: nn.Module,
        ema: EMA,
        fp32_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        state = Trainer._state_dict_for_save(model, fp32_names=fp32_names)
        for name, shadow in ema.shadow.items():
            if name in state:
                state[name] = shadow.detach().cpu().float().clone()
        return state

    @staticmethod
    def _extract_model_state(state_dict: dict) -> dict:
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        if "model_state_dict" in state_dict:
            return state_dict["model_state_dict"]
        return state_dict

    def _save_checkpoint(self, project_dir: str, ema: EMA | None = None):
        self.accelerator.wait_for_everyone()
        save_dir = os.path.join(project_dir, f"checkpoint-{self.cur_step}")

        if self.process_index == 0:
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            trainable_names = self._trainable_floating_param_names(unwrapped)
            raw_state = self._state_dict_for_save(unwrapped, fp32_names=trainable_names)
            torch.save(raw_state, os.path.join(save_dir, "model.pt"))

            if ema is not None:
                ema_state = self._state_dict_with_ema(unwrapped, ema, fp32_names=ema.tracked_names)
                torch.save(ema_state, os.path.join(save_dir, "model_ema.pt"))
                torch.save(ema.state_dict(cpu=True), os.path.join(save_dir, "ema_state.pt"))

            torch.save(
                {
                    "cur_step": self.cur_step,
                    "ema_updates": ema.updates if ema is not None else 0,
                },
                os.path.join(save_dir, "training_state.pt"),
            )
            meta = {
                "format": "world_action_model_full_checkpoint",
                "raw_model_path": "model.pt",
                "raw_fp32_trainable": True,
                "num_trainable_tensors": len(trainable_names),
                "ema_enabled": ema is not None,
            }
            if ema is not None:
                meta.update({
                    "ema_model_path": "model_ema.pt",
                    "ema_state_path": "ema_state.pt",
                    "ema_decay": ema.decay,
                    "ema_updates": ema.updates,
                    "ema_fp32_trainable": True,
                    "ema_tracked_tensors": len(ema.tracked_names),
                })
            with open(os.path.join(save_dir, "checkpoint_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Checkpoint saved to {save_dir}")

        self.accelerator.wait_for_everyone()

    def _try_resume(self, project_dir: str, ema: EMA | None = None) -> int:
        if not os.path.isdir(project_dir):
            return 0

        checkpoints = [
            d for d in os.listdir(project_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(project_dir, d))
        ]
        if not checkpoints:
            return 0

        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        ckpt_dir = os.path.join(project_dir, latest)

        model_path = os.path.join(ckpt_dir, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = self._extract_model_state(state_dict)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.load_state_dict(state_dict, strict=False)

        step = 0
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            step = state.get("cur_step", 0)
            if ema is not None:
                ema.updates = int(state.get("ema_updates", ema.updates))
            if self.process_index == 0:
                print(f"Resumed from {ckpt_dir} at step {step}")

        if ema is not None:
            ema_state_path = os.path.join(ckpt_dir, "ema_state.pt")
            ema_model_path = os.path.join(ckpt_dir, "model_ema.pt")
            if os.path.exists(ema_state_path):
                ema_state = torch.load(ema_state_path, map_location="cpu", weights_only=False)
                ema.load_state_dict(ema_state)
                if self.process_index == 0:
                    print(f"Loaded EMA state from {ema_state_path} (updates={ema.updates})")
            elif os.path.exists(ema_model_path):
                ema_model_state = torch.load(ema_model_path, map_location="cpu", weights_only=False)
                ema.load_from_model_state_dict(self._extract_model_state(ema_model_state))
                if self.process_index == 0:
                    print(f"Seeded EMA shadow from full EMA checkpoint {ema_model_path}")

        if step:
            return step

        return 0
