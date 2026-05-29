"""
Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import os
import random
import re
import json
import numpy as np
import torch
import torch.distributed as dist
from termcolor import cprint

from accelerate.logging import get_logger
from source.utils import BOLD, RESET_BOLD

logger = get_logger(__name__)


# === Shared training helpers ===


def setup_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """
    Initialize torch distributed backend based on torchrun env vars.
    Returns: (is_distributed, rank, world_size, device)
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return distributed, rank, world_size, device


def build_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    scheduler_type: str,
    min_lr_cfg: Optional[Dict[str, float]] = None,
):
    """Warmup + cosine/constant scheduler with optional per-group min LR clamp.

    min_lr_cfg keys follow param group names (e.g., base, qwen_model, action_model, classifier) with
    min_lr_cfg["base"] as fallback. Warmup supports zero steps.
    """

    def build_lambda(init_lr: float, group_name: str):
        min_lr = None
        if min_lr_cfg:
            min_lr = min_lr_cfg.get(group_name, min_lr_cfg.get("base"))
        min_ratio = (min_lr / init_lr) if (min_lr is not None and init_lr > 0) else None
        if min_ratio is not None:
            min_ratio = min(min_ratio, 1.0)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                factor = float(current_step) / float(max(1, warmup_steps))
            else:
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                factor = 0.5 * (1.0 + np.cos(np.pi * progress)) if scheduler_type == "cosine" else 1.0
            if min_ratio is None:
                return factor
            return factor * (1.0 - min_ratio) + min_ratio

        return lr_lambda

    lambdas = [build_lambda(g.get("lr", 0.0), g.get("name", "")) for g in optimizer.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def extract_scalar_metrics(outputs: Dict) -> Dict[str, float]:
    """Convert tensor/numeric loss dict to plain floats for logging/printing."""
    scalar_logs: Dict[str, float] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            scalar_logs[key] = float(value.detach().float().item())
        elif isinstance(value, (float, int, np.floating, np.integer)):
            scalar_logs[key] = float(value)
    return scalar_logs


def build_classifier_wandb_payload(metrics: Dict[str, float]) -> Dict[str, float]:
    """Strip console-only classifier fields and map _recent values onto base keys."""
    payload = {}
    for key, value in metrics.items():
        if key.endswith("_recent") or key in {"classifier_count", "classifier_count_recent", "classifier_window_size"} or key.startswith("cls_"):
            continue
        payload[key] = value
    for base_key, recent_key in {
        "classifier_accuracy": "classifier_accuracy_recent",
        "classifier_precision": "classifier_precision_recent",
        "classifier_recall": "classifier_recall_recent",
        "classifier_f1_score": "classifier_f1_score_recent",
        "classifier_prate": "classifier_prate_recent",
    }.items():
        if recent_key in metrics:
            payload[base_key] = metrics[recent_key]
    return payload


def log_step_metrics(
    scalar_losses: Dict[str, float],
    lr: float,
    global_step: int,
    log_interval: int,
    wandb_run=None,
    console_only: Optional[Dict[str, float]] = None,
):
    """
    Consolidated per-step logging plus optional W&B push.
    Prints ordered metrics in the format:
    loss / action_loss / classifier_loss / classifier_accuracy = v1 / v2 / v3 / v4
    """
    log_payload = {**scalar_losses, "lr": lr, "step": global_step}
    console_payload = log_payload if console_only is None else {**log_payload, **console_only}
    if global_step % max(1, log_interval) == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
        # Build ordered metrics string: loss / action_loss / classifier_loss / classifier_accuracy ...
        keys_order = ["loss", "action_loss", "classifier_loss", "classifier_accuracy", "classifier_prate"]
        ordered = [k for k in keys_order if k in console_payload]
        remaining = [k for k in console_payload if k not in ordered]
        ordered += remaining
        loss_keys = [k for k in ["loss", "action_loss", "classifier_loss"] if k in console_payload]
        acc_keys = [k for k in ["classifier_accuracy", "classifier_precision", "classifier_recall"] if k in console_payload]
        tail_keys = [k for k in ["classifier_f1_score", "classifier_prate", "classifier_count"] if k in console_payload]
        counts_parts = []
        tp = console_payload.get("cls_tp_recent")
        fp = console_payload.get("cls_fp_recent")
        fn = console_payload.get("cls_fn_recent")
        correct = console_payload.get("cls_correct_recent")
        total = console_payload.get("cls_total_recent")
        positive = console_payload.get("cls_positive_recent")
        prec_den = tp + fp if tp is not None and fp is not None else None
        recall_den = tp + fn if tp is not None and fn is not None else None
        f1_den = (2 * tp + fp + fn) if tp is not None and fp is not None and fn is not None else None
        if correct is not None and total is not None:
            counts_parts.append(f"acc = {BOLD}{correct:.1f}{RESET_BOLD} / {BOLD}{total:.1f}{RESET_BOLD}")
        if tp is not None and prec_den is not None:
            counts_parts.append(f"prec = {BOLD}{tp:.1f}{RESET_BOLD} / {BOLD}{prec_den:.1f}{RESET_BOLD}")
        if tp is not None and recall_den is not None:
            counts_parts.append(f"recall = {BOLD}{tp:.1f}{RESET_BOLD} / {BOLD}{recall_den:.1f}{RESET_BOLD}")
        if tp is not None and f1_den is not None:
            counts_parts.append(f"f1 = {BOLD}{(2*tp):.1f}{RESET_BOLD} / {BOLD}{f1_den:.1f}{RESET_BOLD}")
        if positive is not None and total is not None:
            counts_parts.append(f"prate = {BOLD}{positive:.1f}{RESET_BOLD} / {BOLD}{total:.1f}{RESET_BOLD}")
        lines = [
            f"[train] step = {BOLD}{global_step}{RESET_BOLD}; lr = {BOLD}{lr:.6e}{RESET_BOLD}",
        ]
        for key in loss_keys:
            lines.append(f"        {key} = {BOLD}{console_payload[key]:.4f}{RESET_BOLD}")
        for key in acc_keys:
            lines.append(f"        {key} = {BOLD}{console_payload[key]:.4f}{RESET_BOLD}")
        for key in tail_keys:
            lines.append(f"        {key} = {BOLD}{console_payload[key]:.4f}{RESET_BOLD}")
        if counts_parts:
            lines.append("        classifier counts:")
            for part in counts_parts:
                lines.append(f"            {part}")
        cprint("\n".join(lines), "cyan")
    if wandb_run is not None and global_step % max(1, log_interval) == 0:
        wandb_run.log(log_payload, step=global_step)


def rank0_print(message: str, color: str = "green") -> None:
    """Print only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        cprint(message, color)


def destroy_distributed() -> None:
    """Clean up distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


class CheckpointManager:
    """Handle checkpoint persistence with step-based frequency and optional final dump."""

    def __init__(self, save_dir: Optional[Path], save_every_steps: int, save_final: bool, rank: int):
        self.rank = rank
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every_steps = save_every_steps
        self.save_final_flag = save_final
        self.enabled = self.save_dir is not None and self.rank == 0 and (save_every_steps > 0 or save_final)

        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            freq_msg = f"every {self.save_every_steps} steps" if self.save_every_steps > 0 else "final-only"
            cprint(f"[ckpt] Enabled checkpointing -> dir = {BOLD}{self.save_dir}{RESET_BOLD} ({freq_msg})", "green")

    def _dump(self, model, optimizer, scheduler, global_step: int, tag: str) -> None:
        """Serialize training state to disk; DDP unwraps automatically."""
        model_to_save = model.module if hasattr(model, "module") else model
        payload = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
        }
        ckpt_path = self.save_dir / f"{tag}.pt"
        torch.save(payload, ckpt_path)
        cprint(f"[ckpt] Saved checkpoint -> {BOLD}{ckpt_path}{RESET_BOLD}", "green")

    def maybe_save(self, model, optimizer, scheduler, global_step: int) -> None:
        """Save on configured step boundaries."""
        if not self.enabled or self.save_every_steps <= 0:
            return
        if global_step % self.save_every_steps == 0:
            self._dump(model, optimizer, scheduler, global_step, tag=f"step{global_step}")

    def save_final(self, model, optimizer, scheduler, global_step: int) -> None:
        """Optional final checkpoint at training end."""
        if not self.enabled or not self.save_final_flag:
            return
        self._dump(model, optimizer, scheduler, global_step, tag=f"final_step{global_step}")


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
        else:
            pass  # skip orphaned values
    return normalized


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

    # unwrap DDP/Accelerate wrappers so attr traversal works
    base_model = model.module if hasattr(model, "module") else model

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = base_model
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
        # try to find the module under vla by module_name (support nested paths)
        module = base_model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters
            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in base_model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
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


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


class TrainerUtils:
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

        if dist.is_initialized():
            dist.barrier()  # synchronize when distributed training
            if dist.get_rank() == 0:
                print(f"🔒 Frozen modules with re pattern: {frozen}")
        else:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
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
        if dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
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
                        if dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=True)
                if dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
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
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    # public alias
    reset_dataloader = _reset_dataloader

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
        # dist.barrier()
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


import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0
