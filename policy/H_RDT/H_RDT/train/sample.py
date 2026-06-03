from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    hrdt, args, config, accelerator, weight_dtype, dataset_id2name, 
    dataloader, logger, vision_encoder
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )
    hrdt.eval()

    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    # Initialize overall counters
    loss_counter["overall_avg_sample_mse"] = 0
    loss_counter["overall_avg_sample_l2err"] = 0

    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break

        # Process image data
        if isinstance(batch["images"], dict):
            # {"dino": (B, T, C, H, W), "dino": (B, T, C, H, W)}
            images = {k: v.to(dtype=weight_dtype) for k, v in batch["images"].items()}
        else:
            raise ValueError(f"Unsupported `batch[\"images\"]` type = {type(batch['images'])}")

        # Extract VLM features
        with torch.no_grad():
            k = next(iter(images))
            batch_size, _, C, H, W = images[k].shape
            for k in images:
                images[k] = images[k].view(-1, C, H, W)
            image_features = vision_encoder(images).detach()
            image_features = image_features.view((batch_size, -1, vision_encoder.embed_dim))

        # Process language data based on training mode
        lang_embeds = None
        lang_attn_mask = None
        if args.training_mode == "lang":
            lang_embeds = batch["lang_embeds"].to(dtype=weight_dtype)
            lang_attn_mask = batch["lang_attn_mask"].to(dtype=weight_dtype)

        # Get current state
        states = batch["states"].to(dtype=weight_dtype)
        
        # Get ground truth actions for evaluation
        actions = batch["actions"].to(weight_dtype)
        action_norm = batch["action_norm"].to(weight_dtype)
        dataset_indices = batch["data_indices"]
        
        # Sample actions using the model
        pred_actions = hrdt.predict_action(
            state_tokens=states,
            image_tokens=image_features,
            lang_tokens=lang_embeds,
            lang_attn_mask=lang_attn_mask,
        )
        
        num_steps = pred_actions.shape[1]
        expanded_action_norm = action_norm.float()

        # Compute metrics
        loss = F.mse_loss(pred_actions, actions, reduction='none').float()

        batch_size = pred_actions.shape[0]
        mse_loss_per_entry = loss.reshape((batch_size, -1)).mean(1)
        l2_loss_per_entry = loss.sqrt() / (expanded_action_norm + 1e-3)
        l2_loss_per_entry = l2_loss_per_entry.reshape((batch_size, -1)).mean(1)

        # Gather metrics across processes
        dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics(
            (torch.LongTensor(dataset_indices).to(device=pred_actions.device),
             mse_loss_per_entry, l2_loss_per_entry),
        )
        dataset_indices = dataset_indices.tolist()
        
        mse_loss_all = mse_losses
        overall_mse = mse_loss_all.mean().item()
        loss_for_log["overall_avg_sample_mse"] += overall_mse

        l2_loss_all = l2_losses
        overall_l2 = l2_loss_all.mean().item()
        loss_for_log["overall_avg_sample_l2err"] += overall_l2

        # Log metrics per dataset
        if accelerator.is_main_process:
            for loss_suffix, losses in zip(["_sample_mse", "_sample_l2err"], [mse_losses, l2_losses]):
                for dataset_idx, loss_tensor in zip(dataset_indices, losses):
                    loss_name = dataset_id2name[dataset_idx] + loss_suffix
                    loss_for_log[loss_name] += loss_tensor.item()
                    loss_counter[loss_name] += 1

        # Increment overall counters
        loss_counter["overall_avg_sample_mse"] += 1
        loss_counter["overall_avg_sample_l2err"] += 1

    # Average metrics
    for name in loss_for_log:
        loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)

    result_dict = {}
    for name, value in dict(loss_for_log).items():
        if name.startswith("overall_avg_"):
            new_name = name.replace("overall_avg_sample_", "overall_avg_")
            result_dict[f"action/metrics/{new_name}"] = value
        else:
            new_name = name.replace("_sample_", "_")
            result_dict[f"action/dataset_metrics/{new_name}"] = value

    hrdt.train()
    torch.cuda.empty_cache()

    return result_dict