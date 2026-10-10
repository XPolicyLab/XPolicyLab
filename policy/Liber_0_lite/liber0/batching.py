"""Batched conditioning and sampling with independent per-item noise."""
import torch

from .attention import build_batched_joint_mask
from .backbone import _build_raw_image_rope_layout


def prepare_condition(model, prompts, images, target_len):
    batch, _, height, width = images.shape
    templates = [model.processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}],
        tokenize=False, add_generation_prompt=True,
    ) for prompt in prompts]
    processed = model._process_semantic_references(templates, (images,), padding='longest')
    tokenizer = model.processor.tokenizer
    suffix = tokenizer.encode(tokenizer.boi_token + tokenizer.tms_token,
                              return_tensors='pt', add_special_tokens=False).expand(batch, -1)
    ids = torch.cat([processed.input_ids, suffix], dim=-1)
    valid = torch.cat([processed.attention_mask.bool(), torch.ones_like(suffix, dtype=torch.bool)], dim=-1)
    semantic_grids = processed.image_grid_thw.clone()
    semantic_grids[:, 1:] //= int(model.backbone.config.vision_config.spatial_merge_size)
    if semantic_grids.shape[0] != batch:
        raise ValueError('Expected one semantic image per batch item.')
    grid = model._target_grids((images,), height, width, target_len)
    image_token = int(model.backbone.config.image_token_id)
    vision_start = int(model.backbone.config.vision_start_token_id)
    raw_tokens, raw_grids, skip = _build_raw_image_rope_layout(
        target_grid=grid, reference_grids=[grid[0]], target_len=target_len,
        reference_lengths=[target_len], image_token_id=image_token,
        vision_start_token_id=vision_start, batch_size=batch, dtype=ids.dtype,
    )
    rope_ids = torch.cat([ids, raw_tokens], dim=-1)
    rope_valid = torch.cat([valid, torch.ones_like(raw_tokens, dtype=torch.bool)], dim=-1)
    positions = []
    for index in range(batch):
        position, _ = model._get_rope_index(
            1, image_token, int(model.backbone.config.video_token_id), vision_start,
            input_ids=rope_ids[index:index + 1],
            image_grid_thw=torch.cat([semantic_grids[index:index + 1], raw_grids]),
            video_grid_thw=None, attention_mask=rope_valid[index:index + 1],
            skip_vision_start_token=skip,
        )
        positions.append(position)
    ids = ids.to(model.device)
    text = model.backbone.model.get_input_embeddings()(ids)
    image_embeds, deepstack = model.backbone.model.get_image_features(
        processed.pixel_values.to(model.device, model.torch_dtype),
        processed.image_grid_thw.to(model.device),
    )
    image_embeds = torch.cat(image_embeds).to(text.dtype)
    image_mask, _ = model.backbone.model.get_placeholder_mask(ids, inputs_embeds=text, image_features=image_embeds)
    return dict(text=text.masked_scatter(image_mask, image_embeds),
                tms_mask=ids == int(model.backbone.model.tms_token_id),
                valid=valid.to(model.device), positions=torch.cat(positions, dim=1).to(model.device),
                visual_mask=image_mask[..., 0].bool(),
                deepstack=[value.to(model.device, model.torch_dtype) for value in deepstack])


def joint_forward(model, condition, refs, proprio, video, action, video_time, action_time):
    batch, action_len = action.shape[:2]
    target_len, ref_len = video.shape[1], refs.shape[1]
    text = condition['text']
    time = model.backbone.model.t_embedder1(video_time.expand(batch)).to(text.dtype)
    text = torch.where(condition['tms_mask'].unsqueeze(-1), time.unsqueeze(1).expand_as(text), text)
    video_tokens = model._embed_video_tokens(video.to(model.torch_dtype), refs.to(model.torch_dtype))
    actions = model._embed_action_tokens(action, action_time.expand(batch))
    state = model.dit.proprio_encoder(proprio.to(model.torch_dtype)).unsqueeze(1) + model.dit.proprio_type_embedding
    embeds = torch.cat([text, state, video_tokens, actions], dim=1)
    positions = model._joint_positions(condition['positions'], condition['valid'], target_len, ref_len, action_len)
    visual_mask = torch.cat([condition['visual_mask'], torch.zeros(
        batch, 1 + target_len + ref_len + action_len, device=model.device, dtype=torch.bool,
    )], dim=1)
    mask = build_batched_joint_mask(condition['valid'], target_len, ref_len, action_len, model.torch_dtype)
    output = model._language_model_forward(
        input_ids=None, position_ids=positions, attention_mask=mask,
        inputs_embeds=embeds, use_cache=False, visual_pos_masks=visual_mask,
        deepstack_visual_embeds=condition['deepstack'],
    ).last_hidden_state
    video_start = text.shape[1] + 1
    action_start = video_start + target_len + ref_len
    prediction = model.backbone.model.final_layer2(output[:, video_start:video_start + target_len])
    return prediction, model._predict_action(output[:, action_start:action_start + action_len])


def initial_noise(model, images, action_horizon, seeds):
    videos, actions = [], []
    for seed in seeds:
        generator = torch.Generator('cpu').manual_seed(int(seed))
        videos.append(torch.randn((1, *images.shape[1:]), generator=generator).to(model.device, model.sample_dtype) * model.noise_scale)
        actions.append(torch.randn((1, action_horizon, model.dit.action_dim), generator=generator).to(model.device, model.sample_dtype))
    return torch.cat(videos), torch.cat(actions)


@torch.no_grad()
def infer_actions(model, prompts, images, proprio, action_horizon, seeds, steps):
    batch = images.shape[0]
    if not batch or len(prompts) != batch or len(seeds) != batch or proprio.shape != (batch, model.dit.proprio_dim):
        raise ValueError('Inconsistent inference batch dimensions.')
    if not 0 < action_horizon <= model.dit.max_action_horizon:
        raise ValueError('Unsupported action horizon.')
    images = images.to(model.device, model.sample_dtype)
    proprio = proprio.to(model.device, model.sample_dtype)
    video, action = initial_noise(model, images, action_horizon, seeds)
    video = model._patchify(video, model.patch_size)
    refs = model._patchify(images, model.patch_size)
    condition = prepare_condition(model, prompts, images, video.shape[1])
    video_t, video_delta = model.infer_video_scheduler.build_inference_schedule(steps, model.device, model.sample_dtype)
    action_t, action_delta = model.infer_action_scheduler.build_inference_schedule(steps, model.device, model.sample_dtype)
    for index in range(steps):
        video_sigma = video_t[index:index + 1] / model.infer_video_scheduler.num_train_timesteps
        action_sigma = action_t[index:index + 1] / model.infer_action_scheduler.num_train_timesteps
        pred_x0, pred_action = joint_forward(model, condition, refs, proprio, video, action,
                                           1 - video_sigma, 1 - action_sigma)
        video_velocity = (video - pred_x0) / video_sigma[:, None, None].clamp_min(model.timestep_eps)
        video = model.infer_video_scheduler.step(video_velocity, video_delta[index], video)
        action_velocity = model._action_model_output_to_velocity(pred_action, action, action_sigma)
        action = model.infer_action_scheduler.step(action_velocity, action_delta[index], action)
    return action.cpu().float()
