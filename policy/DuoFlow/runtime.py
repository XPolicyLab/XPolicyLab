"""加载 DuoFlow 使用的固定 Clean 骨干与原生部署变换。"""
import importlib.util
import inspect
import json
import os
from functools import lru_cache
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get('DUOFLOW_ASSETS', ROOT / 'assets')).expanduser().resolve()
MODEL_ROOT = ASSETS / 'backbone'
BACKBONE_SOURCE = ROOT / 'vendor/backbone_lib'
NATIVE_SOURCE = ROOT / 'vendor/native_release'


def _install_image_transform_compat() -> None:
    """允许旧 ImageTransform 接收官方 sidecar 的缺省 high_res_cam。"""
    from duoflow_models.pipelines.vla.action_core import action_core_utils as utils

    original = utils.ImageTransform
    if 'high_res_cam' in inspect.signature(original).parameters:
        return
    if getattr(original, '_sr_adapter_high_res_compat', False):
        return

    class CompatibleImageTransform(original):
        _sr_adapter_high_res_compat = True

        def __init__(
            self,
            is_train: bool,
            resize_imgs_with_padding: tuple[int, int],
            present_img_keys: list[str] | None = None,
            primary_img_key: str = 'observation.images.cam_high',
            enable_image_aug: bool = False,
            enable_depth_img: bool = False,
            depth_img_prefix_name: str | None = None,
            depth_img_mask_ratio: float = 0.5,
            history_frame_dropout_prob: float = 0.0,
            vlm_type: str = 'paligemma2',
            history_frame_dropout_mode: str = 'all',
            *,
            high_res_cam=None,
        ):
            if high_res_cam is not None:
                raise TypeError(
                    'inference_config.json sets high_res_cam, but the loaded '
                    'ImageTransform does not support that argument'
                )
            super().__init__(
                is_train=is_train,
                resize_imgs_with_padding=resize_imgs_with_padding,
                present_img_keys=present_img_keys,
                primary_img_key=primary_img_key,
                enable_image_aug=enable_image_aug,
                enable_depth_img=enable_depth_img,
                depth_img_prefix_name=depth_img_prefix_name,
                depth_img_mask_ratio=depth_img_mask_ratio,
                history_frame_dropout_prob=history_frame_dropout_prob,
                vlm_type=vlm_type,
                history_frame_dropout_mode=history_frame_dropout_mode,
            )

    CompatibleImageTransform.__name__ = original.__name__
    CompatibleImageTransform.__qualname__ = original.__qualname__
    CompatibleImageTransform.__module__ = original.__module__
    utils.ImageTransform = CompatibleImageTransform


def load_official():
    for path in (BACKBONE_SOURCE, NATIVE_SOURCE):
        sys.path.insert(0, str(path))
    path = NATIVE_SOURCE / 'scripts/inference/inference_agilex_server_unified.py'
    spec = importlib.util.spec_from_file_location('duoflow_native_server', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    _install_image_transform_compat()
    spec.loader.exec_module(module)
    model = module.get_policy(
        model_path=str(MODEL_ROOT),
        pretrained_path=str(ASSETS / 'paligemma2_tokenizer'),
        fast_tokenizer_path=str(ASSETS / 'fast_tokenizer'),
        embodiment_id=0,
        norm_stats_path=str(MODEL_ROOT / 'norm_stats.json'),
        use_bf16=True,
        robot_type='agilex_cobot_magic',
        original_action_dim=14,
        expected_state_dim=14,
    )
    return model


def build_demo_transforms(model):
    from duoflow_models.pipelines.vla.action_core.action_core_utils import (
        ImageTransform, PromptTokenizerTransform,
        infer_end_effector_type_from_delta_mask,
    )
    config = json.loads((MODEL_ROOT / 'inference_config.json').read_text())
    image_cfg, prompt_cfg = config['image_cfg'], config['prompt_cfg']
    delta_mask = config['delta_action_cfg']['mask']['agilex_cobot_magic']
    image_transform = ImageTransform(
        is_train=False,
        resize_imgs_with_padding=image_cfg['resize_imgs_with_padding'],
        present_img_keys=image_cfg['present_img_keys'],
        enable_image_aug=False,
        vlm_type='paligemma2',
    )
    prompt_transform = PromptTokenizerTransform(
        is_train=False,
        tokenizer_model_path=str(ASSETS / 'paligemma2_tokenizer'),
        fast_tokenizer_path=str(ASSETS / 'fast_tokenizer'),
        max_length=prompt_cfg['max_length'],
        discrete_state_input=prompt_cfg['discrete_state_input'],
        discrete_state_input_for_pose_embodiments=prompt_cfg.get(
            'discrete_state_input_for_pose_embodiments', False),
        encode_action_input=prompt_cfg['encode_action_input'],
        fast_token_vocab_mode=getattr(model, 'fast_token_vocab_mode', None),
        fast_token_tail_skip_tokens=getattr(model, 'fast_token_tail_skip_tokens', 128),
        fast_token_tail_vocab_size=getattr(model, 'fast_token_tail_vocab_size', None),
        encoded_action_horizon=prompt_cfg.get('encoded_action_horizon'),
        encode_sub_task_input=prompt_cfg['encode_sub_task_input'],
        enable_control_mode_token=prompt_cfg.get('enable_control_mode_token', False),
        control_mode_override='joint',
        enable_end_effector_token=prompt_cfg.get('enable_end_effector_token', False),
        end_effector_override=infer_end_effector_type_from_delta_mask(delta_mask),
        vlm_type='paligemma2',
        prefix_lm_text=prompt_cfg.get('prefix_lm_text', False),
        image_attn=prompt_cfg.get('image_attn', 'bidirectional'),
        use_chat_template=prompt_cfg.get('use_chat_template', False),
        chat_system_prompt=prompt_cfg.get('chat_system_prompt'),
        prompt_filler_text=prompt_cfg.get('prompt_filler_text'),
        resize_imgs_with_padding=image_cfg['resize_imgs_with_padding'],
        state_input_mode=config['state_input_mode'],
    )
    if prompt_transform.propri_token_id != model.propri_token_id:
        raise ValueError('官方tokenizer与模型的propri token不匹配')

    @lru_cache(maxsize=32)
    def tokenize_task(task):
        tokens, masks, attention_masks, *_ = prompt_transform(
            {'task': task, 'embodiment_id': 0})
        return {'lang_tokens': tokens, 'lang_masks': masks,
                'lang_att_masks': attention_masks}

    def tokenize(context):
        return tokenize_task(context['task'])

    return image_transform, tokenize
