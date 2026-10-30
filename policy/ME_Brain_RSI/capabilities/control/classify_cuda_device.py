"""Use resident FP32 vision models on the worker-assigned CUDA device."""
from functools import lru_cache

import numpy as np


_STATE = {}


def _move_inputs(module, args, kwargs):
    """Move model tensors while preserving their original dtype and structure."""
    torch = _STATE['torch']

    def move(value):
        if torch.is_tensor(value):
            return value.to('cuda:0')
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    return move(args), move(kwargs)


def _cpu_detector_output(module, args, output):
    """Keep the original GroundingDINO thresholds and text decoding on CPU."""
    for field in ('logits', 'pred_boxes'):
        value = getattr(output, field).cpu()
        setattr(output, field, value)
        output[field] = value
    return output


def _encode(crops):
    """Retain the original four-rotation DINOv2 CLS encoding and NumPy result."""
    from PIL import Image, ImageOps

    torch = _STATE['torch']
    model, processor = _STATE['dino_model'], _STATE['dino_processor']
    pictures = []
    for crop in crops:
        square = ImageOps.pad(Image.fromarray(np.asarray(crop, np.uint8)).convert('RGB'),
            (224, 224), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
        pictures.extend(square.rotate(angle) for angle in (0, 90, 180, 270))
    inputs = processor(images=pictures, return_tensors='pt', do_resize=False, do_center_crop=False)
    with torch.inference_mode():
        features = model(**inputs).last_hidden_state[:, 0].float()
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features.reshape(len(crops), 4, -1).mean(1)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy()


@lru_cache(maxsize=1)
def enable():
    """Configure private process bindings before either classification observer runs."""
    import torch
    from coap.tasks import classify_grounding_dino as grounding
    from coap.tasks import classify_r05_grounded_clip as clipping
    from coap.tasks import sorting_r03_dino as dino
    from coap.tasks import classify_r07_n2_material_centroid as normal
    from coap.tasks import classify_r25_b_language as language

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    detector, clip = grounding.proposer(), clipping.encoder()
    dino_model, dino_processor = dino._load_encoder()
    _STATE.update(torch=torch, dino_model=dino_model, dino_processor=dino_processor)
    for model in (detector.model, clip.model, dino_model):
        model.float().eval().to('cuda:0')
    detector.model.register_forward_pre_hook(_move_inputs, with_kwargs=True)
    detector.model.register_forward_hook(_cpu_detector_output)
    clip.model.vision_model.register_forward_pre_hook(_move_inputs, with_kwargs=True)
    clip.model.text_model.register_forward_pre_hook(_move_inputs, with_kwargs=True)
    dino_model.register_forward_pre_hook(_move_inputs, with_kwargs=True)
    clip.crop_cache.clear()
    clip.text_cache.clear()
    language._text_vectors.cache_clear()
    normal._encode = language._encode = _encode
    detector.environment.update(device='cuda:0', dtype='float32', postprocessing_device='cpu')
    return dict(helper='classify_r28_cuda_device', device='cuda:0', dtype='float32',
        postprocessing_device='cpu', autocast=False, tf32_matmul=False, tf32_cudnn=False,
        added_motion_calls=0, runtime_safety_checks_added=False)
