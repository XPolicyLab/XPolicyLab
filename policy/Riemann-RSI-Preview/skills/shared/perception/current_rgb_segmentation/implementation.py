from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

from rsi.resource_locations import load_manifest, reference_for

import cv2
import numpy as np

from harness.execution import ExecutionError
from harness.runtime.gpu_profiles import public_perception_resource
from skills.shared.motion.public_phase_execution.implementation import PublicRobotAPI
from skills.shared.perception.current_vessel_regions.implementation import boundary


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def validate_prompts(prompts):
    if (not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 3
            or any(not isinstance(prompt, str) or not 1 <= len(prompt) <= 128
                   or any(ord(character) < 32 for character in prompt) for prompt in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError('Current semantic segmentation accepts at most three bounded text descriptions')
    return tuple(prompts)


def observed_regions(image, masks, scores, maximum=12):
    image, masks, scores = np.asarray(image), np.asarray(masks), np.asarray(scores)
    if (image.shape != (480, 640, 3) or image.dtype != np.uint8 or masks.ndim != 3
            or masks.shape[1:] != (480, 640) or masks.dtype != bool or scores.shape != (len(masks),)
            or not np.isfinite(scores).all() or maximum != 12):
        raise ValueError('Segmentation returns only bounded masks of the current native public image')
    found = []
    for mask, score in sorted(zip(masks, scores), key=lambda value: value[1], reverse=True):
        if score < .45:
            continue
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            area = float(cv2.contourArea(contour))
            bbox, moments = cv2.boundingRect(contour), cv2.moments(contour)
            if not 40 <= area <= 480 * 640 * .30 or min(bbox[2:]) < 4 or not moments['m00']:
                continue
            outline = boundary(contour)
            if outline is None:
                continue
            centroid = np.asarray([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
            if any(np.linalg.norm(centroid - item['centroid']) <= max(3., min(bbox[2:]) / 3.) for item in found):
                continue
            found.append({'outline': outline, 'bbox': list(bbox), 'area': area, 'centroid': centroid.tolist(),
                'confidence': float(score), 'clipped': bool(np.any(outline <= 1.) or np.any(outline[:, 0] >= 638.)
                                                        or np.any(outline[:, 1] >= 478.)),
                'source': 'sam3_current_public_rgb_mask'})
            if len(found) == maximum:
                return found
    return found


class CurrentRGBSegmentation:
    def __init__(self):
        root = Path(__file__).resolve().parents[4]
        settings = json.loads((root / 'configs/native-perception.json').read_text())
        manifest = load_manifest(root / settings['manifest'])
        if (settings['model'] != 'SAM3' or settings['maximum_regions_per_label'] != 12
                or manifest['model'] != 'SAM3' or reference_for(manifest['path']) != 'resource://models/sam3-robodojo-v21'):
            raise ExecutionError('Current segmentation uses only its declared public pretrained model')
        assets = Path(manifest['path'])
        for relative in ('sam3_source', 'python_deps', 'overlay'):
            path = str(assets / relative)
            if path not in sys.path:
                sys.path.append(path)
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.torch = torch
        from harness.runtime.gpu_profiles import public_perception_cuda_device
        device = public_perception_cuda_device(root, settings['cuda_device'])
        self.device = 'cuda:' + str(device)
        if not torch.cuda.is_available() or torch.cuda.device_count() <= device:
            raise ExecutionError('The declared segmentation GPU must be included in this evaluation lease')
        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(checkpoint_path=str(assets / 'sam3.pt'), load_from_HF=False,
                device=self.device, eval_mode=True, compile=False).to(self.device)
            self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=.45)
        self.calls, self.images, self.prompts = 0, 0, 0
        self.cache_hits = 0
        self.current_image_hashes = None
        self.current_mask_outputs = {}

    def current_frame(self, api):
        images = {camera: api.image(camera) for camera in CAMERAS}
        hashes = tuple(hashlib.sha256(images[camera].tobytes()).hexdigest() for camera in CAMERAS)
        if hashes != getattr(self, 'current_image_hashes', None):
            self.current_image_hashes = hashes
            self.current_mask_outputs = {}
        return images

    def cached_current(self, key):
        if key not in self.current_mask_outputs:
            return None
        self.cache_hits = getattr(self, 'cache_hits', 0) + 1
        return deepcopy(self.current_mask_outputs[key])

    def remember_current(self, key, found):
        if len(self.current_mask_outputs) >= 8:
            self.current_mask_outputs.pop(next(iter(self.current_mask_outputs)))
        self.current_mask_outputs[key] = deepcopy(found)

    def prompted(self, api, prompts):
        if type(api) is not PublicRobotAPI:
            raise ExecutionError('Semantic masks receive only the restricted current public robot API')
        prompts = validate_prompts(prompts)
        images = self.current_frame(api)
        key = ('prompted', prompts)
        previous = self.cached_current(key)
        if previous is not None:
            return previous
        from PIL import Image
        found = {}
        with self.torch.cuda.device(self.device), self.torch.inference_mode(), self.torch.autocast(
                device_type='cuda', dtype=self.torch.bfloat16):
            for camera in CAMERAS:
                image = images[camera]
                image_state = self.processor.set_image(Image.fromarray(image), state=None)
                self.images += 1
                candidates = []
                for prompt in prompts:
                    self.processor.reset_all_prompts(image_state)
                    result = self.processor.set_text_prompt(prompt, image_state)
                    self.prompts += 1
                    masks = result['masks'].detach().cpu().numpy().reshape(-1, 480, 640).astype(bool)
                    scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                    candidates = observed_regions(image, masks, scores)
                    del result, masks, scores
                    if candidates:
                        break
                del image_state
                found[camera] = [dict(item, camera=camera) for item in candidates]
        self.calls += 1
        self.remember_current(key, found)
        return found

    def current(self, api):
        if type(api) is not PublicRobotAPI:
            raise ExecutionError('A segmentation skill receives only the restricted public robot API')
        images = self.current_frame(api)
        key = ('native_vessel_labels',)
        previous = self.cached_current(key)
        if previous is not None:
            return previous
        from PIL import Image
        found = {}
        with self.torch.cuda.device(self.device), self.torch.inference_mode(), self.torch.autocast(
                device_type='cuda', dtype=self.torch.bfloat16):
            for camera in CAMERAS:
                image = images[camera]
                image_state = self.processor.set_image(Image.fromarray(image), state=None)
                self.images += 1
                labels = {}
                for label in ('bottle', 'cup'):
                    candidates = []
                    prompts = (label, 'mug', 'wine glass') if camera == 'cam_head' and label == 'cup' else (label,)
                    for prompt in prompts:
                        self.processor.reset_all_prompts(image_state)
                        result = self.processor.set_text_prompt(prompt, image_state)
                        self.prompts += 1
                        masks = result['masks'].detach().cpu().numpy().reshape(-1, 480, 640).astype(bool)
                        scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                        candidates = observed_regions(image, masks, scores)
                        del result, masks, scores
                        if candidates:
                            break
                    labels[label] = [dict(item, camera=camera) for item in candidates]
                del image_state
                found[camera] = labels
        self.calls += 1
        self.remember_current(key, found)
        return found


@lru_cache(maxsize=1)
def backend():
    return CurrentRGBSegmentation()


def run(api):
    if type(api) is not PublicRobotAPI:
        raise ExecutionError('A segmentation skill receives only the restricted public robot API')
    with public_perception_resource(Path(__file__).resolve().parents[4]):
        return backend().current(api)


def prompted(api, prompts):
    if type(api) is not PublicRobotAPI:
        raise ExecutionError('Semantic masks receive only the restricted current public robot API')
    prompts = validate_prompts(prompts)
    with public_perception_resource(Path(__file__).resolve().parents[4]):
        return backend().prompted(api, prompts)


def statistics():
    value = backend() if backend.cache_info().currsize else None
    return {'triplet_inferences': value.calls if value else 0, 'image_inferences': value.images if value else 0,
            'text_prompts': value.prompts if value else 0, 'history_images_retained': 0,
            'same_current_frame_cache_hits': getattr(value, 'cache_hits', 0) if value else 0,
            'current_frame_prompt_sets_retained': len(getattr(value, 'current_mask_outputs', {})) if value else 0,
            'physical_asset_geometry_used': False, 'current_rgb_only': True}
