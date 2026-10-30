"""Ground independent proposal crops with a fixed local public CLIP checkpoint."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path

import numpy as np


MODEL_ID = 'openai/clip-vit-base-patch32'
MODEL_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
DEFAULT_MODEL_PATH = str(Path(__policy_root__) / 'model_assets/classification/clip-vit-base-patch32')
ROTATIONS = (0, 90, 180, 270)
PROMPTS = {
    'watch': (
        'a photo of a wristwatch lying on a wooden table',
        'a small silver or black wrist watch with a circular wrist strap',
        'a close-up of an unfastened wristwatch with a metal case and a band',
        'a black or silver watch bracelet with a hollow circular strap'),
    'car': (
        'a photo of a small toy sports car on a wooden table',
        'a miniature die-cast toy car with four black wheels',
        'a green, red, white or yellow toy automobile',
        'a close-up of a toy racing car'),
    'wooden_toy': (
        'a photo of a small wooden toy animal on a table',
        'a pale wooden toy horse or rocking horse',
        'a small carved wooden animal toy for children',
        'a close-up of a wooden toy made of cream and brown wood'),
    'pepper': (
        'a photo of a bell pepper vegetable on a wooden table',
        'a red, green or yellow bell pepper vegetable with a small stem',
        'a close-up of a whole sweet pepper',
        'a small round bell pepper vegetable'),
    'chocolate_bar': (
        'a photo of a wrapped chocolate candy bar on a wooden table',
        'a long thin rectangular chocolate bar in a printed plastic wrapper',
        'a chocolate candy bar in a white or purple packet',
        'a close-up of a packaged chocolate bar'),
    'toy_car': (
        'a photo of a small toy vehicle on a wooden table',
        'a small wooden toy car or wooden toy truck',
        'a small toy truck or toy excavator',
        'a close-up of a toy car'),
    'action_camera': (
        'a photo of a small black action camera on a table',
        'a rectangular GoPro action camera with a lens',
        'a small black camera body',
        'a close-up of an action camera'),
    'pen': (
        'a photo of a writing pen on a wooden table',
        'a long thin black or white ballpoint pen',
        'a pen or marker used for writing',
        'a close-up of a slim writing pen'),
    'garage': (
        'a photo of a small stuffed toy animal on a wooden table',
        'a plush rabbit or gray cartoon animal figurine with big eyes',
        'a small fluffy toy monster or toy rabbit',
        'a close-up of a cute stuffed animal toy'),
    'phone': (
        'a photo of a small toy mobile phone on a table',
        'a blue or turquoise mobile phone with white buttons',
        'a small cell phone with a keypad',
        'a close-up of a toy telephone'),
}
NEGATIVE_CATEGORIES = ('robot_gripper', 'robot_arm', 'table', 'table_line', 'basket')
PROMPTS.update({
    'robot_gripper': ('a white and black robot gripper', 'a two finger robotic gripper',
                      'a robotic hand', 'a white plastic camera housing on a robot gripper'),
    'robot_arm': ('a black robot arm', 'an industrial robotic arm',
                  'a metal robot wrist', 'a robot manipulator'),
    'table': ('an empty wooden tabletop', 'a bare wooden table surface',
              'brown wood grain on an empty table', 'a wooden table with no objects'),
    'table_line': ('a dark thin line in wood grain', 'a long narrow shadow on a wooden table',
                   'a shadow cast by a plastic basket', 'a scratch in a wooden tabletop'),
    'basket': ('an empty plastic basket', 'a red blue or white plastic storage tray',
               'a rectangular plastic basket', 'a plastic storage bin'),
})

_ENCODER = None


class ClipCropEncoder:
    """Share fixed text vectors and content-addressed rotation-averaged crops."""

    def __init__(self, folder=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(1)
        self.torch = torch
        self.folder = Path(folder or os.environ.get('CLASSIFY_CLIP_PATH', DEFAULT_MODEL_PATH))
        self.processor = CLIPProcessor.from_pretrained(str(self.folder), local_files_only=True)
        self.model = CLIPModel.from_pretrained(str(self.folder), local_files_only=True).eval().cpu()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.text_cache = {}
        self.crop_cache = OrderedDict()
        self.statistics = dict(crop_calls=0, crops_encoded=0, crop_cache_hits=0, text_calls=0)

    def _normalize(self, features):
        return self.torch.nn.functional.normalize(features.float(), dim=-1)

    def text_features(self, categories):
        categories = tuple(str(category) for category in categories)
        if categories not in self.text_cache:
            prompts = []
            counts = []
            for category in categories:
                values = PROMPTS.get(category, ('a photo of a '+category.replace('_', ' '),))
                prompts.extend(values)
                counts.append(len(values))
            tokens = self.processor.tokenizer(prompts, padding=True, return_tensors='pt')
            with self.torch.inference_mode():
                output = self.model.text_model(**tokens)
                features = self._normalize(self.model.text_projection(output.pooler_output))
                averages = []
                start = 0
                for count in counts:
                    averages.append(features[start:start+count].mean(dim=0))
                    start += count
                vectors = self._normalize(self.torch.stack(averages))
            self.text_cache[categories] = vectors
            self.statistics['text_calls'] += 1
        return self.text_cache[categories]

    @staticmethod
    def _crop_key(crop):
        array = np.ascontiguousarray(crop, dtype=np.uint8)[..., :3]
        digest = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
        return digest, array

    def image_features(self, crops):
        from PIL import Image, ImageOps
        self.statistics['crop_calls'] += 1
        entries = [self._crop_key(crop) for crop in crops]
        pending = {}
        for key, crop in entries:
            if key in self.crop_cache:
                self.statistics['crop_cache_hits'] += 1
                self.crop_cache.move_to_end(key)
            elif key not in pending:
                pending[key] = crop
        keys = list(pending)
        # Small batches bound CPU inference memory independently of scene size.
        for begin in range(0, len(keys), 12):
            batch_keys = keys[begin:begin+12]
            pictures = []
            for key in batch_keys:
                picture = Image.fromarray(pending[key]).convert('RGB')
                square = ImageOps.pad(picture, (224, 224), method=Image.Resampling.BICUBIC,
                                      color=(128, 128, 128))
                pictures.extend(square.rotate(angle) for angle in ROTATIONS)
            inputs = self.processor.image_processor(images=pictures, return_tensors='pt',
                                                     do_resize=False, do_center_crop=False)
            with self.torch.inference_mode():
                output = self.model.vision_model(pixel_values=inputs['pixel_values'])
                features = self._normalize(self.model.visual_projection(output.pooler_output))
                features = self._normalize(features.reshape(len(batch_keys), len(ROTATIONS), -1).mean(dim=1))
            for key, vector in zip(batch_keys, features):
                self.crop_cache[key] = vector.clone()
            self.statistics['crops_encoded'] += len(batch_keys)
        vectors = self.torch.stack([self.crop_cache[key] for key, _ in entries])
        while len(self.crop_cache) > 2048:
            self.crop_cache.popitem(last=False)
        return vectors

    def __call__(self, crops, categories):
        if not crops:
            return np.empty((0, len(categories)), dtype=np.float32)
        images = self.image_features(crops)
        text = self.text_features(categories)
        with self.torch.inference_mode():
            scores = images@text.T
        return scores.cpu().numpy()


class ContextClipClassifier:
    """Include nearby observed pixels so a thin segmented rim keeps its full outline."""

    def __init__(self, ctx, model):
        self.ctx = ctx
        self.model = model
        self.statistics = model.statistics

    def __call__(self, crops, categories):
        import cv2
        if not crops:
            return np.empty((0, len(categories)), dtype=np.float32)
        cameras = self.ctx.obs['vision']
        camera = next((name for name in cameras if 'head' in name), next(iter(cameras)))
        image = np.asarray(cameras[camera]['color'], np.uint8)[..., :3]
        contexts = []
        for crop in crops:
            crop = np.asarray(crop, np.uint8)[..., :3]
            # Every crop comes from this observed RGB frame; exact template alignment
            # recovers its box without object state, labels, depth, or layout metadata.
            errors = cv2.matchTemplate(image, crop, cv2.TM_SQDIFF)
            _, _, location, _ = cv2.minMaxLoc(errors)
            x, y = location
            height, width = crop.shape[:2]
            margin = max(8, min(16, int(round(.20*max(height, width)))))
            contexts.append(image[max(0, y-margin):min(image.shape[0], y+height+margin),
                                  max(0, x-margin):min(image.shape[1], x+width+margin)].copy())
        scores = self.model(list(crops)+contexts, categories)
        count = len(crops)
        return .25*scores[:count]+.75*scores[count:]


def encoder():
    global _ENCODER
    folder = Path(os.environ.get('CLASSIFY_CLIP_PATH', DEFAULT_MODEL_PATH))
    if _ENCODER is None or _ENCODER.folder != folder:
        _ENCODER = ClipCropEncoder(folder)
    return _ENCODER
