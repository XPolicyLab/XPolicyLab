"""Bounded SAM3 masks from explicit public image arrays, on the existing GPU."""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
import numpy as np

from rsi.resource_locations import load_manifest


class ArraySegmenter:
    def __init__(self, manifest_path):
        manifest = load_manifest(manifest_path)
        if manifest['model'] != 'SAM3':
            raise ValueError('An explicit pretrained SAM3 manifest is required')
        root = Path(manifest['path']).resolve(strict=True)
        # Explicit manifest members only: no directory walk, download or install.
        for name, record in manifest['files'].items():
            path = root / name
            if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
                raise ValueError('Model member escapes its declared asset root')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Pretrained public model member changed: ' + name)
        for name in ('sam3_source', 'python_deps', 'overlay'):
            path = str(root / name)
            if path not in sys.path:
                sys.path.append(path)
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not torch.cuda.is_available():
            raise RuntimeError('Public segmentation requires the leased simulator GPU')
        self.torch = torch
        self.model = build_sam3_image_model(checkpoint_path=str(root / 'sam3.pt'),
            load_from_HF=False, device='cuda:0', eval_mode=True, compile=False).to('cuda:0')
        self.processor = Sam3Processor(self.model, device='cuda:0', confidence_threshold=.45)

    def run(self, image, prompts, *, minimum_score=.45):
        from PIL import Image
        if type(minimum_score) not in (int,float) or minimum_score not in (.2,.45):
            raise ValueError('Only the default .45 or corroborated-candidate .2 threshold is supported')
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError('An explicit native public RGB array is required')
        if (not isinstance(prompts, (tuple, list)) or not 1 <= len(prompts) <= 3
                or any(not isinstance(p, str) or not 1 <= len(p) <= 128 for p in prompts)):
            raise ValueError('Provide one to three bounded semantic descriptions')
        processor = self.processor
        if minimum_score == .2:
            # Keep the default processor unchanged for other callers. These
            # processors share the same authenticated model weights.
            if not hasattr(self, '_candidate_processor'):
                from sam3.model.sam3_image_processor import Sam3Processor
                self._candidate_processor = Sam3Processor(self.model, device='cuda:0', confidence_threshold=.2)
            processor = self._candidate_processor
        result = []
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            state = processor.set_image(Image.fromarray(image), state=None)
            for prompt in prompts:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(prompt, state)
                masks = output['masks'].detach().cpu().numpy().reshape(-1, 480, 640).astype(bool)
                scores = output['scores'].detach().float().cpu().numpy().reshape(-1)
                for index in np.argsort(scores)[::-1][:8]:
                    mask = masks[index]
                    if scores[index] < minimum_score or not 30 <= mask.sum() <= 480 * 640 * .3:
                        continue
                    if any(np.count_nonzero(mask & r['mask']) / max(1, np.count_nonzero(mask | r['mask'])) > .8 for r in result):
                        continue
                    result.append({'mask': mask.copy(), 'score': float(scores[index]), 'prompt': prompt})
                    if len(result) == 8:
                        break
                del output, masks, scores
                if result:
                    break
            del state
        return result


@lru_cache(maxsize=1)
def backend(manifest_path):
    return ArraySegmenter(manifest_path)


def run(image, prompts, manifest_path, *, minimum_score=.45):
    return backend(str(manifest_path)).run(image, prompts, minimum_score=minimum_score)
