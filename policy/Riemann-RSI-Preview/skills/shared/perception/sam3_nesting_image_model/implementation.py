"""Pinned public SAM3 inference on an explicit current RGB image."""
import hashlib
from pathlib import Path
import sys
import numpy as np


def image_configuration(root=None):
    """Use the canonical model declaration on this single leased policy GPU."""
    from rsi.resource_locations import load_manifest, reference_for
    root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
    config = load_manifest(root/'configs/native-perception.json')
    if (config.get('schema') != 'riemann-current-rgb-segmentation-v1'
            or config.get('model') != 'SAM3' or config.get('maximum_regions_per_label') != 12
            or config.get('history_retained') is not False
            or config.get('manifest') != 'configs/models/sam3.json'):
        raise ValueError('Use the declared public pretrained segmentation model')
    manifest = load_manifest(root/config['manifest'])
    if (manifest.get('model') != 'SAM3'
            or reference_for(manifest['path']) != 'resource://models/sam3-robodojo-v21'):
        raise ValueError('Use the declared public pretrained segmentation model')
    # The former skill-local profile selected device0. The common profile's
    # device1 belongs to its two-device consumers, not this single-GPU binding.
    return dict(config, cuda_device=0), manifest


class PublicSAM3ImageModel:
    def __init__(self):
        config, manifest = image_configuration()
        assets = Path(manifest['path'])
        # Read exactly the declared members once. No directory or NAS scan.
        for name, record in manifest['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe model manifest member')
            path = assets/relative
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
            if path.stat().st_size != record['bytes'] or digest.hexdigest() != record['sha256']:
                raise ValueError('Declared public segmentation model changed: '+name)
        for member in ('sam3_source', 'python_deps', 'overlay'):
            path = str(assets/member)
            if path not in sys.path: sys.path.append(path)
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not torch.cuda.is_available() or torch.cuda.device_count() <= config['cuda_device']:
            raise ValueError('Use the explicitly leased segmentation GPU')
        self.torch, self.device = torch, 'cuda:'+str(config['cuda_device'])
        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(checkpoint_path=str(assets/'sam3.pt'),
                load_from_HF=False, device=self.device, eval_mode=True, compile=False).to(self.device)
            self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=.45)

    def prompted_masks(self, rgb, prompts):
        image = np.asarray(rgb)
        if image.shape != (480,640,3) or image.dtype != np.uint8:
            raise ValueError('Require an explicit current public 640x480 RGB image')
        if (not 1 <= len(prompts) <= 3 or any(type(p) is not str or not 1 <= len(p) <= 64 for p in prompts)):
            raise ValueError('Use at most three bounded appearance prompts')
        from PIL import Image
        records = []
        with self.torch.cuda.device(self.device), self.torch.inference_mode(), self.torch.autocast(
                device_type='cuda', dtype=self.torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(image), state=None)
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(prompt, state)
                masks = result['masks'].detach().cpu().numpy().reshape(-1,480,640).astype(bool)
                scores = result['scores'].detach().float().cpu().numpy().reshape(-1)
                indices = np.argsort(-scores)[:12]
                records.append((masks[indices].copy(),scores[indices].copy()))
                del result
            del state
        return records
