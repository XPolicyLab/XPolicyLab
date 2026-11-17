"""Serve SAM3 RGB segmentation requests from a shared directory."""

import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image


def configured_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'Set {name} to a local directory')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def write_json(path, data):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def segment(request, processor, model, torch):
    with np.load(request) as source:
        image = Image.fromarray(source['rgb'])
        prompts = source['prompts'].tolist()
        boxes = source['boxes'].tolist() if 'boxes' in source else None

    masks, scores = [], []
    extra = {} if boxes is None else dict(input_boxes=[boxes], input_boxes_labels=[[1] * len(boxes)])
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors='pt', **extra).to('cuda')
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=.25, mask_threshold=.5,
            target_sizes=inputs['original_sizes'].tolist(),
        )[0]
        for mask, score in zip(result['masks'], result['scores']):
            masks.append(mask.detach().cpu().numpy().astype(np.uint8))
            scores.append(float(score))
    return np.asarray(masks, dtype=np.uint8).reshape((-1, image.height, image.width)), np.asarray(scores)


def process_request(request, processor, model, torch):
    stem = request.name.removesuffix('.request.npz')
    response = request.with_name(stem + '.response.npz')
    error = request.with_name(stem + '.error.json')
    processed = request.with_name(stem + '.processed.npz')
    if not response.exists() and not error.exists():
        temporary = response.with_suffix('.writing')
        try:
            masks, scores = segment(request, processor, model, torch)
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, masks=masks, scores=scores)
            for attempt in range(4):
                try:
                    temporary.replace(response)
                    break
                except OSError:
                    if response.exists():
                        break
                    if attempt == 3:
                        raise
                    time.sleep(.25)
            print(stem, len(masks), flush=True)
        except Exception:
            write_json(error, {'error': traceback.format_exc()})
            traceback.print_exc()
        finally:
            temporary.unlink(missing_ok=True)
    request.replace(processed)


def main():
    root = configured_path('SND_SAM_ROOT')
    weights = configured_path('SND_SAM_WEIGHTS')
    if not weights.is_dir():
        raise NotADirectoryError(f'SND_SAM_WEIGHTS is not a directory: {weights}')
    root.mkdir(exist_ok=True, parents=True)

    import torch
    from transformers import Sam3Model, Sam3Processor

    torch.set_num_threads(4)
    processor = Sam3Processor.from_pretrained(str(weights), local_files_only=True)
    model = Sam3Model.from_pretrained(str(weights), local_files_only=True).to('cuda').eval()
    write_json(root / 'ready.json', {'pid': os.getpid(), 'weights': str(weights),
                                     'device': str(next(model.parameters()).device)})
    print('READY', flush=True)

    while not (root / 'STOP').exists():
        requests = sorted(root.glob('*.request.npz'), key=lambda path: path.stat().st_mtime)
        if not requests:
            time.sleep(.1)
        for request in requests:
            process_request(request, processor, model, torch)


if __name__ == '__main__':
    main()
