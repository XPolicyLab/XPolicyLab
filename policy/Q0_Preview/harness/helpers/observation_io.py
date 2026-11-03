"""Agent-side access to immutable, attributed RGB observations; no simulator access."""
import hashlib
import json
from pathlib import Path
import numpy as np

_CURRENT = None
_WORK = Path('/work')
MAX_IMAGE_EDGE_PX = 1000


def configure(manifest_path):
    global _CURRENT
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    for item in manifest['files'].values():
        source = path.parent / item['name']
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Observation integrity failure: ' + item['name'])
    _CURRENT = path
    return manifest['observation_id']


def manifest(observation_id=None):
    if _CURRENT is None:
        raise RuntimeError('Observation workspace has not been configured')
    if observation_id is None:
        path = _CURRENT
    else:
        if not isinstance(observation_id, str) or not observation_id.startswith('obs_') or not observation_id[4:].isdigit():
            raise ValueError('Use an observation ID from the conversation, not a guessed filename')
        path = _CURRENT.parent.parent / observation_id / 'manifest.json'
    return path, json.loads(path.read_text())


def state(observation_id=None):
    """Full observation dict, including state and calibration; default to the latest."""
    path, m = manifest(observation_id)
    item = m['files']['state']
    source = path.parent / item['name']
    if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Observation integrity failure: ' + item['name'])
    return json.loads(source.read_text())


def image_path(camera='cam_head', observation_id=None):
    path, m = manifest(observation_id)
    if camera not in m['cameras']:
        raise ValueError('Unknown camera; available: ' + ', '.join(m['cameras']))
    item = m['files'][camera]
    target = path.parent / item['name']
    if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('Image integrity failure: ' + camera)
    return str(target)


def image(camera='cam_head', observation_id=None):
    """Independent writable uint8 RGB ndarray (H,W,3), in original camera pixels."""
    # Producer uses XPolicyLab's shared decoder on the exact JPEG sent to the API.
    # The sandbox needs only NumPy, not a second image decoder or color conversion.
    image_path(camera, observation_id)
    path, m = manifest(observation_id)
    item = m['files'][camera+'_rgb']
    target = path.parent / item['name']
    if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('RGB array integrity failure: '+camera)
    return np.load(target, allow_pickle=False)  # np.load already owns an independent writable array.


def save_crop(camera, box, path, observation_id=None):
    """Save a native-resolution crop and return its path and source-pixel mapping.

    box=(x0,y0,x1,y1) contains integer original-image coordinates; x1/y1 are exclusive.
    Reject invalid/out-of-bounds boxes rather than clipping, padding or resizing.
    path must be an absolute /work PNG/JPEG path outside the observation directory.
    Metadata is also saved as <path>.crop.json; use return_images to display the image.
    """
    from PIL import Image
    import io
    bounds = np.asarray(box)
    if bounds.shape != (4,) or bounds.dtype.kind not in 'iu':
        raise ValueError('box must be four integer pixels (x0,y0,x1,y1); x1/y1 are exclusive')
    x0, y0, x1, y1 = [int(v) for v in bounds]
    pixels = image(camera, observation_id)
    height, width = pixels.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'box must be nonempty and inside the {width}x{height} original image')
    if max(x1-x0, y1-y0) > MAX_IMAGE_EDGE_PX:
        raise ValueError(f'crop must have a longest edge of {MAX_IMAGE_EDGE_PX} px or less; choose a smaller box')
    if not isinstance(path, str) or not path.startswith('/work/'):
        raise ValueError('crop path must be an absolute /work PNG/JPEG path')
    destination = (_WORK/path.removeprefix('/work/')).resolve()
    work = _WORK.resolve()
    observations = (work/'observations').resolve()
    metadata_path = destination.with_name(destination.name+'.crop.json')
    if (not destination.is_relative_to(work) or destination.is_relative_to(observations)
            or destination.suffix.lower() not in ('.png', '.jpg', '.jpeg')
            or metadata_path.is_symlink()):
        raise ValueError('crop path must stay under /work, outside observations, with a PNG/JPEG extension')
    buffer = io.BytesIO()
    fmt = 'PNG' if destination.suffix.lower() == '.png' else 'JPEG'
    Image.fromarray(pixels[y0:y1, x0:x1]).save(buffer, format=fmt, **({'quality':95} if fmt=='JPEG' else {}))
    data = buffer.getvalue()
    if len(data) > 8_000_000:
        raise ValueError('crop exceeds 8 MB')
    _, source = manifest(observation_id)
    metadata = {'path': path, 'source_observation_id': source['observation_id'],
                'camera': camera, 'source_image_sha256': source['files'][camera]['sha256'],
                'bbox_xyxy': [x0,y0,x1,y1], 'source_size_wh': [width,height],
                'size_wh': [x1-x0,y1-y0], 'original_pixel_offset_xy': [x0,y0],
                'scale_xy': [1,1], 'sha256': hashlib.sha256(data).hexdigest()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
