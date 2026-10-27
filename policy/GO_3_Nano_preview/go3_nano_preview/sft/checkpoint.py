import json
from pathlib import Path

from go3_nano_preview.sft.paths import publish_link

def save_checkpoint(model, output, step, checkpoint_link=None):
    folder = output / f'step_{step}'
    folder.mkdir(parents=True, exist_ok=False)
    model.model.vgen.save_pretrained(folder, safe_serialization=False, max_shard_size='100GB')
    model.model.ae.save_pretrained(folder / 'action_model', safe_serialization=False, max_shard_size='100GB')
    config = folder / 'config.json'
    data = json.loads(config.read_text())
    if 'vlm_model_id' in data:
        data['vlm_model_id'] = '${VLM_MODEL_PATH}'
    config.write_text(json.dumps(data, indent=2) + '\n')
    required = ('config.json', 'diffusion_pytorch_model.bin',
                'action_model/config.json', 'action_model/diffusion_pytorch_model.bin')
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f'Incomplete inference checkpoint: {folder / name}')
    if checkpoint_link:
        publish_link(Path(checkpoint_link), folder, checkpoint_output=output)
    return folder
