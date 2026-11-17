"""Bounded recording of allowed policy observations, never simulator truth."""
import json
from pathlib import Path
from PIL import Image
from .contracts import restrict_observation


def save_replan_observation(directory,token,env_idx,step,observation):
    if not token.isalnum() or env_idx<0 or step<1:
        raise ValueError('Invalid observation audit identity')
    observation=restrict_observation(observation)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    stem=f'{token}-e{env_idx}-s{step:04d}'
    paths=[directory/(stem+'-cam_head.png'),directory/(stem+'-robot.json')]
    if any(p.exists() for p in paths):raise FileExistsError(stem)
    Image.fromarray(observation['vision']['cam_head']['color']).save(paths[0])
    paths[1].write_text(json.dumps({k:v.tolist() for k,v in observation['state'].items()}))
    return stem
