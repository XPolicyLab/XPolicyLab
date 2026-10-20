import argparse
import json
import os
from pathlib import Path
import sys

import torch
import yaml

from ..backbone import _configure_ieee_float32_matmul
from .data import RoboDojoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    _configure_ieee_float32_matmul()
    print(sys.executable, torch.__version__, torch.version.cuda,
          os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'], torch.get_float32_matmul_precision(),
          torch.backends.cuda.matmul.allow_tf32, flush=True)
    if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 11):
        raise RuntimeError('PyTorch 2.11 or newer is required.')
    inference = yaml.safe_load((args.checkpoint / 'config.yaml').read_text())['inference']
    stats = json.loads((args.checkpoint / 'dataset_stats.json').read_text())
    data = RoboDojoDataset(args.dataset, stats, inference, seed=42, training=False, color={})
    for episode in data.episodes:
        for path in (data.episode_paths(episode['episode_index'])[0], *data.episode_paths(episode['episode_index'])[1]):
            if not path.is_file():
                raise FileNotFoundError(path)
    indices = sorted({0, data.episodes[0]['length'] - 1, len(data) - 1})
    for index in indices:
        sample = data[index]
        if not all(torch.isfinite(sample[key]).all() for key in ('video', 'action', 'proprio')):
            raise ValueError(f'Non-finite sample at {index}.')
        print(json.dumps(dict(index=index, video=list(sample['video'].shape), action=list(sample['action'].shape),
                              padded_actions=int(sample['action_is_pad'].sum()))))
    print(json.dumps(dict(episodes=len(data.episodes), frames=len(data),
                          verified='file inventory and selected decoded samples, not every frame')))


if __name__ == '__main__':
    main()
