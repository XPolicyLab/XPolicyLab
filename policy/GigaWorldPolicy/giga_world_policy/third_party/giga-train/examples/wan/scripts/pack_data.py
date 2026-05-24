import os
from glob import glob
from typing import List

import tyro
from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from tqdm import tqdm


def pack_data(video_dir: str, save_dir: str) -> None:
    """Pack raw videos and prompts into a dataset folder.

    Expects each video file `name.mp4` to have a paired `name.txt` containing
    a single-line prompt.

    Args:
        video_dir (str): Input directory containing `*.mp4` and `*.txt` pairs.
        save_dir (str): Output directory to write the dataset files.
    """
    video_paths: List[str] = glob(os.path.join(video_dir, '*.mp4'))
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    video_writer = FileWriter(os.path.join(save_dir, 'videos'))
    for idx in tqdm(range(len(video_paths))):
        anno_file = video_paths[idx].replace('.mp4', '.txt')
        prompt = open(anno_file, 'r').read().strip()
        label_dict = dict(data_index=idx, prompt=prompt)
        label_writer.write_dict(label_dict)
        video_writer.write_video(idx, video_paths[idx])
    label_writer.write_config()
    video_writer.write_config()
    label_writer.close()
    video_writer.close()
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
    dataset = Dataset([label_dataset, video_dataset])
    dataset.save(save_dir)


if __name__ == '__main__':
    tyro.cli(pack_data)
