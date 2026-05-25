import os

import tyro
from decord import VideoReader
from tqdm import tqdm

from giga_datasets import Dataset, FileWriter, LmdbWriter, PklWriter, load_dataset, utils


def main(video_dir: str, save_dir: str, pack_lmdb: bool = False, only_path: bool = False):
    video_paths = utils.list_dir(video_dir, recursive=True, exts=['.mp4'])
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    if pack_lmdb:
        video_writer = LmdbWriter(os.path.join(save_dir, 'videos'))
    else:
        video_writer = FileWriter(os.path.join(save_dir, 'videos'))
    # Iterate over each annotation file and process
    for idx in tqdm(range(len(video_paths))):
        video = VideoReader(video_paths[idx])
        # save meta information
        label_dict = {
            'data_index': idx,
            'video_length': len(video),
            'video_height': video[0].shape[0],
            'video_width': video[0].shape[1],
            'video_fps': video.get_avg_fps(),
            'video_duration': len(video) / video.get_avg_fps(),
        }
        # Write the annotation dictionary
        label_writer.write_dict(label_dict)
        # Write the corresponding image file with data index
        if only_path:
            video_writer.write_video_path(idx, video_paths[idx])
        else:
            video_writer.write_video(idx, video_paths[idx])
    # Save configuration files for both writers and Close the writers to finalize files
    label_writer.write_config()
    video_writer.write_config()
    label_writer.close()
    video_writer.close()
    # Load the packed datasets and Combine label and video datasets into a single Dataset and save
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
    dataset = Dataset([label_dataset, video_dataset])
    dataset.save(save_dir)
    # Load and verify the packed dataset
    packed_dataset = load_dataset(save_dir)
    data_dict = packed_dataset[0]
    print('Packed dataset size:', len(packed_dataset))
    print('First item in packed dataset:', data_dict)


if __name__ == '__main__':
    tyro.cli(main)
