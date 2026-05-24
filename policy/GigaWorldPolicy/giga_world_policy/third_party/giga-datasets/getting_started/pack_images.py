import json
import os

import tyro
from tqdm import tqdm

from giga_datasets import Dataset, LmdbWriter, PklWriter, load_dataset, utils


def main(image_dir: str = './raw_data', save_dir: str = './giga_data'):
    image_paths = utils.list_dir(image_dir, recursive=True, exts=['.png', '.jpg', '.jpeg'])
    # Create writers for labels (pickle format) and images (lmdb format)
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    image_writer = LmdbWriter(os.path.join(save_dir, 'images'))
    # Iterate over each annotation file and process
    for idx in tqdm(range(len(image_paths))):
        label_path = image_paths[idx].replace('.png', '.json')
        label_dict = json.load(open(label_path))
        label_dict['data_index'] = idx
        # Write the annotation dictionary
        label_writer.write_dict(label_dict)
        # Write the corresponding image file with data index
        image_writer.write_image(idx, image_paths[idx])
    # Save configuration files for both writers and Close the writers to finalize files
    label_writer.write_config()
    image_writer.write_config()
    label_writer.close()
    image_writer.close()
    # Load the packed datasets and Combine label and image datasets into a single Dataset and save
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    image_dataset = load_dataset(os.path.join(save_dir, 'images'))
    dataset = Dataset([label_dataset, image_dataset])
    dataset.save(save_dir)
    # Load and verify the packed dataset
    packed_dataset = load_dataset(save_dir)
    data_dict = packed_dataset[0]
    print('Packed dataset size:', len(packed_dataset))
    print('First item in packed dataset:', data_dict)


if __name__ == '__main__':
    tyro.cli(main)
