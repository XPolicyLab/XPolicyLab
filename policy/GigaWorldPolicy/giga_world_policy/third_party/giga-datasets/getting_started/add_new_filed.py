import os

import cv2
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

from giga_datasets import FileWriter, load_dataset


def get_canny_image(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    canny = cv2.Canny(image, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    return canny_image


def main(data_dir: str = './giga_data'):
    file_writer = FileWriter(os.path.join(data_dir, 'canny'))
    dataset = load_dataset(data_dir)
    for idx in tqdm(range(len(dataset))):
        data_dict = dataset[idx]
        image = data_dict['image']
        canny_image = get_canny_image(image)
        data_index = data_dict['data_index']
        file_writer.write_image(data_index, canny_image)
    file_writer.write_config(data_name='canny')
    file_writer.close()
    # Load the canny datasets and Combine label/image/canny datasets into a single Dataset and save
    canny_dataset = load_dataset(os.path.join(data_dir, 'canny'))
    dataset.datasets.append(canny_dataset)
    dataset.save(data_dir)
    # Load new dataset
    new_dataset = load_dataset(data_dir)
    data_dict = new_dataset[0]
    print('New dataset size:', len(new_dataset))
    print('First item in new dataset:', data_dict)


if __name__ == '__main__':
    tyro.cli(main)
