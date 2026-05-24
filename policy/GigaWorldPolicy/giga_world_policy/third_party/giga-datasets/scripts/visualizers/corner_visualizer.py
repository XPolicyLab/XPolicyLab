import numpy as np

from giga_datasets import ImageVisualizer
from .base_visualizer import BaseVisualizer, get_args


class CornerVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(CornerVisualizer, self).__init__(data_path, save_path)
        self.show_size = 540
        self.categories = []

    def draw_one(self, data_dict):
        assert len(data_dict['corners']) == len(data_dict['labels3d'])
        corners = []
        classes = []
        for i, name in enumerate(data_dict['labels3d']):
            if name not in self.categories:
                self.categories.append(name)
            corners.append(data_dict['corners'][i])
            classes.append(self.categories.index(name) + 1)
        if len(corners) > 0:
            corners = np.stack(corners)
            classes = np.array(classes, dtype=np.int32)
        image = data_dict['image']
        image = ImageVisualizer(image)
        image.draw_corners(corners, classes, show_ori=True, bottom_indexes=[2, 3, 6, 7], show_num=True)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


def main():
    args = get_args()
    visualizer = CornerVisualizer(args.data_path, args.save_path)
    visualizer.draw()


if __name__ == '__main__':
    main()
