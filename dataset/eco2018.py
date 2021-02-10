import cv2 as cv

import scipy.io as io
import numpy as np
import os

import sys
sys.path.append('.')

from dataset.data_util import pil_load_img
from dataset.dataload import RootDataset, RootInstance


def roots_to_polygons(annotation_mask) -> [RootInstance]:
    """
    Extracts roots as polygons from binary annotation mask.
    """

    # cv.findContours() needs a CV_8UC1 image
    annotation_mask = cv.cvtColor(annotation_mask, cv.COLOR_BGR2GRAY)

    # Retrieval mode = cv.RETR_EXTERNAL: find outer contours only,
    # no hierarchy established;
    # Contour approximation method = cv.CHAIN_APPROX_SIMPLE: do not
    # store *every* point on the contour, only important ones
    contours, _ = cv.findContours(annotation_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # list of contours, each is a Nx1x2 numpy array,
    # where N is number of points. Remove intermediate
    # dimension of length 1
    contours = [RootInstance(points=c.squeeze()) for c in contours]

    return contours


class Eco2018(RootDataset):
    """
    Iterable to be passed into PyTorch's data.DataLoader.

    This class loads the images and masks, and does all preprocessing, except standard operations
    like normalizing. These are handled by the baseclass.

    Eco2018 differs from Total-Text, because the input for TextSnake, like center lines, is already
    present as additional image masks.
    """
    def __init__(
            self,
            data_root='data/Eco2018',
            is_training=True,
            transform=True,
            normalize=True):

        # TODO mean, std
        super().__init__(transform, normalize, mean=(0, 0, 0), std=(0, 0, 0))
        self.data_root = data_root
        self.is_training = is_training

        self._annotation_names = ['roots', 'centerline', 'radius', 'sin', 'cos']

        self.image_root = os.path.join(data_root, 'images', 'training' if is_training else 'test')
        self.annotation_root = os.path.join(data_root, 'annotation', 'training' if is_training else 'test')

        self.image_list = os.listdir(self.image_root)
        # One list per image with names of root mask, centerline mask, etc.
        self.annotation_lists = {
            key: [
                img_name.replace('-', f'-{key}-') for img_name in self.image_list
            ] for key in self._annotation_names}

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        input_data = {'img': image}
        for annotation_name in self._annotation_names:
            annotation_id = self.annotation_lists[annotation_name][item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            input_data[annotation_name] = pil_load_img(annotation_path)

        polygons = roots_to_polygons(input_data['roots'])
        # TODO dafuq is train_mask ?!
        # image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta
        return self.get_training_data(input_data, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    # TODO
    #means = (0.485, 0.456, 0.406)
    #stds = (0.229, 0.224, 0.225)

    # transform = Augmentation(
    #     size=512, mean=means, std=stds
    # )

    trainset = Eco2018(
        data_root='data/Eco2018-Test',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        normalize=True
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    import matplotlib.pyplot as plt
    titles = ['image']
    titles.extend(trainset._annotation_names)
    for idx in range(0, len(trainset)):
        # img, tr_mask, tcl_mask, radius_map, sin_map, cos_map = trainset[idx]
        # print(idx, img.shape)
        maps = [map for map in trainset[idx]]
        fig, axs = plt.subplots(3, 2)
        for i, map in enumerate(maps):
            axs[int(i / 2)][(i % 2)].imshow(map)
            axs[int(i / 2)][(i % 2)].set_title(titles[i])
        plt.show()
