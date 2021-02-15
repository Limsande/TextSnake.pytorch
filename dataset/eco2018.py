import sys

import cv2 as cv
import numpy as np

sys.path.append('.')

from dataset.data_util import pil_load_img
from dataset.dataload import RootDataset, RootInstance


def roots_to_polygons(annotation_mask) -> [RootInstance]:
    """
    Extracts roots as polygons from binary annotation mask.
    """

    # With given options, cv.findContours() only supports uint8 input.
    if annotation_mask.dtype is not np.uint8:
        annotation_mask = annotation_mask * 255
        annotation_mask = annotation_mask.astype(np.uint8)

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

    This class loads the images and masks, and extracts root polygons from the binary annotation
    mask (we must feed these into the net). Additional stuff like image type conversions etc.
    is handled by the baseclass.

    Eco2018 differs from Total-Text, because the input for TextSnake, like center lines, is already
    present as additional image masks.
    """
    def __init__(
            self,
            data_root='data/Eco2018',
            is_training=True,
            transformations=None):

        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.transformations = transformations

        self._annotation_names = ['roots', 'centerline', 'radius', 'sin', 'cos']

        self.image_root = os.path.join(data_root, 'images', 'training' if is_training else 'validation')
        self.annotation_root = os.path.join(data_root, 'annotation', 'training' if is_training else 'validation')

        self.image_list = os.listdir(self.image_root)
        # One list per image with names of root mask, center line mask, etc.
        self.annotation_lists = {
            key: [
                img_name.replace('-', f'-{key}-') for img_name in self.image_list
            ] for key in self._annotation_names}

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotations and build a dict with them
        img_and_masks = {'img': image}
        for annotation_name in self._annotation_names:
            annotation_id = self.annotation_lists[annotation_name][item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            img_and_masks[annotation_name] = pil_load_img(annotation_path)

        # Apply augmentations to image and masks
        if self.transformations:
            img_and_masks = self.transformations(img_and_masks)

        polygons = roots_to_polygons(img_and_masks['roots'])

        # image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta
        return self.get_training_data(img_and_masks, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    from util.augmentation import RootAugmentation

    # TODO
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transformations = RootAugmentation(mean=means, std=stds)

    trainset = Eco2018(
        data_root='data/Eco2018-Test',
        is_training=True,
        transformations=transformations
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    import matplotlib.pyplot as plt
    titles = ['image', 'train_mask']
    titles.extend(trainset._annotation_names)
    for idx in range(0, len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[idx]
        # print(idx, img.shape)
        maps = [map for map in trainset[idx]]
        maps[0] = np.moveaxis(maps[0], 0, 2)
        fig, axs = plt.subplots(4, 2)
        for i, map in enumerate(maps[:-1]):
            axs[int(i / 2)][(i % 2)].imshow(map)
            axs[int(i / 2)][(i % 2)].set_title(titles[i])
        plt.show()

        print('Image:', idx, img.shape)
        print('Train mask:', train_mask.shape, train_mask.dtype, train_mask.min(), train_mask.max())
        print('TR mask:', tr_mask.shape, tr_mask.dtype, tr_mask.min(), tr_mask.max())
        print('TCL mask:', tcl_mask.shape, tcl_mask.dtype, tcl_mask.min(), tcl_mask.max())
        print('Radius map:', radius_map.shape, radius_map.dtype, radius_map.min(), radius_map.max())
        print('Sin map:', sin_map.shape, sin_map.dtype, sin_map.min(), sin_map.max())
        print('Cos map:', cos_map.shape, cos_map.dtype, cos_map.min(), cos_map.max())
