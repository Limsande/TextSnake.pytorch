import copy

import cv2
import numpy as np
import scipy.io as io
import torch.utils.data as data
from PIL import Image
from skimage.draw import polygon as drawpoly

from util.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4: # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(cfg.input_size, cfg.input_size))
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius, \
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.3, shrink=1):

        # TODO: shrink 1/2 * radius at two line end
        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(n_disk=cfg.n_disk)
                self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map, cos_map)
        tr_mask, train_mask = self.make_text_region(image, polygons)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)

        for i, polygon in enumerate(polygons):
            pts = polygon.points
            points[i, :pts.shape[0]] = polygon.points
            length[i] = pts.shape[0]

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'Height': H,
            'Width': W
        }
        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()


class RootInstance(object):
    """
    Represents a single root as a polygon.
    """

    def __init__(self, points: np.array):
        """
        :param points: Nx2 numpy array, where N is number of points
            in this root (i.e. polygon)
        """
        remove_points = []

        # Try to reduce number of points in this polygon without
        # loosing to much information.
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

        self.length = len(self.points)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class RootDataset(data.Dataset):
    """
    Only  implements some basic preparations to put images into the neural net.
    Any subclass has to take care of loading images and calculating all the input
    for TextSnake, like center line, root polygons etc. as well as applying
    augmentation.
    """

    def __init__(self):
        super().__init__()

    def get_training_data(self, img_and_masks, polygons, image_id, image_path):
        """
        Prepares meta data the network needs for training.

        :param img_and_masks: dictionary with input image and one mask per TextSnake input
        :param polygons: list of RootInstance objects defining the roots in this img_and_masks['img']
        """

        img_height, img_width, _ = img_and_masks['img'].shape

        # train_mask = self.make_text_region(image, polygons)
        # Extracted from make_text_region. No idea what this is for
        train_mask = np.ones(img_and_masks['img'].shape[:2], np.uint8)

        # to pytorch channel sequence
        img_and_masks['img'] = img_and_masks['img'].transpose(2, 0, 1)

        # max_annotation = max #polygons per image
        # max_points = max #points per polygons
        max_points = max([p.length for p in polygons])
        all_possible_points_for_each_possible_polygon = np.zeros((cfg.max_annotation, max_points, 2))
        n_points_per_polygon = np.zeros(cfg.max_annotation, dtype=int)
        for i, polygon in enumerate(polygons):
            # polygon.length = #points in this polygon
            all_possible_points_for_each_possible_polygon[i, :polygon.length] = polygon.points
            n_points_per_polygon[i] = polygon.length

        # All input images are uint8. Do some type conversions
        # to match expected model input:
        #   Train mask: uint8, 0 or 1
        #   Root mask: uint8, 0 or 1
        #   Center line mask: uint8, 0 or 1
        #   Radius map: float32
        #   Sin map: float32, -1.0 to 1.0
        #   Cos map: float32, -1.0 to 1.0
        for mask in [img_and_masks['roots'], img_and_masks['centerlines']]:
            if mask.max() > 1:
                # Store result of array division in int array
                # without type conversions.
                # See https://github.com/numpy/numpy/issues/17221
                np.divide(mask, 255, out=mask, casting='unsafe')

        img_and_masks['radii'] = img_and_masks['radii'].astype(np.float32)

        # Map [0, 255] to [-1, 1]
        for key in ['sin', 'cos']:
            map = img_and_masks[key].astype(np.float32)
            map -= 255 / 2  # [0, 255] -> [-127.5, 127.5]
            map /= 255 / 2  # [-127.5, 127.5] -> [-1, 1]
            img_and_masks[key] = map

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': all_possible_points_for_each_possible_polygon,
            'n_annotation': n_points_per_polygon,
            'Height': img_height,
            'Width': img_width
        }

        #return img_and_masks, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta
        return (img_and_masks['img'],
                train_mask,
                img_and_masks['roots'],
                img_and_masks['centerlines'],
                img_and_masks['radii'],
                img_and_masks['sin'],
                img_and_masks['cos'],
                meta)

    def get_test_data(self, image, image_id, image_path):
        # TODO

        H, W, _ = image.shape

        if self.transform:
            # TODO mean and stds
            # image, polygons = self.transform(image)
            pass

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
