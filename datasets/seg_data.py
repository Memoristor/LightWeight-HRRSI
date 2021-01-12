# coding=utf-8

from utils.data_convert import str_to_arr
from torch.utils.data import Dataset
from datasets import transformers
from PIL import Image
import numpy as np
import glob
import cv2
import os

__all__ = ['SegDataset']


class SegDataset(Dataset):
    """
    Basic dataset for segmentation.

    Params:
        root_path: str. The root path to the data folder, which contains `image/` and `label/`
        image_size: tuple. The size of output image, which format is (H, W)
        phase: str. Indicates that the dataset is used for {`train`, `test`, `valid`}
        class_rgb: dict. The classes' RGB value, e.g {'IS': [255, 255, 255], 'BD': [0, 0, 255]}
        class_wo_encode: list. The ignored classes which will be encoded as 255, e.g ['IS']
        class_wo_score: list. The classes which will not be counted when calculate scores
        mean: list (default (0, 0, 0)). The mean value of normed RGB
        std: list (default (1, 1, 1)). The std value of normed RGB
        div_std: bool. (default True). Whether the normed data will be divided by `std`
        chw_format: bool (default True). If True, the output data's format is CxHxW, otherwise HxWxC
        edge_width: int (default 5). The edge with for edge map
        img_ext: str (default 'png'). File suffix for original image file
        lbl_ext: str (default 'png'). File suffix for original label file
        sort_key: callable (default None). Sort the items of dataset
   """

    def __init__(self, root_path: str, image_size: int or tuple, phase: str, class_rgb: dict, class_wo_encode: list,
                 class_wo_score: list, mean=(0, 0, 0), std=(1, 1, 1), div_std=True,
                 chw_format=True, edge_width=5, img_ext='png', lbl_ext='png', sort_key=None):

        self.root_path = root_path
        self.image_size = image_size
        self.phase = phase
        self.class_rgb = class_rgb
        self.class_wo_encode = class_wo_encode
        self.class_wo_score = class_wo_score

        self.mean = mean
        self.std = std
        self.div_std = div_std
        self.chw_format = chw_format
        self.edge_width = edge_width
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext

        # Get number of all classes
        self.num_class = len(self.class_rgb)

        # Get classes will be encoded
        if self.class_wo_encode is None:
            self.class_wo_encode = []

        self.clazz_encode = []
        for cls in sorted(self.class_rgb.keys()):
            if cls not in self.class_wo_encode:
                self.clazz_encode.append(cls)

        # Init augmentation sequence
        if phase == 'train':
            self.aug_seq = transformers.Compose([
                transformers.RandomHorizontalFlip(),
                transformers.RandomVerticalFlip(),
                transformers.RandomScaleCrop(self.image_size),
                transformers.RandomGaussianBlur(),
                transformers.Normalize(self.mean, self.std, self.div_std),
                transformers.ToNumpy(),
            ])
        else:  # 'test' or 'valid'
            self.aug_seq = transformers.Compose([
                transformers.Normalize(self.mean, self.std, self.div_std),
                transformers.ToNumpy(),
            ])

        # Find images from ${root_path}/image and sort it by ${sort_key}
        self.image_items = [os.path.basename(x)[0:-4] for x in
                            glob.glob(os.path.join(self.root_path, 'image/*.' + self.img_ext))]
        self.image_items.sort(key=sort_key)

        print('[{}] Found {} images from `{}`'.format(self.__class__.__name__, len(self), self.root_path))

    def encode_lbl(self, label):
        """
        The image is encoded from RGB format to the format required by the training model

        Params:
            label: 3-D numpy array. RGB images to be encoded. Note that the ignored
            category is encoded as 255

        Return:
            return the encoded label
        """
        lbl = np.ones(label.shape[0:2], dtype=np.int16) * 255
        for i, cls in enumerate(self.clazz_encode):
            lbl[np.where(np.all(label == self.class_rgb[cls], axis=-1))[:2]] = i
        return lbl

    def decode_lbl(self, label):
        """
        The image is decoded from the format required by the training model into RGB format

        Params:
            label: 2-D numpy array. Label to de decoded. Note that the ignored
            category will be decoded as 255

        Return:
            return the decoded label
        """
        if len(self.class_wo_encode) == 0:
            lbl = np.zeros((*label.shape[0:2], 3), dtype=np.uint8)
        else:
            lbl = np.ones((*label.shape[0:2], 3), dtype=np.uint8) * self.class_rgb[self.class_wo_encode[0]]

        for i, cls in enumerate(self.clazz_encode):
            lbl[label == i, :] = self.class_rgb[cls]

        if self.chw_format:
            lbl = lbl.transpose([2, 0, 1])

        return lbl

    def get_edge_map(self, label):
        """
        Obtain the edge map of the real segmentation label, pixels which are close to
        edge will be defined toward `1`, otherwise toward `0`

        Params:
            label: 3-D numpy array. RGB images to be encoded. Note that the ignored
            category will be defined as `255`

        Return:
            return the edge map
        """
        # Get label's edge by `Laplacian`
        gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

        edge_x = cv2.Laplacian(gray, cv2.CV_8U, 1, 3)
        edge_y = cv2.Laplacian(gray, cv2.CV_8U, 0, 3)
        edge = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)
        edge = cv2.GaussianBlur(edge, ksize=(self.edge_width, self.edge_width), sigmaX=0)

        # Norm edge to the range of [0, 1]
        edge_max, edge_min = edge.max(), edge.min()
        if edge_max - edge_min > 0:
            edge = (edge - edge_min) / (edge_max - edge_min)
        else:  # edge_max == edge_min
            if edge_max > 0:
                edge = edge / edge_max
            else:
                edge = edge.astype(np.float32)
        edge[edge > 0] = 1

        # Class without encoding
        for i, cls in enumerate(self.class_wo_encode):
            edge[np.where(np.all(label == self.class_rgb[cls], axis=-1))[:2]] = 255.0

        return edge

    def get_class_proportion(self):
        """
        Count and calculate the probability (proportion) of all categories, it should be
        noted that the ignored categories are not calculated

        Return:
            Returns the proportion of each annotation pixel in all labels
        """
        print('[{}] Get class proportion, please wait...'.format(self.__class__.__name__))

        counts = dict((cls, 0) for cls in self.clazz_encode)
        prob = dict((cls, 0) for cls in self.clazz_encode)

        for item in self.image_items:
            lbl_path = os.path.join(self.root_path, 'label', item + '.png')
            lbl_mask = np.asarray(Image.open(lbl_path)).reshape([-1, 3]).tolist()

            for cls in self.clazz_encode:
                counts[cls] += lbl_mask.count(self.class_rgb[cls])

        values = np.array([counts[k] for k in counts.keys()])
        value_sum = np.sum(values)
        for k in counts.keys():
            prob[k] = counts[k] / value_sum

        return np.array([prob[cls] for cls in self.clazz_encode])

    def get_class_weight(self, upper_bound=1.0):
        """
        Get the weight of each classes for training or testing losses

        Note that the range of output weight of each class is [1, 2]
        """
        weight = self.get_class_proportion()
        weight = ((weight != 0) * upper_bound * (1 - weight)) + 1
        return weight

    def __getitem__(self, index):
        """
        Get item by index
        """
        # 1, Open image and label
        img = Image.open(os.path.join(self.root_path, 'image', self.image_items[index]) + '.' + self.img_ext)
        lbl = Image.open(os.path.join(self.root_path, 'label', self.image_items[index]) + '.' + self.lbl_ext)

        assert img.size[0:2] == lbl.size[0:2]
        hw = img.size[0:2]

        # 2, Data argumentation
        img, lbl = self.aug_seq(img, lbl)

        # 3, Get binary edge map
        edg = self.get_edge_map(lbl)

        # 4, Encode label
        lbl = self.encode_lbl(lbl)

        # 5, Channel transpose
        if self.chw_format:
            img = img.transpose([2, 0, 1])

        return {
            'img': np.array(img, dtype=np.float32),
            'lbl': np.array(lbl, dtype=np.float32),
            'edg': np.array(edg, dtype=np.float32),
            'fnm': str_to_arr(os.path.basename(self.image_items[index])),
            'hw': np.array(hw, dtype=np.int32),
        }

    def __len__(self):
        return len(self.image_items)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils.img_visual import VisualByPlt
    from utils.data_convert import arr_to_str
    from utils.data_statistics import statistics

    ############################################################

    root = "/home/dandri/data/vaismall/train"

    vaihingen = {
        'class_rgb': {
            'IS': [255, 255, 255],  # Impervious surface
            'BD': [0, 0, 255],  # Buildings
            'LV': [0, 255, 255],  # Low vegetation
            'TR': [0, 255, 0],  # Tree
            'CR': [255, 255, 0],  # Car
            'BG': [255, 0, 0],  # Clutter/background
            'IG': [0, 0, 0],  # Ignore
        },
        'class_wo_encode': ['IG'],
        'class_wo_score': ['BG'],
        'mean': [0.46956375, 0.32316217, 0.3183202],
        'std': [0.22002102, 0.15978466, 0.15127575],
    }

    seg = SegDataset(
        root_path=root,
        image_size=512,
        phase='train',
        chw_format=False,
        **vaihingen,
    )

    seg_loader = DataLoader(dataset=seg, batch_size=2, shuffle=True)

    ############################################################

    # print('=' * 60)
    # # prob, count = seg.get_class_proportion()
    # # print('prob: {}'.format(prob.values()))
    # # print('count: {}'.format(count.values()))
    # weight = seg.get_class_weight(upper_bound=10.0)
    # print('weight: {}'.format(weight))

    ############################################################

    print('=' * 60)
    seg_dict = next(iter(seg_loader))

    print(seg_dict.keys())
    for k in seg_dict.keys():
        print("{}: {}".format(k, seg_dict[k].shape))

    ############################################################

    print('=' * 60)
    img = seg_dict['img']
    lbl = seg_dict['lbl']
    edg = seg_dict['edg']
    fnm = seg_dict['fnm']
    hw = seg_dict['hw']

    lbl[lbl == 255] = len(seg.clazz_encode)
    lbl /= len(seg.clazz_encode)
    print(lbl.min(), lbl.max())

    edg[edg == 255] = 0

    print('=' * 60)
    if seg.chw_format:
        img = img.permute((0, 2, 3, 1))

    for i in range(img.shape[0]):
        VisualByPlt(img[i], title='img: {}'.format(i + 1)).show()
        VisualByPlt(lbl[i], title='lbl: {}'.format(i + 1)).show()
        VisualByPlt(edg[i], title='edg: {}'.format(i + 1)).show()

    for i in range(fnm.shape[0]):
        print(arr_to_str(fnm[i]))

    ############################################################

    # mean, std = statistics(os.path.join(root, 'image'), image_size=(512, 512))
    # print('mean: {}, std: {}'.format(mean, std))

    ############################################################
