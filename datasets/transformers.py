# coding=utf-8

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

__all__ = [
    'Compose',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomScaleCrop',
    'RandomScaleCropEx',
    'RandomGaussianBlur',
    'Normalize',
    'ToNumpy',
]


class Compose(object):
    """
    Composes several transforms together.

    Params:
        trans_seq (list of ``Transform`` objects): list of transforms to compose.

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, trans_seq):
        self.trans_seq = trans_seq

    def __call__(self, img, lbl):
        for t in self.trans_seq:
            img, lbl = t(img, lbl)
        return img, lbl


class RandomHorizontalFlip(object):
    """
    Horizontally flip the given image randomly with a given probability.

    The image must be a PIL image, in which case it is expected

    Params:
        p (float): probability of the image being flipped. Default value is 0.5

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, lbl):
        if random.random() > self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


class RandomVerticalFlip(object):
    """
    Vertically flip the given image randomly with a given probability.

    The image must be a PIL image, in which case it is expected

    Params:
        p (float): probability of the image being flipped. Default value is 0.5

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, lbl):
        if random.random() > self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
        return img, lbl


class RandomScaleCrop(object):
    """
    Randomly scale and crop the given image randomly with a given probability.

    The image must be a PIL image, in which case it is expected

    Params:
        crop_size (int): The crop size of image

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def __call__(self, img, lbl):
        short_size = random.randint(int(self.crop_size * 0.5), int(self.crop_size * 2.0))
        w, h = img.size

        oh = short_size
        ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)
        lbl = lbl.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size <= self.crop_size:
            pad_h = self.crop_size - oh if oh < self.crop_size else 0
            pad_w = self.crop_size - ow if ow < self.crop_size else 0

            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            lbl = ImageOps.expand(lbl, border=(0, 0, pad_w, pad_h), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        lbl = lbl.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, lbl


class RandomScaleCropEx(object):
    """
    Randomly scale and crop the given image randomly with a given probability.

    The image must be a PIL image, in which case it is expected

    Params:
        crop_size (int or 2-int list): The crop size of image
        image_scale_ratio (2-float list): The scale rate of crop size
        padding_crop_ratio (float): The padding ratio of crop size

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, crop_size, image_scale_ratio=(0.5, 2.0), padding_crop_ratio=0.25):
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, list) or isinstance(crop_size, tuple):
            self.crop_size = tuple(crop_size)
        else:
            raise AttributeError(
                'Expected type of `crop_size` is int or 2-int list, but input type: {}'.format(type(crop_size))
            )

        self.image_scale_ratio = image_scale_ratio
        self.padding_crop_ratio = padding_crop_ratio

    def __call__(self, img, lbl):
        w, h = img.size
        cw, ch = self.crop_size

        image_scale_ratio = np.random.uniform(*self.image_scale_ratio)

        new_w, new_h = int(w * image_scale_ratio), int(h * image_scale_ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        lbl = lbl.resize((new_w, new_h), Image.NEAREST)

        x = random.randint(int(-cw * self.padding_crop_ratio), int(new_w - (1 - self.padding_crop_ratio) * cw))
        y = random.randint(int(-ch * self.padding_crop_ratio), int(new_h - (1 - self.padding_crop_ratio) * ch))

        img = img.crop((x, y, x + cw, y + ch))
        lbl = lbl.crop((x, y, x + cw, y + ch))

        return img, lbl


class RandomGaussianBlur(object):
    """
    Blurs image with randomly chosen Gaussian blur.

    Params:
        p (float): probability of the image being gaussian blurred. Default value is 0.5

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur())
        return img, lbl


class Normalize(object):
    """
    Normalize a PIL image with mean and standard deviation.

    Params:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        div_std (bool): Whether the data will be divided by `std`. Default False

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self, mean, std, div_std=False):
        self.mean = mean
        self.std = std
        self.div_std = div_std

    def __call__(self, img, lbl):
        img = np.array(img).astype(np.float32) / 255.0
        img -= self.mean

        if self.div_std is True:
            img /= self.std

        return img, lbl


class ToNumpy(object):
    """
    Convert PIL image to Numpy

    Call:
        Note that the image and label must be PIL image.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img, lbl):
        return np.asarray(img), np.asarray(lbl)
