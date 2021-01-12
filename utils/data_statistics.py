# coding=utf-8

from tqdm import tqdm
import numpy as np
import cv2
import os


def statistics(data_root, image_size=(256, 256)):
    """
    Statistics the mean and variance of all pictures in the data set

    Params:
        data_root: str. All pictures in this folder will be counted
        image_size: 2-int list. The reshaped image size, you can adjust it appropriately
            according to your own data set and have little effect on the results

    Return:
        Returns the normalized mean and variance
    """
    img_h, img_w = image_size
    mean, std = [], []
    img_list = []

    img_path_list = os.listdir(data_root)

    for i, item in enumerate(tqdm(img_path_list)):
        img = cv2.imread(os.path.join(data_root, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    img = np.concatenate(img_list, axis=3)
    img = img.astype(np.float32) / 255.

    for i in range(3):
        pixels = img[:, :, i, :].ravel()
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))

    mean.reverse()
    std.reverse()

    return mean, std
