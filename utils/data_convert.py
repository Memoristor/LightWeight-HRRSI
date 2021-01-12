# coding=utf-8

import io
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

__all__ = [
    'line_to_img',
    'str_to_arr',
    'arr_to_str',
]


def line_to_img(line, title=None, y_min=None, y_max=None, x_label=None, y_label=None, figure_size=None):
    """
    Convert 1-D or 2-D numpy array to RGB image data
    """
    plt.figure(figsize=figure_size)

    if len(line.shape) == 1:
        plt.plot(line)
    elif len(line.shape) == 2:
        for i in range(line.shape[0]):
            plt.plot(line[i])
            # plt.hold()
    else:
        raise AttributeError('Expected 1-D or 2-D numpy array, but given {}-D.'.format(len(line.shape)))

    if title:
        plt.title(title)
    if y_min and y_max:
        plt.ylim(y_min, y_max)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')
    img = np.asarray(img)
    buf.close()

    return img


def str_to_arr(string, max_len=128):
    """
    Convert string to numpy array (ASCII)
    """
    arr = np.zeros(max_len, dtype=np.uint8)
    for i, s in enumerate(string):
        if i < max_len:
            arr[i] = ord(s)
        else:
            break
    return arr


def arr_to_str(arr, max_len=128):
    """
    Convert numpy array (ASCII) to str
    """
    string = ''
    for i, a in enumerate(arr):
        if i < max_len and a != 0:
            string = string + chr(a)
        else:
            break
    return string
