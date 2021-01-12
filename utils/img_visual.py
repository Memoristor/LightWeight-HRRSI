# coding=utf-8

# import matplotlib
# matplotlib.use('Agg')

from matplotlib import pyplot as plt, patches
from PIL import Image
import numpy as np
import cv2

__all__ = ['VisualByPlt', 'VisualByCV2']


class VisualByPlt(object):
    """
    Use matplotlib.pyplot to draw boxes, dots or mask on a image.
    Allow users to define the labels and colors of boxes and the radius and colors of dots.

    Params:
        image_data: 2-D or 3-D numpy array. The original background image to be drawn
        kwargs: dict. The key params, and the following params are supported:
                >> title: str. Title of the whole image.
                >> title_size: int. The size of the title. Default value is 20
                >> boxes: 2-D numpy array. 2-D coordinates preserved in the format of (y1, x1, y2, x2) for each row.
                >> boxes_label: str or 1-D list. The labels of each boxes.
                >> boxes_color: str or 1-D list. The colors of each boxes.
                >> dots: 2-D numpy array. A 2-D numpy array in the form of (y1, x1,..., yn, xn) for each row.
                >> dots_radius. int or 1-D list. The radius of each dots.
                >> dots_color. str or 1-D list. The color of each dots.
                >> mask. 2-D numpy array. The mask of segmentation objects or classes, note that the mask should
                        be the P mode of PIL.Image. The size of mask should be the same with image_data

    Return:
        plt(matplotlib.pyplot)
    """

    def __init__(self, image_data, save_path=None, **kwargs):
        self.image_data = image_data
        self.save_path = save_path
        self.kwargs = kwargs

        self.draw()

    def draw(self):
        # plt.style.use('seaborn-white')
        plt.figure()

        # 1, Set title
        if 'title' in self.kwargs.keys():
            # plt.title(self.kwargs['title'],
            #           {'family': 'Times New Roman', 'weight': 'normal',
            #            'size': self.kwargs['title_size'] if 'title_size' in self.kwargs.keys() else 20})
            plt.title(self.kwargs['title'], {
                'weight': 'normal',
                'size': self.kwargs['title_size'] if 'title_size' in self.kwargs.keys() else 20
            })

        # 2, Draw boxes
        if 'boxes' in self.kwargs.keys():

            # 2.1 Init boxes' parameters.
            boxes = np.array(self.kwargs['boxes'], ndmin=2, dtype=np.int)

            boxes_color = np.tile('b', [boxes.shape[0]])
            if 'boxes_color' in self.kwargs.keys():
                boxes_color = self.kwargs['boxes_color']
                if isinstance(boxes_color, str):
                    boxes_color = np.tile(boxes_color, [boxes.shape[0]])

            boxes_label = np.tile('', [boxes.shape[0]])
            if 'boxes_label' in self.kwargs.keys():
                boxes_label = self.kwargs['boxes_label']
                if isinstance(boxes_label, str):
                    boxes_label = np.tile(boxes_label, [boxes.shape[0]])

            # 2.2 Draw every boxes.
            for i in range(boxes.shape[0]):

                y1, x1, y2, x2 = boxes[i]
                rec = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", edgecolor=boxes_color[i])
                plt.gca().add_patch(rec)

                if boxes_label[i]:
                    plt.text(x1 + 5, y1 - 8, boxes_label[i], style='italic',
                             bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})

        # 3, Draw dots
        if 'dots' in self.kwargs.keys():

            # 3.1 Init dots' parameters
            dots = np.array(self.kwargs['dots'], ndmin=2, dtype=np.int)

            dots_color = np.tile('r', [dots.shape[0]])
            if 'dots_color' in self.kwargs.keys():
                dots_color = self.kwargs['dots_color']
                if isinstance(dots_color, str):
                    dots_color = np.tile(dots_color, [dots.shape[0]])

            dots_radius = np.tile(2, [dots.shape[0]])
            if 'dots_radius' in self.kwargs.keys():
                dots_radius = self.kwargs['dots_radius']
                if isinstance(dots_radius, int):
                    dots_radius = np.tile(dots_radius, [dots.shape[0]])

            # 3.2 Draw each dots.
            for i in range(dots.shape[0]):
                for j in range(dots.shape[1] // 2):
                    circle = patches.CirclePolygon((dots[i, j * 2 + 1], dots[i, j * 2]),
                                                   color=dots_color[i], radius=dots_radius[i])
                    plt.gca().add_patch(circle)

        # 4, Draw mask
        if 'mask' in self.kwargs.keys():
            img = Image.fromarray(self.image_data)
            msk = Image.fromarray(self.kwargs['mask'])
            bkg = Image.new('L', img.size, 128)
            self.image_data = np.array(Image.composite(msk, img, bkg))

        # Show image figure.
        plt.imshow(self.image_data)
        plt.tick_params(labelsize=15)

        # Save image
        if isinstance(self.save_path, str):
            plt.savefig(self.save_path)

    def show(self):
        plt.show()


class VisualByCV2(object):
    """
    Use CV2 to draw boxes, dots, polygons or mask on a image.
    Allow users to define the labels and colors of boxes and the radius and colors of dots.

    Params:
        image_data: 2-D or 3-D numpy array. The original background image to be drawn
        kwargs: dict. The key params, and the following params are supported:
                >> boxes: 2-D numpy array. 2-D coordinates preserved in the format of (y1, x1, y2, x2) for each row.
                >> boxes_label: str or 1-D list. The labels of each boxes.
                >> boxes_color: str or 1-D list. The colors of each boxes.
                >> dots: 2-D numpy array. A 2-D numpy array in the form of (y1, x1,..., yn, xn) for each row.
                >> dots_radius. int or 1-D list. The radius of each dots.
                >> dots_color. str or 1-D list. The color of each dots.
                >> mask. 2-D numpy array. The mask of segmentation objects or classes, note that the mask should
                        be the P mode of PIL.Image. The size of mask should be the same with image_data

    Return:
        None
    """

    def __init__(self, image_data, save_path=None, **kwargs):
        self.image_data = image_data
        self.save_path = save_path
        self.kwargs = kwargs

        self.draw()

    def draw(self):
        pass

    def show(self):
        pass
