# coding=utf-8

import torch
from torch import nn
from backbone.resnet import *
from backbone.basic import *
from backbone.seg_frame import *

__all__ = [
    'SegResNet18',
    'SegResNet34',
    'SegResNet50',
    'SegResNet101'
]


class SegResNet18(SegmentationThreeSkips):
    """
    Simple resnet18 for segmentation (Version with three skip connections)

    Params:
        num_class: int. The number of categories of segmentation
    """

    def __init__(self, num_class):
        super(SegResNet18, self).__init__()

        self.num_class = num_class

        # The main sequence of network
        self.base_network = resnet18(pretrained=True)

        # The combine sequence
        self.dec_c1 = CombinationModule(128, 64)
        self.dec_c2 = CombinationModule(256, 128)
        self.dec_c3 = CombinationModule(512, 256)

        # The segmentation sequence
        self.final_out = nn.Sequential(
            ConvBNRelu(64, 64, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, self.num_class, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return {'lbl': Upsample(super(SegResNet18, self).forward(x), x.size()[2:])}


class SegResNet34(SegResNet18):
    """
    Simple resnet34 for segmentation (Version with three skip connections)

    Params:
        num_class: int. The number of categories of segmentation
    """

    def __init__(self, *args, **kwargs):
        super(SegResNet34, self).__init__(*args, **kwargs)

        # The main sequence of network
        self.base_network = resnet34(pretrained=True)

        # The combine sequence
        self.dec_c1 = CombinationModule(128, 64)
        self.dec_c2 = CombinationModule(256, 128)
        self.dec_c3 = CombinationModule(512, 256)

        # The segmentation sequence
        self.final_out = nn.Sequential(
            ConvBNRelu(64, 64, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, self.num_class, kernel_size=1, bias=False),
        )


class SegResNet50(SegResNet18):
    """
    Simple resnet50 for segmentation (Version with three skip connections)

    Params:
        num_class: int. The number of categories of segmentation
    """

    def __init__(self, *args, **kwargs):
        super(SegResNet50, self).__init__(*args, **kwargs)

        # The main sequence of network
        self.base_network = resnet50(pretrained=True)

        # The combine sequence
        self.dec_c1 = CombinationModule(512, 256)
        self.dec_c2 = CombinationModule(1024, 512)
        self.dec_c3 = CombinationModule(2048, 1024)

        # The segmentation sequence
        self.final_out = nn.Sequential(
            ConvBNRelu(256, 256, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(256, 256, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(256, self.num_class, kernel_size=1, bias=False),
        )


class SegResNet101(SegResNet18):
    """
    Simple resnet50 for segmentation (Version with three skip connections)

    Params:
        num_class: int. The number of categories of segmentation
    """

    def __init__(self, *args, **kwargs):
        super(SegResNet101, self).__init__(*args, **kwargs)

        # The main sequence of network
        self.base_network = resnet101(pretrained=True)

        # The combine sequence
        self.dec_c1 = CombinationModule(512, 256)
        self.dec_c2 = CombinationModule(1024, 512)
        self.dec_c3 = CombinationModule(2048, 1024)

        # The segmentation sequence
        self.final_out = nn.Sequential(
            ConvBNRelu(256, 256, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(256, 256, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(256, self.num_class, kernel_size=1, bias=False),
        )


if __name__ == '__main__':

    model = SegResNet101(num_class=6)

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for k in y.keys():
        print('{}.shape: {}'.format(k, y[k].shape))
