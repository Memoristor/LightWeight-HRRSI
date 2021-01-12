# coding=utf-8

import torch
from torch import nn

from backbone.basic import *
from backbone.ghostnet import *
from backbone.ghostnet_seg import *
from backbone.seg_frame import *

__all__ = [
    'SegGhostNetBasic',
    'SegGhostNet1p0',
    'SegGhostNet1p3',
    'SegGhostNet1p5'
]


class SegGhostNetBasic(SegmentationThreeSkips):
    """
    Basic ghostnet for segmentation (Version with three skip connections)

    Params:
        in_channel: int. The number of categories of segmentation
        width_mult: float (default 1.0).
    """

    def __init__(self, num_class, width_mult=1.0, pretrained=None):
        super(SegGhostNetBasic, self).__init__()

        # The main sequence of network
        self.base_network = GhostNetMaps(width=width_mult, pretrained=pretrained)

        # The combine sequence
        c1, c2, c3, c4 = 24, 40, 112, 160
        c1 = make_divisible(c1 * width_mult, 4)
        c2 = make_divisible(c2 * width_mult, 4)
        c3 = make_divisible(c3 * width_mult, 4)
        c4 = make_divisible(c4 * width_mult, 4)

        self.dec_c1 = CombinationModule(c2, c1)
        self.dec_c2 = CombinationModule(c3, c2)
        self.dec_c3 = CombinationModule(c4, c3)

        # The segmentation sequence
        self.final_out = nn.Sequential(
            ConvBNRelu(c1, c1, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(c1, num_class, kernel_size=1, bias=False),
        )


class SegGhostNet1p0(SegGhostNetBasic):
    def __init__(self, num_class: int):
        super(SegGhostNet1p0, self).__init__(num_class, width_mult=1.0, pretrained='../pretrain/ghostnet_x1.0.pth')

    def forward(self, x):
        return {'lbl': Upsample(super(SegGhostNet1p0, self).forward(x), x.size()[2:])}


class SegGhostNet1p3(SegGhostNetBasic):
    def __init__(self, num_class: int):
        super(SegGhostNet1p3, self).__init__(num_class, width_mult=1.3, pretrained='../pretrain/ghostnet_x1.3.pth')

    def forward(self, x):
        return {'lbl': Upsample(super(SegGhostNet1p3, self).forward(x), x.size()[2:])}


class SegGhostNet1p5(SegGhostNetBasic):
    def __init__(self, num_class: int):
        super(SegGhostNet1p5, self).__init__(num_class, width_mult=1.5, pretrained='../pretrain/ghostnet_x1.5.pth')

    def forward(self, x):
        return {'lbl': Upsample(super(SegGhostNet1p5, self).forward(x), x.size()[2:])}


if __name__ == '__main__':

    model = SegGhostNet1p3(num_class=6)

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for k in y.keys():
        print('{}.shape: {}'.format(k, y[k].shape))
