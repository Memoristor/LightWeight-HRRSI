# coding=utf-8

import torch
from torch import nn

from backbone.basic import *
from backbone.seg_frame import *
from models.seg_ghostnet import *

__all__ = [
    'SegGhostNetDecoupleBasic',
    'SegGhostNetDecouple1p0',
    'SegGhostNetDecouple1p3',
    'SegGhostNetDecouple1p5'
]


class SegGhostNetDecoupleBasic(SegGhostNetBasic):
    """
    Basic ghostnet for segmentation (Version with three skip connections)

    Params:
        in_channel: int. The number of categories of segmentation
        width_mult: float (default 1.0).
    """

    def __init__(self, num_class, width_mult=1.0, pretrained=None):
        super(SegGhostNetDecoupleBasic, self).__init__(24, width_mult, pretrained)

        # The feature map is decoupled to the body and the edge
        self.decouple_body_edge = DecoupleBodyEdge(24)

        # Conv for decoupled body and edge
        self.decouple_body = nn.Sequential(
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
        )

        self.decouple_edge = nn.Sequential(
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
        )

        # The conv for feature map of body and edge
        # self.fuse_body_edge = nn.Conv2d(24 + 24, 24, kernel_size=1)

        # Conv for output seg maps
        self.seg_out = nn.Sequential(
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(24, num_class, kernel_size=1),
        )

        self.body_out = nn.Sequential(
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(24, num_class, kernel_size=1),
        )

        self.edge_out = nn.Sequential(
            ConvBNRelu(24, 24, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(24, 1, kernel_size=1),
        )

    def forward(self, x):
        # Get feature map
        f = super(SegGhostNetDecoupleBasic, self).forward(x)

        # Get decoupled body and edge
        dec_body, dec_edge = self.decouple_body_edge(f)

        # Conv for decoupled body and edge
        body = self.decouple_body(dec_body)
        edge = self.decouple_edge(dec_edge)

        # print('body.shape: {}'.format(body.shape))
        # print('edge.shape: {}'.format(edge.shape))

        # The fusion of body and edge
        # fusion = self.fuse_body_edge(torch.cat([body, edge], dim=1))
        fusion = body + edge

        # Conv for output seg maps
        seg = Upsample(self.seg_out(fusion), x.size()[2:])
        bdy = Upsample(self.body_out(body), x.size()[2:])
        edg = Upsample(self.edge_out(edge), x.size()[2:])

        return seg, bdy, edg


class SegGhostNetDecouple1p0(SegGhostNetDecoupleBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecouple1p0, self).__init__(
            num_class=num_class,
            width_mult=1.0,
            pretrained='../pretrain/ghostnet_x1.0.pth'
        )

    def forward(self, x):
        seg, bdy, edg = super(SegGhostNetDecouple1p0, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
        }


class SegGhostNetDecouple1p3(SegGhostNetDecoupleBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecouple1p3, self).__init__(
            num_class=num_class,
            width_mult=1.3,
            pretrained='../pretrain/ghostnet_x1.3.pth'
        )

    def forward(self, x):
        seg, bdy, edg = super(SegGhostNetDecouple1p3, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
        }


class SegGhostNetDecouple1p5(SegGhostNetDecoupleBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecouple1p5, self).__init__(
            num_class=num_class,
            width_mult=1.5,
            pretrained='../pretrain/ghostnet_x1.5.pth'
        )

    def forward(self, x):
        seg, bdy, edg = super(SegGhostNetDecouple1p5, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
        }


if __name__ == '__main__':

    model = SegGhostNetDecouple1p0(num_class=6)

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for k in y.keys():
        print('{}.shape: {}'.format(k, y[k].shape))
