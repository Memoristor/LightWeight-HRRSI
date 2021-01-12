# coding=utf-8

import torch

from backbone.basic import *
from backbone.seg_frame import *
from models.seg_ghostnet_decouple import *

__all__ = [
    'SegGhostNetDecoupleScoreBasic',
    'SegGhostNetDecoupleScore1p0',
    'SegGhostNetDecoupleScore1p3',
    'SegGhostNetDecoupleScore1p5'
]


class SegGhostNetDecoupleScoreBasic(SegGhostNetDecoupleBasic):
    """
    Basic ghostnet for segmentation (Version with three skip connections)

    Params:
        in_channel: int. The number of categories of segmentation
        width_mult: float (default 1.0).
    """

    def __init__(self, num_class, width_mult=1.0, pretrained=None):
        super(SegGhostNetDecoupleScoreBasic, self).__init__(num_class, width_mult, pretrained)

        # Fusion module of body and edge
        self.fusion_score = SpatialFusionScore(24)

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
        # fusion = self.fuse_edge(torch.cat([body, edge], dim=1))
        score = self.fusion_score(body + edge)
        fusion = body * score + edge * (1 - score)

        # print('score.shape: {}'.format(score.shape))
        # print('fusion.shape: {}'.format(edge.shape))

        # Conv for output seg maps
        seg = Upsample(self.seg_out(fusion), x.size()[2:])
        bdy = Upsample(self.body_out(body), x.size()[2:])
        edg = Upsample(self.edge_out(edge), x.size()[2:])

        return seg, bdy, edg, score


class SegGhostNetDecoupleScore1p0(SegGhostNetDecoupleScoreBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecoupleScore1p0, self).__init__(
            num_class=num_class,
            width_mult=1.0,
            pretrained='../pretrain/ghostnet_x1.0.pth'
        )

    def forward(self, x):
        seg, bdy, edg, score = super(SegGhostNetDecoupleScore1p0, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
            'scr_sgm': score,
        }


class SegGhostNetDecoupleScore1p3(SegGhostNetDecoupleScoreBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecoupleScore1p3, self).__init__(
            num_class=num_class,
            width_mult=1.3,
            pretrained='../pretrain/ghostnet_x1.3.pth'
        )

    def forward(self, x):
        seg, bdy, edg, score = super(SegGhostNetDecoupleScore1p3, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
            'scr_sgm': score,
        }


class SegGhostNetDecoupleScore1p5(SegGhostNetDecoupleScoreBasic):
    def __init__(self, num_class: int):
        super(SegGhostNetDecoupleScore1p5, self).__init__(
            num_class=num_class,
            width_mult=1.5,
            pretrained='../pretrain/ghostnet_x1.5.pth'
        )

    def forward(self, x):
        seg, bdy, edg, score = super(SegGhostNetDecoupleScore1p5, self).forward(x)
        return {
            'lbl': seg,
            'bdy': bdy,
            'edg': edg.squeeze(1),
            'edg_sgm': torch.sigmoid(edg).squeeze(1),
            'scr_sgm': score,
        }


if __name__ == '__main__':

    model = SegGhostNetDecoupleScore1p0(num_class=5)

    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    for k in y.keys():
        print('{}.shape: {}'.format(k, y[k].shape))
