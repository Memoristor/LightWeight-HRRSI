# coding=utf-8

from backbone.ghostnet import *
import torch
import os

__all__ = ['GhostNetMaps', 'GhostNet1p0', 'GhostNet1p3', 'GhostNet1p5']


class GhostNetMaps(GhostNet):
    """
    Get GhostNet's feature maps, RGB image as input data, it should be noted that
    the layers for classification has been removed in this module
    """

    def __init__(self, cfg=ghostnet_cfgs, width=1.0, pretrained=None):
        # num_class and dropout will not be used
        super(GhostNetMaps, self).__init__(cfg, 1000, width, 0.2)

        # Try to load parameters if `pretrained` is not None
        if pretrained is not None:
            if os.path.exists(pretrained):
                print('[{}] Load parameters from `{}`'.format(self.__class__.__name__, pretrained))
                self.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage), strict=False)
            else:
                print('[{}] `{}` do not exist'.format(self.__class__.__name__, pretrained))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        out = list()
        for block in self.blocks[:-1]:
            if len(out) == 0:
                out.append(block(x))
            else:
                out.append(block(out[-1]))

        # for o in out:
        #     print(o.shape)

        out = [out[0], out[2], out[4], out[6], out[8]]

        # print('-' * 40)
        # for o in out:
        #     print(o.shape)

        return out


class GhostNet1p0(GhostNet):
    def __init__(self, num_class=1000, dropout=0.2):
        super(GhostNet1p0, self).__init__(ghostnet_cfgs, num_class, 1.0, dropout)


class GhostNet1p3(GhostNet):
    def __init__(self, num_class=1000, dropout=0.2):
        super(GhostNet1p3, self).__init__(ghostnet_cfgs, num_class, 1.3, dropout)


class GhostNet1p5(GhostNet):
    def __init__(self, num_class=1000, dropout=0.2):
        super(GhostNet1p5, self).__init__(ghostnet_cfgs, num_class, 1.5, dropout)
