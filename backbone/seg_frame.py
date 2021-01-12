# coding=utf-8

import torch
from torch import nn
from torch.nn import functional as F

from backbone.basic import *

__all__ = [
    'CombinationModule',
    'SpatialFusionScore',
    'DecoupleBodyEdge',
    'SegmentationThreeSkips',
    'SegmentationFourSkips'
]


class CombinationModule(nn.Module):
    """
    The combination module for skip connection

    Params:
        channel_low: int. The channel number of low-resolution feature map
        channel_high: int. The channel number of high-resolution feature map
    """

    def __init__(self, channel_low, channel_high):
        super(CombinationModule, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(channel_low, channel_high, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channel_high),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(channel_high * 2, channel_high, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_high),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_low, x_high):
        x_low = self.up(F.interpolate(x_low, x_high.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_high, x_low), 1))


class DecoupleBodyEdge(nn.Module):
    """
    Get body and edge from current feature map

    Modified from:
    https://github.com/donnyyou/torchcv/blob/1875b7905f4aa0aed7d6cf49ae0ed4512d5f6ae7/model/seg/nets/sfnet.py
    """

    def __init__(self, input_channel):
        super(DecoupleBodyEdge, self).__init__()

        # Conv layers for high resolution feature
        self.conv_high_res = nn.Conv2d(input_channel, input_channel, 1)

        # Conv layers for low resolution feature
        self.conv_low_res = nn.Sequential(
            ConvBN(input_channel, input_channel, 3, stride=2, padding=1),
            ConvBN(input_channel, input_channel, 3, stride=2, padding=1),
            nn.Conv2d(input_channel, input_channel, 1)
        )

        self.flow_make = nn.Conv2d(input_channel * 2, 2, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def flow_warp(inputs, flow, size):
        # Size of feature map corresponding to high resolution low level feature
        out_h, out_w = size
        n, c, h, w = inputs.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)

        # `out_h` points are generated equidistant from -1 to 1, `out_w` points are repeated
        # in each row, and finally (H, W) pixels are generated, which will be defined as `w`,
        # then the transpose matrix of `w` will be generated as `h`
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        # Merge after expansion
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        # `grid` specifies the sampling pixel position normalized by the input space
        # dimension, and most of its values should be in the range of [-1,1]. For example,
        # (x, y) = (-1, -1) is the upper left corner pixel of input, (x, y) = (1, 1) is
        # the lower right corner pixel of input.
        output = F.grid_sample(inputs, grid, align_corners=False)
        return output

    def forward(self, x):
        # Get the size of `x`
        h, w = x.size()[2:]

        # Get high resolution and low resolution feature maps
        high_res = self.conv_high_res(x)
        low_res = self.conv_low_res(x)

        # Wrap high-level and low-level feature
        low_res = Upsample(low_res, (h, w))
        flow = self.flow_make(torch.cat([high_res, low_res], dim=1))
        wrap = self.flow_warp(high_res, flow, (h, w))

        # Get body and edge
        body, edge = wrap, high_res - wrap

        return body, edge


class SpatialFusionScore(nn.Module):
    """
    The fusion module for body and edge (Spatial score-based)
    """

    def __init__(self, input_channel):
        super(SpatialFusionScore, self).__init__()

        # Conv map for H and W
        self.h_map = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.w_map = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)

        # Conv for x
        self.x_map = nn.Conv2d(input_channel, input_channel, kernel_size=1)

        # Fusion score
        self.fusion_score = nn.Sigmoid()

    def forward(self, x):
        # Get size of `x`
        n, c, h, w = x.size()

        # Get `conv_h` and `conv_w`
        conv_h = torch.mean(self.h_map(x), dim=3, keepdim=True)  # n, c, h, 1
        conv_w = torch.mean(self.w_map(x), dim=2, keepdim=True)  # n, c, 1, w
        conv_x = self.x_map(x)  # n, c, h ,w

        conv_h = conv_h.reshape((-1, h, 1))  # n*c, h, 1
        conv_w = conv_w.reshape((-1, 1, w))  # n*c, 1, w

        # Get fusion matrix
        matrix = torch.bmm(conv_h, conv_w).reshape((n, c, h, w))
        fusion = matrix + conv_x

        # Get fusion score
        score = self.fusion_score(fusion)
        return score


class SegmentationThreeSkips(nn.Module):
    """
    Basic framework for segmentation with three combine module
    """

    def __init__(self):
        super(SegmentationThreeSkips, self).__init__()

        # The main sequence of network
        self.base_network = None

        # The combine sequence
        self.dec_c1 = None
        self.dec_c2 = None
        self.dec_c3 = None

        # The segmentation sequence
        self.final_out = None

    def forward(self, x):
        x = self.base_network(x)
        c3_combine = self.dec_c3(x[-1], x[-2])
        c2_combine = self.dec_c2(c3_combine, x[-3])
        c1_combine = self.dec_c1(c2_combine, x[-4])

        return self.final_out(c1_combine)


class SegmentationFourSkips(nn.Module):
    """
    Basic framework for segmentation with four combine module
    """

    def __init__(self):
        super(SegmentationFourSkips, self).__init__()

        # The main sequence of network
        self.base_network = None

        # The combine sequence
        self.dec_c1 = None
        self.dec_c2 = None
        self.dec_c3 = None
        self.dec_c4 = None

        # The segmentation sequence
        self.final_out = None

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        c1_combine = self.dec_c1(c2_combine, x[-5])

        return self.final_out(c1_combine)
