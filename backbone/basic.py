# coding=utf-8

from torch.nn import functional as F
from torch import nn
import torch

__all__ = ['Upsample', 'Upsample2x', 'ConvBN', 'ConvBNRelu']


def Upsample(x, size, align_corners=True):
    """
    Wrapper Around the Upsample Call
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


def Upsample2x(x, align_corners=True):
    """
    Wrapper Around the Upsample Call
    """
    assert len(x.shape) == 4

    b, c, h, w = x.size()
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=align_corners)


class ConvBN(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        """
        Simple Conv2d + BatchNorm2d + Relu

        Params:
            channel_in: int. The input channel number of hourglass block.
            channel_out: int. The output channel number of hourglass block.
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        """

        super(ConvBN, self).__init__()

        self.primary = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias, padding_mode=padding_mode),
            nn.BatchNorm2d(channel_out),
        )

    def forward(self, x):
        return self.primary(x)


class ConvBNRelu(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        """
        Simple Conv2d + BatchNorm2d + Relu

        Params:
            channel_in: int. The input channel number of hourglass block.
            channel_out: int. The output channel number of hourglass block.
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        """

        super(ConvBNRelu, self).__init__()

        self.primary = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias, padding_mode=padding_mode),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.primary(x)
