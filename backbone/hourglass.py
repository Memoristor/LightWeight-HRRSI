# coding=utf-8

from backbone.basic import *
import torch.nn as nn
import torch

__all__ = ['HourglassNet']


class HourglassResNet(nn.Module):
    """
    Define a residual network blocks for hourglass network.

    Params:
        channel_in: int. The input channel number of ResNet block.
        channel_out: int. The output channel number of ResNet block.
    """

    def __init__(self, channel_in, channel_out):
        super(HourglassResNet, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.channel_reduce = self.channel_out // 2

        # The main sequence of ResNet block
        self.main = nn.Sequential(
            ConvBNRelu(self.channel_in, self.channel_reduce, kernel_size=1, stride=1),
            ConvBNRelu(self.channel_reduce, self.channel_reduce, kernel_size=3, stride=1, padding=1),
            ConvBN(self.channel_reduce, self.channel_out, kernel_size=1, stride=1)
        )

        # The shortcut sequence of ResNet block
        self.shortcut = ConvBN(self.channel_in, self.channel_out, kernel_size=3, stride=1, padding=1)

        # The concat sequence
        self.relu = nn.ReLU()

    def forward(self, x):
        net = torch.add(self.main(x), self.shortcut(x))
        return self.relu(net)


class HourglassNet(nn.Module):
    def __init__(self, channel_in, channel_out, channel_hidden=256, hourglass_order=3):
        """
       Define a residual network blocks for hourglass network.

       Params:
           channel_in: int. The input channel number of hourglass block.
           channel_out: int. The output channel number of hourglass block.
           channel_hidden: int. The hidden channel number of 1th-order hourglass block
           hourglass_order: int. The N-orders of hourglass block.
       """

        super(HourglassNet, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.channel_hidden = channel_hidden
        self.hourglass_order = hourglass_order

        # The main sequence of Hourglass block
        if self.hourglass_order == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.channel_in, self.channel_hidden, kernel_size=3, stride=2, padding=1),  # Replace MaxPool
                HourglassResNet(self.channel_hidden, self.channel_hidden),
                HourglassResNet(self.channel_hidden, self.channel_hidden),

                HourglassResNet(self.channel_hidden, self.channel_hidden),
                HourglassResNet(self.channel_hidden, self.channel_hidden),

                HourglassResNet(self.channel_hidden, self.channel_hidden),
                HourglassResNet(self.channel_hidden, self.channel_out),  # Output channel
                nn.Upsample(scale_factor=2)
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(self.channel_in, self.channel_hidden, kernel_size=3, stride=2, padding=1),  # Replace MaxPool
                HourglassResNet(self.channel_hidden, self.channel_hidden),
                HourglassResNet(self.channel_hidden, self.channel_hidden),

                HourglassNet(self.channel_hidden, self.channel_hidden,
                             channel_hidden=self.channel_hidden * 2,
                             hourglass_order=self.hourglass_order - 1),

                HourglassResNet(self.channel_hidden, self.channel_hidden),
                HourglassResNet(self.channel_hidden, self.channel_out),
                nn.Upsample(scale_factor=2)
            )

        # The shortcut sequence of Hourglass block
        self.shortcut = nn.Sequential(
            HourglassResNet(self.channel_in, self.channel_hidden),
            HourglassResNet(self.channel_hidden, self.channel_out),
        )

        # The relu sequence
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(torch.add(self.main(x), self.shortcut(x)))
