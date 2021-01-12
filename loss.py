# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CrossEntropy2D',
    'BinaryCrossEntropy2D',
    'ImageBasedCrossEntropy2D',
    'BoundariesRelaxation2D',
    'LabelSmoothCrossEntropy',
    'LabelSmoothCrossEntropy2D'
]


class BasicLossModule(nn.Module):
    """
    Basic loss module, please do not call this module

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)
    """

    def __init__(self, ignore_index=255, custom_weight=None, batch_weight=True,
                 size_average=True, batch_average=True, upper_bound=1.0):
        super(BasicLossModule, self).__init__()

        # Init custom weight
        if custom_weight is not None:
            if isinstance(custom_weight, list):
                custom_weight = torch.from_numpy(np.array(custom_weight, dtype=np.float32)).float()
            else:
                custom_weight = torch.from_numpy(custom_weight).float()

        # Add to properties
        self.ignore_index = ignore_index
        self.custom_weight = custom_weight
        self.batch_weight = batch_weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.upper_bound = upper_bound

    def calculate_weights(self, target, num_classes):
        """
        Calculate weights of classes based on the training crop

        Params:
            target: 3-D torch.Tensor. The input target which shape is (n, h, w)
            num_classes: int. The number of classes
        """
        hist = torch.histc(target, bins=num_classes, min=0, max=num_classes - 1)
        hist = hist / hist.sum()
        hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist


class CrossEntropy2D(BasicLossModule):
    """
    The 2D cross entropy loss.

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        logit: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def __init__(self, *args, **kwargs):
        super(CrossEntropy2D, self).__init__(*args, **kwargs)

    def forward(self, logit, target):

        assert len(logit.shape) == 4 and len(target.shape) == 3

        # Get the size of `logit`
        n, c, h, w = logit.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(logit.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(logit.device)
            else:
                weight = None

        # `size_average` and `reduce` of `nn.CrossEntropyLoss` are in the process of being deprecated
        loss = F.cross_entropy(logit, target.long(), weight, ignore_index=self.ignore_index, reduction='sum')

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class BinaryCrossEntropy2D(BasicLossModule):
    """
    The binary 2D cross entropy loss.
    Note that `F.binary_cross_entropy_with_logits` do not support `ignore_index`

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.8], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        logit: 3-D or 4-D torch.Tensor. The predict result without `sigmoid/softmax`, if `logit` is
            3-D/4D Tensor, the shape of `logit` should be (n,h,w)/(n,c,h,w) respectively
        target: torch.Tensor. The input target which shape should be same as `logit`

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def __init__(self, *args, **kwargs):
        super(BinaryCrossEntropy2D, self).__init__(*args, **kwargs)

    def forward(self, logit, target):

        assert len(logit.shape) == len(target.shape)

        # Get the size of `logit`
        if len(logit.shape) == 4:
            n, c, h, w = logit.size()
        elif len(logit.shape) == 3:
            n, h, w = logit.size()
        else:
            raise AttributeError('Expect `logit` is a 3-D or 4-D Tensor, but {}-D instead'.format(len(logit.shape)))

        # Reshape as (n, h*w*c)/(n, h*w)
        if len(logit.shape) == 4:
            logit_rsp = logit.transpose(1, 2).transpose(2, 3).reshape(1, -1)
            target_rsp = target.transpose(1, 2).transpose(2, 3).reshape(1, -1)
        else:
            logit_rsp = logit.reshape(1, -1)
            target_rsp = target.reshape(1, -1)

        # Get positive/negative/ignore index
        pos_index = (target_rsp == 1)
        neg_index = (target_rsp == 0)
        # ign_index = (target_rsp == self.ignore_index)
        ign_index = (target_rsp > 1)

        # Convert `target_rsp[ign_index]` to `0` first
        target_rsp[ign_index] = 0

        # Convert `positive/negative/ignore index` as `bool`
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ign_index = ign_index.data.cpu().numpy().astype(bool)

        # Calculate the weight
        weight = np.zeros(logit_rsp.size(), dtype=np.float32)
        if self.custom_weight is not None:
            weight[neg_index] = self.weight[0] * 1.0
            weight[pos_index] = self.weight[1] * 1.0
            weight[ign_index] = 0  # weight for `ignore_index` is 0 !

            weight = torch.from_numpy(weight.astype(np.float32)).to(logit.device)
        else:
            if self.batch_weight:
                pos_num = pos_index.sum()
                neg_num = neg_index.sum()
                sum_num = pos_num + neg_num
                if sum_num != 0:
                    weight[pos_index] = 1 + (neg_num * 1.0 / sum_num) if pos_num > 0 else 1
                    weight[neg_index] = 1 + (pos_num * 1.0 / sum_num) if neg_num > 0 else 1
                    weight[ign_index] = 0  # weight for `ignore_index` is 0 !
                else:
                    raise AttributeError('The sum of `pos_index` and `neg_index` is 0')

                weight = torch.from_numpy(weight.astype(np.float32)).to(logit.device)
            else:
                weight = None

        # Calculate binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(logit_rsp, target_rsp, weight=weight, reduction='sum')

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class ImageBasedCrossEntropy2D(BasicLossModule):
    """
    Image Weighted Cross Entropy Loss

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        logit: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def __init__(self, *args, **kwargs):
        super(ImageBasedCrossEntropy2D, self).__init__(*args, **kwargs)

    def forward(self, logit, target):

        assert len(logit.shape) == 4 and len(target.shape) == 3

        # Get the size of `logit`
        n, c, h, w = logit.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(logit.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(logit.device)
            else:
                weight = None

        # Calculate the loss
        loss = F.nll_loss(F.log_softmax(logit, dim=1), target.long(), weight,
                          ignore_index=self.ignore_index, reduction='sum')

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class BoundariesRelaxation2D(BasicLossModule):
    """
    The boundaries relaxation loss, which details can be seen here:
    https://ieeexplore.ieee.org/abstract/document/8954327

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)
        window_size: int, list or tuple (default 3). The slide window size of boundaries relaxation loss
        stride: int, list or tuple (default 1). The strode of slide window

    forward:
        logit: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def __init__(self, window_size=3, stride=1, *args, **kwargs):
        super(BoundariesRelaxation2D, self).__init__(*args, **kwargs)

        # Init window size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        elif isinstance(window_size, list or tuple) and len(window_size) == 2:
            window_size = tuple(window_size)
        else:
            raise AttributeError('Expect type of `window_size`: int, 2-elem list or tuple')

        # Init stride
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, list or tuple) and len(stride) == 2:
            stride = tuple(stride)
        else:
            raise AttributeError('Expect type of `stride`: int, 2-elem list or tuple')

        self.window_size = window_size
        self.stride = stride
        self.pool2d = nn.AvgPool2d(kernel_size=self.window_size, stride=self.stride)

    def forward(self, logit, target):

        # Get the size of `logit`
        n, c, h, w = logit.size()

        # Init weight
        if self.custom_weight is not None:
            weight = self.custom_weight.to(logit.device)
        else:
            if self.batch_weight:
                weight = self.calculate_weights(target, c).to(logit.device)
            else:
                weight = None

        # Get soft output of `logit`
        logit_soft = F.softmax(logit, dim=1)

        # Get `ignore_index`
        ignore_index = (target == self.ignore_index)

        # Convert 3-D `target` Tensor to 4-D `onehot` Tensor
        target_clamps = target.clone()
        target_clamps[ignore_index] = c - 1
        target_onehot = F.one_hot(target_clamps.long(), c)  # n,h,w,c
        target_onehot[ignore_index] = 0  # n,h,w,c
        target_onehot_trans = target_onehot.transpose(2, 3).transpose(1, 2)  # n,c,h,w

        # Get the boundaries relaxation result of `logit` and `target`
        logit_br = self.pool2d(logit_soft)
        target_br = self.pool2d(target_onehot_trans.float())

        # Get loss, note that the loss' lower bound is 0
        loss = - target_br * (torch.log(logit_br + 1e-14) - torch.log(target_br + 1e-14))

        # Get new loss if `weight` is not None
        if weight is not None:
            weight_matrix = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1,c,1,1
            loss = loss * weight_matrix

        # Get sum of loss
        loss = loss.sum()

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class LabelSmoothCrossEntropy(BasicLossModule):
    """
    Labels Smooth Loss for classification

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w) (--unused for this version)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        logit: 2-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c)
        target: 1-D torch.Tensor. The input target which shape is (n)

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def __init__(self, epsilon=0.1, *args, **kwargs):
        super(LabelSmoothCrossEntropy, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit, target):

        assert len(logit.shape) == 2 and len(target.shape) == 1

        # Get the size of `logit`
        n, c = logit.size()

        # Get `ign_index` and `pos_index`
        ign_index = (target == self.ignore_index)
        pos_index = (target != self.ignore_index)
        pos_index = pos_index.data.cpu().numpy().astype(bool)

        # Init weight
        weight = np.zeros(logit.size(), dtype=np.float32)
        if self.custom_weight is not None:
            weight[pos_index, :] = self.custom_weight
            weight = torch.from_numpy(weight).to(logit.device)
        else:
            if self.batch_weight:
                weight[pos_index, :] = self.calculate_weights(target, c).data.cpu().numpy()
                weight = torch.from_numpy(weight).to(logit.device)
            else:
                weight = None

        # Convert 1-D `target` Tensor to 2-D `onehot` Tensor
        target_clamps = target.clone()
        target_clamps[ign_index] = c - 1
        target_onehot = F.one_hot(target_clamps.long(), c)  # n,c

        # Soft target and log logit
        target_soft = (1 - self.epsilon) * target_onehot + self.epsilon / c
        log_logit = self.log_softmax(logit)

        # Get sum of loss
        loss = -target_soft * log_logit
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()

        # # loss will be divided by (h * w)
        # if self.size_average:
        #     loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


class LabelSmoothCrossEntropy2D(LabelSmoothCrossEntropy):
    """
    Labels Smooth Loss 2D for segmentation

    Params:
        ignore_index: int (default 255). Categories to be ignored of `target`
        custom_weight: numpy array or list (default None). The weight of each category. For example,
            [0.2, 0.1, ..., 0.1], if `custom_weight` is not None, `batch_weight` will be ignored
        batch_weight: bool (default True). If true, the whole batch is used to calculate weights
        size_average: bool (default True). Loss will be divided by (h * w)
        batch_average: bool (default True). Loss will be divided by (n)

    Forward:
        logit: 4-D torch.Tensor. The predict result without `sigmoid/softmax`, which shape is (n, c, h, w)
        target: 3-D torch.Tensor. The input target which shape is (n, h, w)

        Note that there's no need to use `softmax/sigmoid` for `logit` before calling this loss module.
    """

    def forward(self, logit, target):

        assert len(logit.shape) == 4 and len(target.shape) == 3

        # Get the size of `logit`
        n, c, h, w = logit.size()

        # Get `ign_index` and `pos_index`
        ign_index = (target == self.ignore_index)
        pos_index = (target != self.ignore_index)
        pos_index = pos_index.data.cpu().numpy().astype(bool)  # n,h,w

        # Init weight
        weight = torch.Tensor(logit.size()).fill_(0).numpy().transpose((0, 2, 3, 1))  # n,h,w,c
        if self.custom_weight is not None:
            weight[pos_index] = self.custom_weight
            weight = weight.transpose((0, 3, 1, 2))  # n,c,h,w
            weight = torch.from_numpy(weight).to(logit.device)
        else:
            if self.batch_weight:
                weight[pos_index] = self.calculate_weights(target, c).data.cpu().numpy()
                weight = weight.transpose((0, 3, 1, 2))  # n,c,h,w
                weight = torch.from_numpy(weight).to(logit.device)
            else:
                weight = None

        # Convert 1-D `target` Tensor to 2-D `onehot` Tensor
        target_clamps = target.clone()
        target_clamps[ign_index] = c - 1
        target_onehot = F.one_hot(target_clamps.long(), c)  # n,h,w,c

        # Soft target and log logit
        target_soft = (1 - self.epsilon) * target_onehot + self.epsilon / c
        target_soft = target_soft.permute((0, 3, 1, 2))  # n,c,h,w
        log_logit = self.log_softmax(logit)

        # Get sum of loss
        loss = -target_soft * log_logit
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()

        # loss will be divided by (h * w)
        if self.size_average:
            loss /= (h * w)

        # loss will be divided by (n)
        if self.batch_average:
            loss /= n

        return loss


if __name__ == '__main__':
    size = (256, 256)
    num_class = 9
    batch_size = 5

    # torch.manual_seed(1)
    torch.clear_autocast_cache()

    print('=' * 30 + ' CrossEntropy2D ' + '=' * 30)
    z = torch.randn(batch_size, num_class, *size, requires_grad=True).cuda()
    y = torch.randint(num_class, (batch_size, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = CrossEntropy2D()(z, y)
    print(l.detach().cpu().numpy())

    print('=' * 30 + ' BinaryCrossEntropy2D ' + '=' * 30)
    z = torch.randn(batch_size, *size, requires_grad=True).cuda()
    y = torch.randint(2, (batch_size, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = BinaryCrossEntropy2D()(z, y)
    print(l.detach().cpu().numpy())

    print()

    z = torch.randn(batch_size, num_class, *size, requires_grad=True).cuda()
    y = torch.randint(2, (batch_size, num_class, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = BinaryCrossEntropy2D()(z, y)
    print(l.detach().cpu().numpy())

    print('=' * 30 + ' ImageBasedCrossEntropy2D ' + '=' * 30)
    z = torch.randn(batch_size, num_class, *size, requires_grad=True).cuda()
    y = torch.randint(num_class, (batch_size, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = ImageBasedCrossEntropy2D(num_class, batch_weight=True)(z, y)
    print(l.detach().cpu().numpy())

    print('=' * 30 + ' BoundariesRelaxation2D ' + '=' * 30)
    z = torch.randn(batch_size, num_class, *size, requires_grad=True).cuda()
    y = torch.randint(num_class, (batch_size, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = BoundariesRelaxation2D(window_size=10, custom_weight=np.arange(num_class) + 1)(z, y)
    print(l.detach().cpu().numpy())

    print('=' * 30 + ' BoundariesRelaxation2D ' + '=' * 30)
    z = torch.randn(batch_size, num_class, requires_grad=True).cuda()
    y = torch.randint(num_class, (batch_size,), dtype=torch.float).cuda()
    y[0] = 255
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = LabelSmoothCrossEntropy(epsilon=0)(z, y)
    print(l.detach().cpu().numpy())

    print('=' * 30 + ' LabelSmoothCrossEntropy2D ' + '=' * 30)
    z = torch.randn(batch_size, num_class, *size, requires_grad=True).cuda()
    y = torch.randint(num_class, (batch_size, *size), dtype=torch.float).cuda()
    print('z.shape: {}, y.shape: {}'.format(z.shape, y.shape))
    l = LabelSmoothCrossEntropy2D(epsilon=0)(z, y)
    print(l.detach().cpu().numpy())
