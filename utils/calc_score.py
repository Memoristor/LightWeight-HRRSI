# coding=utf-8

import prettytable as pt
import numpy as np

__all__ = ['SegmentationScore', 'ClassificationScore']


class SegmentationScore(object):
    """
    Ternary classification confusion matrix:
        ------------------------------
                True1   True2   True3
        Pred1   TP1,1   FN2,1   FN3,1
        Pred2   FN1,2   TP2,2   FN3,2
        Pred3   FN1,3   FN2,3   TP3,3
        ------------------------------
        All     100%    100%    100%

    Params:
        clazz_encode: list. The encoded classes, i.e. ['IS', 'BD', ...]
    """

    def __init__(self, clazz_encode, float_format='5.6f'):
        self.clazz_encode = clazz_encode
        self.float_format = '{:' + float_format + '}'
        self.num_class = len(self.clazz_encode)
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int)

    def get_target_matrix(self, ignore=None):
        """
        Get target confusion matrix from confusion matrix

        Params:
            ignore: str or list of str. The classes which should be ignored.
        """
        if ignore is None:
            target_matrix = self.confusion_matrix
        else:
            if isinstance(ignore, str):
                ignore = [ignore]
            elif isinstance(ignore, list or tuple):
                ignore = list(ignore)
            else:
                raise AttributeError('Expected None, list or tuple, but given: {}'.format(type(ignore)))

            target_idx = list()
            for i in range(self.num_class):
                if self.clazz_encode[i] not in ignore:
                    target_idx.append(i)

            target_matrix = self.confusion_matrix[target_idx][:, target_idx]

        return target_matrix

    def get_target_label(self, ignore=None):
        """
        Get target labels from encoded labels

        Params:
            ignore: str or list of str. The classes which should be ignored.
        """
        if ignore is None:
            target_label = self.clazz_encode
        else:
            if isinstance(ignore, str):
                ignore = [ignore]
            elif isinstance(ignore, list or tuple):
                ignore = list(ignore)
            else:
                raise AttributeError('Expected None, list or tuple, but given: {}'.format(type(ignore)))

            target_label = list()
            for lbl in self.clazz_encode:
                if lbl not in ignore:
                    target_label.append(lbl)

        return target_label

    def get_F1(self, ignore=None):
        """
        Calculate F1 score by the confusion matrix:

        F1 = 2 * (PR) / (P + R)
        """
        target_matrix = self.get_target_matrix(ignore)

        score = []
        for i in range(target_matrix.shape[0]):
            p = target_matrix[i, i] / np.sum(target_matrix[:, i])
            r = target_matrix[i, i] / np.sum(target_matrix[i, :])
            f1 = (2.0 * p * r) / (p + r)
            score.append(f1)

        return np.array(score)

    def get_OA(self, ignore=None):
        """
        Get OA(Overall Accuracy) by the confusion matrix, it is calculated as: the sum
        of diagonal elements divided by the sum of all elements of the matrix.

        OA = (TP + TN) / (TP + TN + FP + TN)
        """
        target_matrix = self.get_target_matrix(ignore)
        return np.diag(target_matrix).sum() / target_matrix.sum()

    def get_CA(self, ignore=None):
        """
        The CA(Class Accuracy)is to calculate the proportion of the correct pixel number
        of each category in all predicted pixels of the category, that is, the accuracy
        rate, and then accumulate the average. The formula for calculating the CA using
        the confusion matrix is: MA is equal to TP on the diagonal divided by the total
        number of pixels in the corresponding column

        MA = (TP) / (TP + FP)
        """
        target_matrix = self.get_target_matrix(ignore)
        return np.diag(target_matrix) / target_matrix.sum(axis=0)

    def get_IoU(self, ignore=None):
        """
        The IoU is the result of summing and averaging the ratio of the intersection
        and union between the predicted result and the true value of each category.

        IoU = (TP) / (TP + FP + FN)
        """
        target_matrix = self.get_target_matrix(ignore)
        return np.diag(target_matrix) / (target_matrix.sum(axis=1) + target_matrix.sum(axis=0) - np.diag(target_matrix))

    def update(self, pred, truth):
        """
        Update the confusion matrix by ground truth seg map and predict seg map.

        Params:
            pred: 2-D/3-D np.array. The predict seg map, which shape is (H, W) or (N, H, W)
            truth: 2-D/3-D np.array. The ground truth seg map, which shape is (H, W) or (N, H, W)
        """
        mask = (truth >= 0) & (truth < self.num_class)
        hist = np.bincount(
            self.num_class * truth[mask].astype(int) + pred[mask],
            minlength=self.num_class ** 2,
        ).reshape(self.num_class, self.num_class).T

        self.confusion_matrix += hist

    def reset(self):
        """
        Reset confusion matrix
        """
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int)

    def pt_confusion_matrix(self, show=True):
        """
        Print confusion matrix by `prettytable`
        """
        tb = pt.PrettyTable()

        field = ["P|T"]
        for cls in self.clazz_encode:
            field.append(cls)

        tb.field_names = field

        for i, cls in enumerate(self.clazz_encode):
            tb.add_row([cls, *self.confusion_matrix[i].tolist()])

        if show:
            print(tb)

        return tb

    def pt_score(self, key, label=None, ignore=None, show=True):
        """
        Print keys by `prettytable`

        Params
            key: str or list of str. The keys for pretty table, support ['CA', 'IoU', 'F1']
        """
        if isinstance(key, str):
            key = [key]
        elif isinstance(key, list or tuple):
            key = list(key)
        else:
            raise AttributeError('Expected keys: None, str, list or tuple, but given {}'.format(type(key)))

        tb = pt.PrettyTable()

        field = ["" if label is None else label]
        field.extend(self.get_target_label(ignore))

        tb.field_names = field
        for k in key:
            if k == 'CA':
                row_str = [self.float_format.format(x) for x in self.get_CA(ignore).tolist()]
            elif k == 'IoU':
                row_str = [self.float_format.format(x) for x in self.get_IoU(ignore).tolist()]
            elif k == 'F1':
                row_str = [self.float_format.format(x) for x in self.get_F1(ignore).tolist()]
            else:
                raise AttributeError('Unknown key: {}'.format(k))
            tb.add_row([k, *row_str])

        if show:
            print(tb)

        return tb


class ClassificationScore(object):
    """
    The top-1 and top-5 evaluation criteria for Imagenet image classification competition.
    """

    def __init__(self, num_class):
        self.num_class = num_class

        self.top1_state = {'sum_acc': 0, 'count': 0}
        self.top5_state = {'sum_acc': 0, 'count': 0}

    @staticmethod
    def topk_acc(pred, truth, topk=(1,)):
        """
        Get top-k accuracy by predict score for each class and the truth classes

        Note: There are still some bugs, which get different result from torch.topk()
        """
        pred_maxk = np.argsort(pred, axis=1)[:, -max(topk):][:, ::-1]
        # print(pred_maxk[0])

        acc = list()
        for k in topk:
            pred_match = np.logical_or.reduce(pred_maxk[:, :k] == truth.reshape((-1, 1)), axis=1)
            acc.append(pred_match.astype(np.int).sum() / pred.shape[0])
        return acc

    def update(self, pred, truth):
        """
        Update state with predict score for each class and the truth classes

        Params:
            pred: 2-D np.array. The predict classification prob which shape is (N, C)
            truth: 1-D np.array. The ground truth classes, which shape is (N, )
        """
        # Get size of predict
        n, c = pred.shape

        # Get top-1 and top-k acc
        topk_acc_list = self.topk_acc(pred, truth, topk=(1, 5))

        # Get top-1 state
        self.top1_state['sum_acc'] += topk_acc_list[0] * n
        self.top1_state['count'] += n

        # Get top-5 state
        self.top5_state['sum_acc'] += topk_acc_list[1] * n
        self.top5_state['count'] += n

    def reset(self):
        """
        Reset state
        """
        self.top1_state = {'sum_acc': 0, 'count': 0}
        self.top5_state = {'sum_acc': 0, 'count': 0}

    def get_top1_acc(self):
        """
        Get top-1 accuracy
        """
        return self.top1_state['sum_acc'] / self.top1_state['count']

    def get_top5_acc(self):
        """
        Get top-5 accuracy
        """
        return self.top5_state['sum_acc'] / self.top5_state['count']

    def pt_score(self, key, label=None, show=True):
        """
        Print keys by `prettytable`

        Params
            key: str or list of str. The keys for pretty table, support ['top1_acc', 'top5_acc']
        """
        if isinstance(key, str):
            key = [key]
        elif isinstance(key, list or tuple):
            key = list(key)
        else:
            raise AttributeError('Expected keys: None, str, list or tuple, but given {}'.format(type(key)))

        tb = pt.PrettyTable()

        field = ["" if label is None else label]
        field.extend(key)
        tb.field_names = field

        row = ['Value']
        for k in key:
            if k == 'top1_acc':
                row.append(self.get_top1_acc())
            elif k == 'top5_acc':
                row.append(self.get_top5_acc())
            else:
                raise AttributeError('Unknown key: {}'.format(k))
        tb.add_row(row)

        if show:
            print(tb)

        return tb
