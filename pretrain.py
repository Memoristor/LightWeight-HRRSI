# coding=utf-8

from modules.pretrain import PretrainModule
import numpy as np
import argparse
import warnings
import torch
import random
import os

from backbone.ghostnet_seg import *

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain model')

    parser.add_argument('--train_size', type=int, default=224, help='Image size for training')
    parser.add_argument('--valid_size', type=int, default=256, help='Image size for validation')
    parser.add_argument('--train_batch', type=int, default=512, help='Batch size for training')
    parser.add_argument('--valid_batch', type=int, default=500, help='Batch size for validation')
    parser.add_argument('--num_epoch', type=int, default=120, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer for training')
    parser.add_argument('--init_lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--milestones', type=str, default='30,60,90', help='The steps for adjusting learning rate')
    parser.add_argument('--milestones_gamma', type=float, default=0.1, help='The gamma for adjusting learning rate')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice = {train, test}')

    parser.add_argument('--seed', type=int, default=132, help='The random seed for python and torch')
    parser.add_argument('--model', type=str, default='DDCNetGhostNet0p5', help='The ConvNet model will be used')

    parser.add_argument('--dataset', type=str, default='imagenet', help='The dataset which will be used')
    parser.add_argument('--num_class', type=int, default=1000, help='The number of classes of used dataset')
    parser.add_argument('--data_root', type=str, help='The path of train dataset', default='./data')
    parser.add_argument('--output_root', type=str, help='The path output sources', default='./output')

    parser.add_argument('--gpus', type=str, default='0', help='Device for model training or testing')
    parser.add_argument('--resume', type=str, default='epoch_last.pth', help='The saved model for resume')
    parser.add_argument('--retrain', action='store_true', help='Retrain model from first epoch or not')

    parser.add_argument('--dali_gpu', action='store_true', help='Allow DALI to take a standalone GPU(No.0) or not')
    parser.add_argument('--prefetch', type=int, default=5, help='The max queue size of data prefetch')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['PYTHONASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_cfg = {
        'GhostNet1p0': GhostNet1p0,
        'GhostNet1p3': GhostNet1p3,
        'GhostNet1p5': GhostNet1p5,
    }

    if args.phase == 'train':

        if ',' in args.model:
            model_list = [model_cfg[m] for m in args.model.split(',')]
        else:
            model_list = [model_cfg[args.model]]

        for model in model_list:
            trainer = PretrainModule(model(num_class=args.num_class), args=args)
            trainer.train_model()

    else:  # test
        pass
