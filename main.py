# coding=utf-8

from datasets.seg_data import SegDataset
from modules.train import TrainModule
from modules.test import TestModule
import numpy as np
import argparse
import warnings
import torch
import random
import os

from models.seg_resnet import *
from models.seg_ghostnet import *
from models.seg_ghostnet_decouple import *
from models.seg_ghostnet_decouple_score import *

from models.previous.deeplabv3_plus import *
from models.previous.fcn import *
from models.previous.psp_net import *
from models.previous.psp_net_vgg import *
from models.previous.seg_net import *
from models.previous.u_net import *

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument('--input_size', type=int, default=512, help='Resized image size')
    parser.add_argument('--num_epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer for training')
    parser.add_argument('--init_lr', type=float, default=0.002, help='Initial learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='The gamma for the learning rate adjustment')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice = {train, test}')

    parser.add_argument('--seed', type=int, default=132, help='The random seed for python and torch')
    parser.add_argument('--model', type=str, default='SegResNet18', help='The ConvNet model will be used')

    parser.add_argument('--dataset', type=str, default='vaihingen', help='The dataset which will be used')
    parser.add_argument('--data_root', type=str, help='The path of train dataset', default='./data')
    parser.add_argument('--output_root', type=str, help='The path output sources', default='./output')

    parser.add_argument('--gpus', type=str, default='0', help='Device for model training or testing')
    parser.add_argument('--resume', type=str, default='epoch_last.pth', help='The saved model for resume')
    parser.add_argument('--retrain', action='store_true', help='Retrain model from first epoch or not')

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

    data_cfg = {
        'vaihingen': {
            'class_rgb': {
                'IS': [255, 255, 255],  # Impervious surface
                'BD': [0, 0, 255],  # Buildings
                'LV': [0, 255, 255],  # Low vegetation
                'TR': [0, 255, 0],  # Tree
                'CR': [255, 255, 0],  # Car
                'BG': [255, 0, 0],  # Clutter/background
                'IG': [0, 0, 0],  # Ignore
            },
            'class_wo_encode': ['IG'],
            'class_wo_score': ['BG'],
            'mean': [0.46956375, 0.32316217, 0.3183202],
            'std': [0.22002102, 0.15978466, 0.15127575],
        },
    }

    model_cfg = {
        'SegResNet18': SegResNet18,
        'SegResNet34': SegResNet34,
        'SegResNet50': SegResNet50,
        'SegResNet101': SegResNet101,

        'SegGhostNet1p0': SegGhostNet1p0,
        'SegGhostNet1p3': SegGhostNet1p3,
        'SegGhostNet1p5': SegGhostNet1p5,

        'SegGhostNetDecouple1p0': SegGhostNetDecouple1p0,
        'SegGhostNetDecouple1p3': SegGhostNetDecouple1p3,
        'SegGhostNetDecouple1p5': SegGhostNetDecouple1p5,

        'SegGhostNetDecoupleScore1p0': SegGhostNetDecoupleScore1p0,
        'SegGhostNetDecoupleScore1p3': SegGhostNetDecoupleScore1p3,
        'SegGhostNetDecoupleScore1p5': SegGhostNetDecoupleScore1p5,

        'FCNs': FCNs,
        'FCN8s': FCN8s,
        'FCN16s': FCN16s,
        'FCN32s': FCN32s,

        'DeepLabV3plus': DeepLabV3plus,
        'DeepLabV3plusResNet50': DeepLabV3plusResNet50,
        'DeepLabV3plusResNet101': DeepLabV3plusResNet101,

        'PSPNetResNet50': PSPNetResNet50,
        'PSPNetResNet101': PSPNetResNet101,
        'PSPNetVGG16': PSPNetVGG16,
        'PSPNetVGG16BN': PSPNetVGG16BN,
        'PSPNetVGG19': PSPNetVGG19,
        'PSPNetVGG19BN': PSPNetVGG19BN,

        'SegNet': SegNet,
        'UNet': UNet,
    }

    if ',' in args.model:
        model_list = [model_cfg[m] for m in args.model.split(',')]
    else:
        model_list = [model_cfg[args.model]]

    if args.phase == 'train':
        train_dataset = SegDataset(
            root_path=os.path.join(args.data_root, 'train'),
            image_size=args.input_size,
            phase='train',
            chw_format=True,
            **data_cfg[args.dataset],
        )

        valid_dataset = SegDataset(
            root_path=os.path.join(args.data_root, 'test'),
            image_size=args.input_size,
            phase='valid',
            chw_format=True,
            **data_cfg[args.dataset],
        )

        assert len(train_dataset) > 0 and len(valid_dataset) > 0

        for model in model_list:
            trainer = TrainModule(
                model=model(num_class=len(train_dataset.clazz_encode)),
                args=args,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                weight=train_dataset.get_class_weight(upper_bound=1.0),
            )

            trainer.train_model()

    else:  # test

        test_dataset = SegDataset(
            root_path=os.path.join(args.data_root, 'test'),
            image_size=args.input_size,
            phase='test',
            chw_format=True,
            **data_cfg[args.dataset],
        )

        assert len(test_dataset) > 0

        for model in model_list:
            tester = TestModule(
                model=model(num_class=len(test_dataset.clazz_encode)),
                args=args,
                test_dataset=test_dataset,
            )

            tester.test_model()
