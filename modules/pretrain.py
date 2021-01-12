# coding=utf-8

from datasets.dali_imagenet import ImageNetDALI
from modules.train import BasicTrainModule
from utils import calc_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import loss as ls
import torch
import json
import os


class PretrainModule(BasicTrainModule):
    """
    The module which is used for the pretrain model on `ImageNet` dataset

    It should be noted that the pre-training module uses the DALI library to accelerate
    data reading and data enhancement. In order to ensure the efficiency of model training,
    you must have more than one GPU. If you have only one GPU, it is recommended that your
    batch should not be set too large (for example, 256). If you have 3 or 4 GPUs, it is
    recommended that you use GPU No.0 for DALI to accelerate data reading and data enhancement,
    and the remaining cards are used for training

    Params:
        model: nn.Module. The `ConvNet` model for pretraining.
        args: parser.parse_args. The other custom arguments.
        weight: None or numpy array (default None). The weight of each classes.
    """

    def __init__(self, model, args, weight=None):
        super(PretrainModule, self).__init__(model, args)

        self.weight = weight

        # Get score state
        self.train_score = calc_score.ClassificationScore(num_class=self.args.num_class)
        self.valid_score = calc_score.ClassificationScore(num_class=self.args.num_class)

        # Get losses for training
        self.lsce = ls.LabelSmoothCrossEntropy(epsilon=0.1, batch_weight=False)

    def prepare_dir(self):
        """Prepare and make needed directories"""
        self.model_name = self.model.__class__.__name__

        self.output_root = os.path.join(
            self.args.output_root, self.model_name, self.args.dataset,
            'lr_{}_wd_{}_sd_{}'.format(
                self.args.init_lr, self.args.weight_decay, self.args.seed
            ),
        )

        self.save_weight = os.path.join(self.output_root, 'weights')
        self.save_log = os.path.join(self.output_root, 'logs')
        self.save_records = os.path.join(self.output_root, 'records')
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.save_weight, exist_ok=True)
        os.makedirs(self.save_log, exist_ok=True)
        os.makedirs(self.save_records, exist_ok=True)

        # Prepare other directories
        self.save_pretrained = os.path.join(self.output_root, 'pretrained')
        os.makedirs(self.save_pretrained, exist_ok=True)

    def prepare_device(self):
        """Prepare and make needed device"""
        # Get `device` and `device ids`
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                if self.args.dali_gpu is True:
                    self.device = 'cuda:1'  # GPU(cuda:0) will be used by DALI
                    self.device_ids = [i for i in range(len(self.args.gpus.split(',')))][1:]
                else:
                    self.device = 'cuda'  # default 'cuda:0'
                    self.device_ids = [i for i in range(len(self.args.gpus.split(',')))]
            else:
                self.device = 'cuda'  # default 'cuda:0'
                self.device_ids = [0]
        else:
            self.device = 'cpu'  # default 'cpu'
            self.device_ids = None
            raise RuntimeError('You must have more than one GPU to pretrain on a huge dataset like ImageNet ...')

    def prepare_optimizer(self):
        """Prepare and make needed optimizer"""
        super(PretrainModule, self).prepare_optimizer()

        # Get the `scheduler` for `MultiStepLR`
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.args.milestones.split(','),
            gamma=self.args.milestones_gamma
        )

    def run_epoch(self, dataloader, epoch, for_train=True):
        """
        Train or test for one epoch and get the scores and losses

        Params:
            dataloader: DataLoader. A data iterator for training or testing
            epoch: int. Current epoch
            for_train: bool (default True). Indicates whether to use for training
                (gradient enabled) or test (gradient disabled)
        """
        if for_train:
            self.model.train()
            score_state = self.train_score
            epoch_tag = 'Train'
        else:
            self.model.eval()
            score_state = self.valid_score
            epoch_tag = 'Valid'

        score_state.reset()

        epoch_loss = dict()
        epoch_score = dict()

        with torch.enable_grad() if for_train else torch.no_grad():

            bar = tqdm(dataloader)
            for i, (img, truth) in enumerate(bar):

                # To cuda if GPU is available
                if torch.cuda.is_available():
                    img = img.to(torch.device(self.device))
                    truth = truth.to(torch.device(self.device))

                # Get predict results and losses
                pred = self.model(img)
                loss = self.lsce(pred, truth)

                # Update score state
                score_state.update(pred.cpu().detach().numpy(), truth.cpu().detach().numpy())

                # Set description
                bar.set_description("[{}] Epoch: {}/{}".format(epoch_tag, epoch + 1, self.args.num_epoch))

                # Update loss
                if 'label_smooth' not in epoch_loss.keys():
                    epoch_loss['label_smooth'] = loss.cpu().detach().numpy()
                else:
                    epoch_loss['label_smooth'] += loss.cpu().detach().numpy()

                # If for training
                if for_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

            # Get loss of this epoch
            for k in epoch_loss.keys():
                epoch_loss[k] = epoch_loss[k] / len(dataloader)

            # Get score of this epoch
            epoch_score['top1_acc'] = score_state.get_top1_acc()
            epoch_score['top5_acc'] = score_state.get_top5_acc()

        return epoch_loss, epoch_score

    def train_model(self):
        """
        Train model
        """
        # Create a DALI dataset
        dataset = ImageNetDALI(
            data_dir=self.args.data_root,
            batch_size=self.args.train_batch,
            val_batch_size=self.args.valid_batch,  # Validation batch size must divide validation dataset size cleanly
            size=self.args.train_size,
            val_size=self.args.valid_size,
            workers=self.args.num_workers,
            prefetch_queue_depth=self.args.prefetch,
            dali_cpu=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # Print key information
        print('[Train] model: {}, init lr: {:5.5f}, current lr: {:5.5f}, seed: {}'.format(
            self.model_name, self.args.init_lr, self.learning_rate, self.args.seed))

        # Create a `SummaryWriter` object
        writer = SummaryWriter(self.save_log)

        # Create or load record as `dict()`
        json_path = os.path.join(self.save_records, '{}.json'.format(self.model_name))
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                records = json.load(f)
        else:
            records = dict()

        # Start training
        best_top1, best_top1_epoch = 0, 0
        best_top5, best_top5_epoch = 0, 0
        for epoch in range(self.start_epoch, self.args.num_epoch):

            # Add learning rate to scalars
            writer.add_scalar('lr/last_lr', self.scheduler.get_last_lr(), epoch)

            # Reset score matrix
            self.train_score.reset()
            self.valid_score.reset()

            # Run epoch
            dataloader = dataset.build_train_loader()
            train_loss, train_score = self.run_epoch(dataloader, epoch, for_train=True)
            dataset.release_train_loader()

            dataloader = dataset.build_valid_loader()
            valid_loss, valid_score = self.run_epoch(dataloader, epoch, for_train=False)
            dataset.release_valid_loader()

            # Add loss and scores to scalars
            for k in train_loss.keys():
                if isinstance(train_loss[k], np.float or np.int):
                    writer.add_scalars('loss/%s' % k, {'train': train_loss[k], 'valid': valid_loss[k]}, epoch)

            for k in train_score.keys():
                if isinstance(train_score[k], np.float or np.int):
                    writer.add_scalars('score/%s' % k, {'train': train_score[k], 'valid': valid_score[k]}, epoch)

            # Flush summary writer
            writer.flush()

            # Add records and save it
            for k in train_loss.keys():
                if k not in records.keys():
                    records[k] = {'train': [train_loss[k]], 'valid': [valid_loss[k]]}
                else:
                    records[k]['train'].append(train_loss[k])
                    records[k]['valid'].append(valid_loss[k])

            for k in train_score.keys():
                if k not in records.keys():
                    records[k] = {'train': [train_score[k]], 'valid': [valid_score[k]]}
                else:
                    records[k]['train'].append(train_score[k])
                    records[k]['valid'].append(valid_score[k])

            with open(json_path, 'w') as f:
                json.dump(records, f)

            # Print key information
            info = '[Valid] top1_acc: {:5.5f}, top5_acc: {:5.5f}, best top1_acc: {:5.5f}/{}, best top5_acc: {:5.5f}/{}'
            print(info.format(self.valid_score.get_top1_acc(), self.valid_score.get_top5_acc(),
                              best_top1, best_top1_epoch + 1, best_top5, best_top5_epoch + 1))
            self.train_score.pt_score(['top1_acc', 'top5_acc'], label='Train')
            self.valid_score.pt_score(['top1_acc', 'top5_acc'], label='Valid')

            # Save last model
            self.save_model(os.path.join(self.save_weight, 'epoch_last.pth'), epoch)

            # Save model according to best `top1_acc`
            if valid_score['top1_acc'] > best_top1:
                best_top1 = valid_score['top1_acc']
                best_top1_epoch = epoch

                # Save model as well as optimizer, epoch
                self.save_model(
                    path=os.path.join(self.save_weight, 'epoch_{}_top1_acc_{:4.4f}.pth'.format(epoch + 1, best_top1)),
                    epoch=epoch)

                # Just save model's `state_dict`
                if isinstance(self.model, torch.nn.DataParallel):
                    checkpoint = self.model.module.state_dict()
                else:
                    checkpoint = self.model.state_dict()
                save_stat_dict = os.path.join(self.save_pretrained, self.model_name + '_top1_{}.pth'.format(best_top1))
                torch.save(checkpoint, save_stat_dict)

        # Close summary
        writer.close()
