# coding=utf-8

from utils.calc_score import SegmentationScore
from utils.data_convert import arr_to_str
from torch.utils.data import DataLoader
from modules.basic import BasicModule
from PIL import Image
from tqdm import tqdm
from time import time
import numpy as np
import torch
import os

__all__ = ['TestModule']


class TestModule(BasicModule):
    """
    The module which is used for the testing dataset

    Params:
        model: nn.Module. The `ConvNet` model for training.
        args: parser.parse_args. The other custom arguments.
        test_dataset: torch.utils.data.Dataset. The dataset for testing
    """

    def __init__(self, model, args, test_dataset):
        super(TestModule, self).__init__(model, args)

        self.test_dataset = test_dataset

        # Get score matrices
        self.test_score = SegmentationScore(clazz_encode=self.test_dataset.clazz_encode)

    def prepare_dir(self):
        """Prepare and make needed directories"""
        super(TestModule, self).prepare_dir()

        self.save_weight = os.path.join(self.output_root, 'weights')
        self.save_images = os.path.join(self.output_root, 'images')
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.save_weight, exist_ok=True)
        os.makedirs(self.save_images, exist_ok=True)

    def test_model(self):
        """
        Test model
        """
        # Prepare model and score matrix
        self.model.eval()
        self.test_score.reset()

        # Create dataloader for testing
        dataloader = DataLoader(
            dataset=self.test_dataset,
            num_workers=self.args.num_workers,
            batch_size=1,
            shuffle=False
        )

        # Eval without grad
        with torch.no_grad():

            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):

                # Get file names
                file_name = truth['fnm']

                # To cuda if GPU is available
                if torch.cuda.is_available():
                    for k in truth.keys():
                        truth[k] = truth[k].to(torch.device(self.device))

                # Get and process predict results
                start_t = time()
                pred = self.model(truth['img'])
                end_t = time()

                fps = 1 / (end_t - start_t)
                bar.set_description("[{}] Model: {}, FPS: {:3.3}".format('Test', self.model_name, fps))

                for k in pred.keys():
                    if k == 'lbl':
                        y, z = truth[k], torch.argmax(pred[k], dim=1)
                        y, z = y.cpu().detach().numpy(), z.cpu().detach().numpy()

                        # Update score matrix
                        self.test_score.update(z, y)

                        # Get save path for `lbl`
                        save_path = os.path.join(self.save_images, k)
                        os.makedirs(save_path, exist_ok=True)

                        # Save image
                        for i in range(z.shape[0]):
                            z_dec = dataloader.dataset.decode_lbl(z[i])
                            if dataloader.dataset.chw_format:
                                z_dec = z_dec.transpose((1, 2, 0)).astype(np.uint8)
                            Image.fromarray(z_dec).save(os.path.join(save_path, arr_to_str(file_name[i]) + '.png'))

                    elif k == 'bdy':
                        z = torch.argmax(pred[k], dim=1).cpu().detach().numpy()

                        # Get save path for `bdy`
                        save_path = os.path.join(self.save_images, k)
                        os.makedirs(save_path, exist_ok=True)

                        # Save image
                        for i in range(z.shape[0]):
                            z_dec = dataloader.dataset.decode_lbl(z[i])
                            if dataloader.dataset.chw_format:
                                z_dec = z_dec.transpose((1, 2, 0)).astype(np.uint8)
                            Image.fromarray(z_dec).save(os.path.join(save_path, arr_to_str(file_name[i]) + '.png'))

                    elif k == 'edg':
                        y, z = truth[k], pred[k + '_sgm']
                        y, z = y.cpu().detach().numpy(), z.cpu().detach().numpy()

                        # Get save path for `bdy`
                        save_truth_path = os.path.join(self.save_images, k, 'truth')
                        save_pred_path = os.path.join(self.save_images, k, 'pred')
                        os.makedirs(save_truth_path, exist_ok=True)
                        os.makedirs(save_pred_path, exist_ok=True)

                        # Save image
                        for i in range(z.shape[0]):
                            y_dec, z_dec = (y[i] * 255).astype(np.uint8), (z[i] * 255).astype(np.uint8)
                            Image.fromarray(y_dec).save(
                                os.path.join(save_truth_path, arr_to_str(file_name[i])) + '.png')
                            Image.fromarray(z_dec).save(
                                os.path.join(save_pred_path, arr_to_str(file_name[i])) + '.png')

        # Save score matrix to file
        with open(os.path.join(self.output_root, 'test.txt'), 'w') as f:
            f.write('[Test] Model: {}, Optimizer: {}, Seed: {}\n'.format(
                self.model_name, self.args.optimizer, self.args.seed
            ))
            f.write('\n')

            # Output fusion matrix
            f.write('[Test] The fusion matrix is\n')
            f.write(str(self.test_score.pt_confusion_matrix(show=False)))
            f.write('\n\n')

            # Scores for all categories
            f.write('[Test] The scores for all categories\n')
            f.write('[Test] OA: {:4.4f}, mCA: {:4.4f}, mIoU: {:4.4f}, mF1: {:4.4f}\n'.format(
                self.test_score.get_OA(),
                np.nanmean(self.test_score.get_CA()),
                np.nanmean(self.test_score.get_IoU()),
                np.nanmean(self.test_score.get_F1()),
            ))
            f.write(str(self.test_score.pt_score(['CA', 'IoU', 'F1'], show=False)))
            f.write('\n\n')

            # Scores for all categories except ignored ones.
            f.write('[Test] The scores for all categories except: {}\n'.format(
                ', '.join(self.test_dataset.class_wo_score)
            ))
            f.write('[Test] OA: {:4.4f}, mCA: {:4.4f}, mIoU: {:4.4f}, mF1: {:4.4f}\n'.format(
                self.test_score.get_OA(self.test_dataset.class_wo_score),
                np.nanmean(self.test_score.get_CA(self.test_dataset.class_wo_score)),
                np.nanmean(self.test_score.get_IoU(self.test_dataset.class_wo_score)),
                np.nanmean(self.test_score.get_F1(self.test_dataset.class_wo_score)),
            ))
            f.write(
                str(self.test_score.pt_score(['CA', 'IoU', 'F1'], ignore=self.test_dataset.class_wo_score, show=False))
            )

        # Print saved information
        with open(os.path.join(self.output_root, 'test.txt'), 'r') as f:
            print(f.read())
