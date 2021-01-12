# coding=utf-8

import os
import gc
import time
import torch
import importlib

from torchvision.datasets import ImageFolder

from nvidia import dali
from datasets.dali_pipeline import HybridTrainPipe, HybridValPipe, DaliIteratorGPU, DaliIteratorCPU

import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

"""
The pytorch demo to use DALI library

Adapted and modified from:
https://github.com/yaysummeriscoming/DALI_pytorch_demo
"""

__all__ = ['ImageNetDALI', 'DataPrefetchInCPU']


class ImageNetDALI(object):
    """
    Pytorch Dataloader, with torchvision or Nvidia DALI CPU/GPU pipelines.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip

    And ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size

    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    batch_size (int): how many samples per batch to load
    size (int): Output size (typically 224 for ImageNet)
    val_size (int): Validation pipeline resize size (typically 256 for ImageNet)
    workers (int): how many workers to use for data loading
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    prefetch_queue_depth (int, optional, default = 2) - Int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
    dali_cpu (bool): Use Nvidia DALI cpu backend, GPU backend otherwise
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32
    pin_memory_dali (bool): Transfer CPU tensor to pinned memory before transfer to GPU (dali only)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    """

    def __init__(self, data_dir, batch_size, size=224, val_batch_size=None, val_size=256, min_crop_size=0.08,
                 workers=4, world_size=1, prefetch_queue_depth=2, dali_cpu=True, fp16=False,
                 pin_memory_dali=False, mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                 std=(0.229 * 255, 0.224 * 255, 0.225 * 255)):

        self.batch_size = batch_size
        self.size = size
        self.val_batch_size = val_batch_size
        self.min_crop_size = min_crop_size
        self.workers = workers
        self.world_size = world_size
        self.prefetch_queue_depth = prefetch_queue_depth
        self.dali_cpu = dali_cpu
        self.fp16 = fp16
        self.pin_memory_dali = pin_memory_dali
        self.mean = mean
        self.std = std

        self.val_size = val_size
        if self.val_size is None:
            self.val_size = self.size

        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        if self.world_size > 1:
            raise NotImplementedError('distributed support not tested yet...')
            # self.train_sampler = DistributedSampler(train_dataset)
            # self.val_sampler = DistributedSampler(val_dataset)

        # Data loading code
        self.train_dir = os.path.join(data_dir, 'train')
        self.valid_dir = os.path.join(data_dir, 'val')

        assert len(ImageFolder(self.valid_dir)) % self.val_batch_size == 0, \
            'Validation batch size must divide validation dataset size cleanly...  DALI has problems otherwise.'

        # Init train and valid sampler
        self.train_sampler = None
        self.valid_sampler = None

        # Init train and valid pipe
        self.train_pipe = None
        self.valid_pipe = None

        # Init train and valid loader
        self.train_loader = None
        self.valid_loader = None

    @staticmethod
    def clear_memory(verbose=False):
        """Clear memory"""
        stt = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
        gc.collect()
        if verbose:
            print('Cleared memory. Time taken was %f secs' % (time.time() - stt))

    def build_train_loader(self):
        """Build train loader by DALI"""
        assert self.world_size == 1, 'Distributed support not tested yet'

        iterator = DaliIteratorCPU if self.dali_cpu else DaliIteratorGPU

        self.train_pipe = HybridTrainPipe(batch_size=self.batch_size, num_threads=self.workers, device_id=0,
                                          data_dir=self.train_dir, crop=self.size, dali_cpu=self.dali_cpu,
                                          mean=self.mean, std=self.std, local_rank=0,
                                          prefetch_queue_depth=self.prefetch_queue_depth,
                                          world_size=self.world_size, shuffle=True, fp16=self.fp16,
                                          min_crop_size=self.min_crop_size)

        self.train_pipe.build()
        self.train_loader = iterator(pipelines=self.train_pipe,
                                     size=self.get_num_train() / self.world_size,
                                     fp16=self.fp16, mean=self.mean, std=self.std,
                                     pin_memory=self.pin_memory_dali)

        return self.train_loader

    def release_train_loader(self):
        """Release DALI train loader"""
        self.clear_memory()

        # Currently we need to delete & rebuild the dali pipeline every epoch,
        # due to a memory leak somewhere in DALI
        print('Release DALI train loader to reduce memory usage')
        del self.train_loader, self.train_pipe
        self.clear_memory()

        # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
        importlib.reload(dali)
        from datasets.dali_pipeline import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

    def build_valid_loader(self):
        """Build valid loader by DALI"""
        assert self.world_size == 1, 'Distributed support not tested yet'

        iterator = DaliIteratorCPU if self.dali_cpu else DaliIteratorGPU

        self.valid_pipe = HybridValPipe(batch_size=self.val_batch_size, num_threads=self.workers, device_id=0,
                                        data_dir=self.valid_dir, crop=self.size, size=self.val_size,
                                        prefetch_queue_depth=self.prefetch_queue_depth, dali_cpu=self.dali_cpu,
                                        mean=self.mean, std=self.std, local_rank=0,
                                        world_size=self.world_size, shuffle=False, fp16=self.fp16)

        self.valid_pipe.build()
        self.valid_loader = iterator(pipelines=self.valid_pipe,
                                     size=self.get_num_valid() / self.world_size,
                                     fp16=self.fp16, mean=self.mean, std=self.std,
                                     pin_memory=self.pin_memory_dali)
        return self.valid_loader

    def release_valid_loader(self):
        """Release DALI train loader"""
        self.clear_memory()

        # Currently we need to delete & rebuild the dali pipeline every epoch,
        # due to a memory leak somewhere in DALI
        print('Release DALI valid loader to reduce memory usage')
        del self.valid_loader, self.valid_pipe
        self.clear_memory()

        # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
        importlib.reload(dali)
        from datasets.dali_pipeline import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

    def get_num_train(self):
        """Number of training examples"""
        return int(self.train_pipe.epoch_size("Reader"))

    def get_num_valid(self):
        """Number of validation examples"""
        return int(self.valid_pipe.epoch_size("Reader"))

    def get_classes(self):
        """Classes in the dataset - as indicated by the validation dataset"""
        return len(ImageFolder(self.valid_dir).classes)


class DataPrefetchInCPU(threading.Thread):
    """
    This function transforms generator into a background-thead generator (Copy tensor to CPU to decrease
    the memory consumption of GPU).

    Note that the iterator pauses output when the current queue length is less than `hold_threshold`
    until the queue is refilled to the size of `max_prefetch` or all remaining data is loaded into the queue

    Params:
        generator: generator or genexp or any. It can be used with any mini-batch generator.
        max_prefetch: int. Defines how many iterations (at most) can background generator keep stored.
        hold_threshold: int. Threshold for iterators to temporarily hold (enter the wait state)
    """

    def __init__(self, generator, max_prefetch=16, hold_threshold=5):
        threading.Thread.__init__(self)

        assert max_prefetch > hold_threshold >= 0

        self.max_prefetch = max_prefetch
        self.hold_threshold = hold_threshold
        self.current_iter = 0

        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for i, item in enumerate(self.generator):
            if isinstance(item, list):
                item = [x.cpu() for x in item]
            elif isinstance(item, dict):
                for k in item.keys():
                    item[k] = item[k].cpu()
            else:
                raise AttributeError('Unknown item, expect list, dict, but input type: {}'.format(type(item)))

            self.queue.put(item)
            self.current_iter = i
        self.queue.put(None)

    def next(self):
        # To maximum the GPUs' performance
        # print('current qsize: {}'.format(self.queue.qsize()))
        while self.queue.qsize() < self.hold_threshold:
            if len(self) - self.current_iter - 1 >= self.max_prefetch - self.queue.qsize():
                while self.queue.qsize() < self.max_prefetch:
                    time.sleep(0.1)  # Hold until the queue is full
            else:
                break

        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        self.queue.task_done()  # It's important to notify queue the task `get` is done
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)
