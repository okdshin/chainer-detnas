from __future__ import division
import argparse
import multiprocessing
import numpy as np

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer.optimizer import WeightDecay
from chainer.optimizers import CorrectedMomentumSGD
from chainer import training
from chainer.training import extensions

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.transforms import center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale
from chainer.training.extensions import LinearShift

from detnas import DetNASSmallCOCO

import chainermn

# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


class TrainTransform(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = random_sized_crop(img)
        img = resize(img, (224, 224))
        img = random_flip(img, x_random=True)
        img -= self.mean
        return img, label


class ValTransform(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = scale(img, 256)
        img = center_crop(img, (224, 224))
        img -= self.mean
        return img, label


def main():
    model_cfgs = {
        'detnas_small_coco': {
            'class': DetNASSmallCOCO,
            'score_layer_name': 'fc',
            'kwargs': {
                #'n_class': 1000
            }
        },
    }
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--trial', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--model',
        '-m',
        choices=model_cfgs.keys(),
        default='detnas_small_coco',
        help='Convnet models')
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument(
        '--batchsize', type=int, help='Batch size for each worker')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    label_names = directory_parsing_label_names(args.train)

    model_cfg = model_cfgs[args.model]
    extractor = model_cfg['class'](
        n_class=len(label_names), **model_cfg['kwargs'])
    extractor.pick = model_cfg['score_layer_name']
    model = Classifier(extractor)

    train_data = DirectoryParsingLabelDataset(args.train)
    val_data = DirectoryParsingLabelDataset(args.val)
    train_data = TransformDataset(train_data, TrainTransform(extractor.mean))
    val_data = TransformDataset(val_data, ValTransform(extractor.mean))
    print('finished loading dataset')

    train_indices = np.arange(len(train_data)//(100 if args.trial else 1))
    val_indices = np.arange(len(val_data))


    """
    train_data = train_data.slice[train_indices]
    val_data = val_data.slice[val_indices]
    """
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, args.batchsize, n_processes=args.loaderjob)
    val_iter = iterators.MultiprocessIterator(
        val_data,
        args.batchsize,
        repeat=False,
        shuffle=False,
        n_processes=args.loaderjob)

    optimizer = CorrectedMomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    for param in model.params():
        if param.name not in ('beta', 'gamma'):
            param.update_rule.add_hook(WeightDecay(args.weight_decay))

    if args.gpu != -1:
        model.to_gpu(args.gpu)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(LinearShift('lr', (args.lr, 0.0),
                   (0, len(train_indices) / args.batchsize)))
    evaluator = extensions.Evaluator(val_iter, model)
    trainer.extend(evaluator, trigger=(1, 'epoch'))

    log_interval = 0.1, 'epoch'
    print_interval = 0.1, 'epoch'

    trainer.extend(
        chainer.training.extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.snapshot_object(extractor,
                                   'snapshot_model_{.updater.epoch}.npz'),
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(
        extensions.PrintReport([
            'iteration', 'epoch', 'elapsed_time', 'lr', 'main/loss',
            'validation/main/loss', 'main/accuracy',
            'validation/main/accuracy'
        ]),
        trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
