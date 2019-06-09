import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links.connection import Conv2DBNActiv
from chainercv.links.connection import SeparableConv2DBNActiv
from chainercv.links import PickableSequentialChain


def channel_shuffle(x, groups):
    batch, channels, height, width = x.shape
    assert channels % groups == 0
    channels_per_group = channels // groups
    x = F.reshape(x, shape=(batch, groups, channels_per_group, height, width))
    x = F.swapaxes(x, axis1=1, axis2=2)
    x = F.reshape(x, shape=(batch, channels, height, width))
    return x


class ShuffleUnit(chainer.Chain):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=None,
                 xception=False,
                 down_sample=False):
        super(ShuffleUnit, self).__init__()
        self.xception = xception
        self.down_sample = down_sample
        mid_channels = out_channels // 2
        with self.init_scope():
            if down_sample:
                self.downsize_depthwise = L.DepthwiseConvolution2D(
                    in_channels,
                    channel_multiplier=1,
                    ksize=ksize,
                    stride=2,
                    pad=ksize // 2)
                self.downsize_depthwise_bn = L.BatchNormalization(
                    size=in_channels)
                self.downsize_pointwise = Conv2DBNActiv(mid_channels, 1)
            if xception:
                self.separable_conv1 = SeparableConv2DBNActiv(
                    in_channels=(in_channels if down_sample else mid_channels),
                    out_channels=mid_channels,
                    ksize=ksize,
                    pad=ksize // 2,
                    stride=(2 if self.down_sample else 1))
                self.separable_conv2 = SeparableConv2DBNActiv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    ksize=ksize,
                    pad=ksize // 2)
                self.separable_conv3 = SeparableConv2DBNActiv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    ksize=ksize,
                    pad=ksize // 2)
            else:
                self.compress = Conv2DBNActiv(mid_channels, 1)
                self.depthwise = L.DepthwiseConvolution2D(
                    mid_channels,
                    channel_multiplier=1,
                    ksize=ksize,
                    pad=ksize // 2,
                    stride=(2 if self.down_sample else 1))
                self.depthwise_bn = L.BatchNormalization(size=mid_channels)
                self.expand = Conv2DBNActiv(mid_channels, 1)

    def __call__(self, x):
        if self.down_sample:
            left = self.downsize_depthwise_bn(self.downsize_depthwise(x))
            left = self.downsize_pointwise(left)
            right = x
        else:
            left, right = F.split_axis(x, indices_or_sections=2, axis=1)
        if self.xception:
            right = self.separable_conv1(right)
            right = self.separable_conv2(right)
            right = self.separable_conv3(right)
        else:
            right = self.compress(right)
            right = self.depthwise_bn(self.depthwise(right))  # No relu
            right = self.expand(right)
        y = F.concat((left, right), 1)
        y = channel_shuffle(y, groups=2)
        return y


_imagenet_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


class DetNASSmallCOCO(PickableSequentialChain):
    def __init__(self, n_class):
        self.mean = _imagenet_mean

        super(DetNASSmallCOCO, self).__init__()
        self.n_class = n_class
        self.blocks = [
            (7, 5, 7, 3),  # stage1
            (7, 5, 5, 7),  # stage2
            ('xception', 'xception', 5, 'xception', 3, 3, 'xception',
             5),  # stage3
            ('xception', 5, 'xception', 7)  # stage4
        ]
        channels_list = [64, 160, 320, 640]

        with self.init_scope():
            self.conv = Conv2DBNActiv(16, 3, pad=1, stride=2)
            before_channels = 16
            for stage_id, (channels, block) in enumerate(
                    zip(channels_list, self.blocks)):
                for block_id, ksize_or_xception in enumerate(block):
                    layer_name = 'stage{sid}_block{bid}'.format(
                        sid=stage_id, bid=block_id)
                    if ksize_or_xception == 'xception':
                        setattr(self, layer_name,
                                ShuffleUnit(
                                    before_channels,
                                    channels,
                                    3,
                                    down_sample=(block_id == 0),
                                    xception=True))
                    else:  # ksize_or_xception is ksize
                        setattr(
                            self,
                            layer_name,
                            ShuffleUnit(
                                before_channels,
                                channels,
                                ksize_or_xception,  # = ksize
                                down_sample=(block_id == 0)))
                    before_channels = channels
            self.global_pool = lambda x : F.average_pooling_2d(x, ksize=(x.shape[2], x.shape[3]))
            if self.n_class is not None:
                self.fc = L.Linear(self.n_class)
