import math

import audobject
import audtorch
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchvision

import numpy as np
from torch.autograd import Variable
import numpy
from tools import augmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from efficientnet_pytorch import EfficientNet
from models.util import init_layer, do_mixup, init_bn
from scipy.signal import butter, filtfilt
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation
from torch.hub import load_state_dict_from_url
import urllib.parse



# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# ========================================= CNNs =======================================================
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# PANNs / ResNets : https://github.com/qiuqiangkong/audioset_tagging_cnn

# The MIT License

# Copyright (c) 2018-2020 Qiuqiang Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


""" PANNs (Pretrained Audio Neural Networks) - https://github.com/qiuqiangkong/audioset_tagging_cnn"""


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Sequential pass through 2 cnns
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), bias=False)

        self.conv2 = torch.nn.Conv2d(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1), bias=False)

        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm2d(self.out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn6(nn.Module, audobject.Object):
    """
            ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
               BatchNorm2d-1           [-1, 64, 500, 1]             128
                    Conv2d-2          [-1, 64, 500, 64]           1,600
               BatchNorm2d-3          [-1, 64, 500, 64]             128
              ConvBlock5x5-4          [-1, 64, 250, 32]               0
                    Conv2d-5         [-1, 128, 250, 32]         204,800
               BatchNorm2d-6         [-1, 128, 250, 32]             256
              ConvBlock5x5-7         [-1, 128, 125, 16]               0
                    Conv2d-8         [-1, 256, 125, 16]         819,200
               BatchNorm2d-9         [-1, 256, 125, 16]             512
             ConvBlock5x5-10           [-1, 256, 62, 8]               0
                   Conv2d-11           [-1, 512, 62, 8]       3,276,800
              BatchNorm2d-12           [-1, 512, 62, 8]           1,024
             ConvBlock5x5-13           [-1, 512, 31, 4]               0
                   Linear-14                  [-1, 512]         262,656
                   Linear-15                    [-1, 1]             513
        ================================================================
        Total params: 4,567,617
        Trainable params: 4,567,617
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.12
        Forward/backward pass size (MB): 66.12
        Params size (MB): 17.42
        Estimated Total Size (MB): 83.67
        ----------------------------------------------------------------

    """
    def __init__(
            self,
            output_dim: int,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):
        super(Cnn6, self).__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.out = nn.Linear(512, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Cnn10(torch.nn.Module, audobject.Object):
    """
            ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
               BatchNorm2d-1           [-1, 64, 500, 1]             128
                    Conv2d-2          [-1, 64, 500, 64]             576
               BatchNorm2d-3          [-1, 64, 500, 64]             128
                    Conv2d-4          [-1, 64, 500, 64]          36,864
               BatchNorm2d-5          [-1, 64, 500, 64]             128
                 ConvBlock-6          [-1, 64, 250, 32]               0
                    Conv2d-7         [-1, 128, 250, 32]          73,728
               BatchNorm2d-8         [-1, 128, 250, 32]             256
                    Conv2d-9         [-1, 128, 250, 32]         147,456
              BatchNorm2d-10         [-1, 128, 250, 32]             256
                ConvBlock-11         [-1, 128, 125, 16]               0
                   Conv2d-12         [-1, 256, 125, 16]         294,912
              BatchNorm2d-13         [-1, 256, 125, 16]             512
                   Conv2d-14         [-1, 256, 125, 16]         589,824
              BatchNorm2d-15         [-1, 256, 125, 16]             512
                ConvBlock-16           [-1, 256, 62, 8]               0
                   Conv2d-17           [-1, 512, 62, 8]       1,179,648
              BatchNorm2d-18           [-1, 512, 62, 8]           1,024
                   Conv2d-19           [-1, 512, 62, 8]       2,359,296
              BatchNorm2d-20           [-1, 512, 62, 8]           1,024
                ConvBlock-21           [-1, 512, 31, 4]               0
                   Linear-22                  [-1, 512]         262,656
                   Linear-23                    [-1, 1]             513
        ================================================================
        Total params: 4,949,441
        Trainable params: 4,949,441
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.12
        Forward/backward pass size (MB): 124.69
        Params size (MB): 18.88
        Estimated Total Size (MB): 143.69
        ----------------------------------------------------------------
    """
    def __init__(
            self,
            output_dim: int,
            sigmoid_output: bool = False,
            segmentwise: bool = False,
            augment: bool = False
    ):

        super().__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise
        self.augment = augment

        self.bn0 = torch.nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        self.out = torch.nn.Linear(512, output_dim, bias=True)

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = augmentation.FilterAugment()

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):

        if self.train and self.augment:
            # x = self.spec_augmenter(x)
            # x = self.filter_augmenter(x)
            pass

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Cnn14(torch.nn.Module, audobject.Object):
    r"""Cnn14 model architecture.

    Args:
        sampling_rate: feature extraction is configurable
            based on sampling rate
        output_dim: number of output classes to be used
        sigmoid_output: whether output should be passed through
            a sigmoid. Useful for multi-label problems
        segmentwise: whether output should be returned per-segment
            or aggregated over the entire clip
        in_channels: number of input channels
    """

    def __init__(
            self,
            output_dim: int,
            sigmoid_output: bool = False,
            segmentwise: bool = False,
            in_channels: int = 1
    ):

        super().__init__()

        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise
        self.in_channels = in_channels

        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = torch.nn.Linear(2048, 2048, bias=True)
        self.out = torch.nn.Linear(2048, self.output_dim, bias=True)

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = augmentation.FilterAugment()

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = self.spec_augmenter(x)  # default augmentation

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Cnn14_DecisionLevelMax(nn.Module, audobject.Object):
    """designed for SED"""
    def __init__(self, classes_num):
        super(Cnn14_DecisionLevelMax, self).__init__()
        self.classes_num = classes_num

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        # frames_num = x.shape[2]

        #print(input.shape)

        if input.ndim == 3:
            input = input.unsqueeze(1)
            # print('unsqueezing input:', input.shape)

        #print(input.shape)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        #framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        #framewise_output = pad_framewise_output(framewise_output, frames_num)

        # output_dict = {'framewise_output': framewise_output,
        #               'clipwise_output': clipwise_output}

        return clipwise_output


# NMELS 128:

class Cnn10M128(torch.nn.Module, audobject.Object):
    def __init__(
            self,
            output_dim: int,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):

        super().__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = torch.nn.BatchNorm2d(128)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=128)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block4 = ConvBlock(in_channels=512, out_channels=1024)

        self.fc1 = torch.nn.Linear(1024, 1024, bias=True)
        self.out = torch.nn.Linear(1024, output_dim, bias=True)

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        ).train()  # 2 2
        self.filter_augmenter = augmentation.FilterAugment().train()
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):

        # if self.training:
        #     # x = self.spec_augmenter(x)
        #     x = self.filter_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


""" WAVEGRAM-CNNs """


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        return x


class WavegramCnn14(nn.Module, audobject.Object):
    def __init__(
            self,
            sample_rate=16000,
            window_size=500,
            hop_size=250,
            mel_bins=64,
            fmin=50,
            fmax=8192,
            classes_num=1,
            sigmoid_output=False
    ):
        super(WavegramCnn14, self).__init__()

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.classes_num = classes_num
        self.sigmoid_output = sigmoid_output

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))  # torch.Size([16, 64, 16000])
        a1 = self.pre_block1(a1, pool_size=4)  # torch.Size([16, 64, 4000])
        a1 = self.pre_block2(a1, pool_size=4)  # torch.Size([16, 128, 1000])
        a1 = self.pre_block3(a1, pool_size=4)  # torch.Size([16, 128, 250])
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)  # torch.Size([16, 4, 250, 32])
        a1 = self.pre_block4(a1, pool_size=(2, 1))  # torch.Size([16, 64, 125, 32])

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            a1 = do_mixup(a1, mixup_lambda)

        x = a1
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)
        if self.sigmoid_output:
            clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        #  return output_dict
        return output_dict['clipwise_output']


class WavegramLogmelCnn14(nn.Module, audobject.Object):
    def __init__(
            self,
            sample_rate=16000,
            tdim=5,
            window_size=320,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8192,
            classes_num=1,
            sigmoid_output=False,
            augment=False
    ):
        super(WavegramLogmelCnn14, self).__init__()
        # parameters
        self.sample_rate = sample_rate
        self.tdim = tdim
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.classes_num = classes_num
        self.sigmoid_output = sigmoid_output
        self.augment = augment

        # first conv block (one of 2 ways):
        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=5, padding=5, bias=False)  # kernel_size = 11 initially
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # concatenated result (conv(+)waveform) goes in main conv block:
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        self.bn0 = nn.BatchNorm2d(64)

        # augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)
        self.filter_augmenter = augmentation.FilterAugment()


        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        data_length should be Sample_rate * wdw_size
        """
        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=2)  # pool_size = 4 initially
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))  # torch.Size([16, 64, 125, 32])

        # # Log mel spectrogram
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,  # 16000
            win_length=self.window_size, # 320, hop default to 160
            n_fft=self.window_size,  # 320
            f_min=self.fmin,  # 50
            f_max=self.fmax,  # 8000
            n_mels=self.mel_bins,  # 64
        )
        x = transform(input.cpu())
        x = x.permute(0, 2, 1)
        x = x[:, None, :]
        stft_crop_len = self.tdim * 100 # roughly equals ms conversion with our stft settings...
        transform = audtorch.transforms.RandomCrop(stft_crop_len, axis=-2) # not needed?
        x = transform(x).cuda()

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.augment == 'sa':
            x = self.spec_augmenter(x)
        elif self.augment == 'fa':
            x = self.filter_augmenter(x)
            
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        # Concatenate Wavegram and Log mel spectrogram along the channel dimension

        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)
        if self.sigmoid_output:
            clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        # return output_dict
        return output_dict['clipwise_output']


""" RESIDUAL-NETS """


def _resnet_conv1x1(in_planes, out_planes):
    # 1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def _resnet_conv3x3(in_planes, out_planes):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet22(nn.Module, audobject.Object):
    def __init__(
            self,
            output_dim,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):
        super(ResNet22, self).__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[2, 2, 2, 2], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class ResNet38(nn.Module, audobject.Object):
    def __init__(
            self,
            output_dim,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):
        super(ResNet38, self).__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class ResNet54(nn.Module, audobject.Object):
    def __init__(
            self,
            output_dim,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):
        super(ResNet54, self).__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBottleneck, layers=[3, 4, 6, 3], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=2048, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class MobileNetV1(nn.Module, audobject.Object):
    def __init__(self, classes_num=1, sigmoid_output=False):

        super(MobileNetV1, self).__init__()
        self.classes_num = classes_num
        self.sigmoid_output = sigmoid_output

        # window = 'hann'
        # center = True
        # pad_mode = 'reflect'
        # ref = 1.0
        # amin = 1e-10
        # top_db = None
        #
        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
        #                                          win_length=window_size, window=window, center=center,
        #                                          pad_mode=pad_mode,
        #                                          freeze_parameters=True)
        #
        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
        #                                          n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
        #                                          top_db=top_db,
        #                                          freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.features = nn.Sequential(
            conv_bn(1, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.features(x)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)
        if self.sigmoid_output:
            clipwise_output = torch.sigmoid(self.fc_audioset(x))

        # output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return clipwise_output


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module, audobject.Object):
    def __init__(self, classes_num=1, sigmoid_output=False):
        super(MobileNetV2, self).__init__()
        self.classes_num = classes_num
        self.sigmoid_output = sigmoid_output
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        width_mult = 1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.features(x)

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.fc_audioset(x)
        if self.sigmoid_output:
            clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return clipwise_output


""" PSLA: https://github.com/YuanGongND/psla/blob/main/src/models/Models.py """

def init_layer2(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()


    def init_weights(self):
        init_layer2(self.att)
        init_layer2(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x, norm_att


class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(MeanPooling, self).__init__()

        self.cla_activation = cla_activation

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer2(self.cla)

    def activate(self, x, activation):
        return torch.sigmoid(x)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        x = torch.mean(cla, dim=2)

        return x, []


class MHeadAttention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num=4):
        super(MHeadAttention, self).__init__()

        self.head_num = head_num

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.ModuleList([])
        self.cla = nn.ModuleList([])
        for i in range(self.head_num):
            self.att.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
            self.cla.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        self.head_weight = nn.Parameter(torch.tensor([1.0/self.head_num] * self.head_num))

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self.activate(att, self.att_activation)

            cla = self.cla[i](x)
            cla = self.activate(cla, self.cla_activation)

            att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
            cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            norm_att = att / torch.sum(att, dim=2)[:, :, None]
            x_out.append(torch.sum(norm_att * cla, dim=2) * self.head_weight[i])

        x = (torch.stack(x_out, dim=0)).sum(dim=0)

        return x, []


class ResNetAttention(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(ResNetAttention, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=False)

        if pretrain == False:
            print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
        else:
            print('Now Use ImageNet Pretrained ResNet50 Model.')

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the original ImageNet classification layers to save space.
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = torch.nn.Identity()

        # attention pooling module
        self.attention = Attention(
            2048,
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape([batch_size, 2048, 4, 33])
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out


class MBNet(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(MBNet, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

    def forward(self, x, nframes):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out


class EffNetAttention(nn.Module, audobject.Object):
    def __init__(
            self,
            label_dim=527,
            b=0,
            pretrain=True,
            head_num=4,
            augment='None'
    ):
        super(EffNetAttention, self).__init__()
        self.label_dim = label_dim
        self.b = b
        self.pretrain = pretrain
        self.head_num = head_num
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.augment = augment

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = augmentation.FilterAugment()

        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

        self.avgpool = nn.AvgPool2d((4, 1))
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()

    def forward(self, x, nframes=1056, mixup_lambda=None):
        # expect input x = (batch_size, 1, time_frame_num, frequency_bins), e.g., (B, 1, 500, 128)

        # data augmentation:
        if self.augment == 'sa':
            x = self.spec_augmenter(x)
        elif self.augment == 'fa':
            x = self.filter_augmenter(x)

        x = x.transpose(2, 3)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out


class CnnSpec(torch.nn.Module, audobject.Object):
    """
        Works on different spectral image representations, such as MFCCs, Mel-Spectrogram and Chromagrams
    """

    def __init__(
            self,
            tdim,
            fdim,
            output_dim: int,
            sigmoid_output: bool = False,
            segmentwise: bool = False,
            augment = False
    ):

        super().__init__()
        self.fdim = fdim
        self.tdim = tdim
        self.augment = augment

        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = torch.nn.BatchNorm2d(fdim)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=fdim)
        self.conv_block2 = ConvBlock(in_channels=fdim, out_channels=fdim*2)
        self.conv_block3 = ConvBlock(in_channels=fdim*2, out_channels=fdim*4)
        # self.conv_block4 = ConvBlock(in_channels=fdim*4, out_channels=fdim*8)

        self.fc1 = torch.nn.Linear(fdim*4, fdim*4, bias=True)
        self.out = torch.nn.Linear(fdim*4, output_dim, bias=True)

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = augmentation.FilterAugment()
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.out)

    def get_embedding(self, x):

        if self.train and self.augment:
            x = self.spec_augmenter(x)
            #x = self.filter_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        if self.segmentwise:
            return self.segmentwise_path(x)
        else:
            return self.clipwise_path(x)

    def segmentwise_path(self, x):
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def forward(self, x):
        """ assumes x = (B, f_dim, t_dim) or (B, 1, f_dim, t_dim)"""
        if x.ndim == 3:
            x = x.unsqueeze(1)  # add channel dimension
        # if x.shape[2:] != torch.Size([self.fdim, self.tdim]):
        #     x = x.permute(0,1,3,2)

        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x



if __name__ == "__main__":

    batch_size = 16
    sample_rate = 16000
    wdw_size = 500
    wdw = 5  # seconds
    n_fft = 320  # 320
    f_min = 0  # 50
    f_max = 256  # 8000
    n_mels = 64  # 64
    input_tdim = 500

    # waveform batch
    wav = torch.zeros([batch_size, sample_rate * wdw])  # [16, 80000]
    # spec batch
    test_input = torch.rand([10, 1, 64, 500])

    # EFFNET :

    # ast_mdl = ResNetNewFullAttention(pretrain=False)
    mdl = EffNetAttention(label_dim=1, pretrain=False, b=0, head_num=0)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_output = mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    # CNN10SED :
    #mdl = Cnn10SED(sigmoid_output=True)
    #test_input = torch.rand([25, 1, input_tdim, n_mels])
    #test_output = mdl(test_input)
    #print(test_output.shape)

    mdl = ResNet38(output_dim=1, sigmoid_output=True)

    from torchsummary import summary
    summary(mdl.cuda(), input_size=(1, input_tdim, n_mels))













