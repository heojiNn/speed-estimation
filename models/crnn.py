import warnings

import audobject
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import audtorch
import torchaudio

from models.cnn import ConvBlock, ConvPreWavBlock
from models.util import init_layer, do_mixup, init_bn



""" FilterAugSED - https://github.com/frednam93/FilterAugSED/blob/main/utils/model.py """


class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):  # x.size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1))  # [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2)  # [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        # self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class FACNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch"):
        super(FACNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):
        x = self.cnn(x)
        return x

import torchaudio.transforms as T

class FACRNN(nn.Module, audobject.Object):
    """
                  Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1          [-1, 64, 500, 64]             640
               BatchNorm2d-2          [-1, 64, 500, 64]             128
                    Linear-3          [-1, 500, 64, 64]           4,160
                   Sigmoid-4          [-1, 64, 500, 64]               0
                       GLU-5          [-1, 64, 500, 64]               0
                   Dropout-6          [-1, 64, 500, 64]               0
                 AvgPool2d-7          [-1, 64, 500, 16]               0

                    Conv2d-8          [-1, 64, 500, 16]          36,928
               BatchNorm2d-9          [-1, 64, 500, 16]             128
                   Linear-10          [-1, 500, 16, 64]           4,160
                  Sigmoid-11          [-1, 64, 500, 16]               0
                      GLU-12          [-1, 64, 500, 16]               0
                  Dropout-13          [-1, 64, 500, 16]               0
                AvgPool2d-14           [-1, 64, 500, 4]               0

                   Conv2d-15           [-1, 64, 500, 4]          36,928
              BatchNorm2d-16           [-1, 64, 500, 4]             128
                   Linear-17           [-1, 500, 4, 64]           4,160
                  Sigmoid-18           [-1, 64, 500, 4]               0
                      GLU-19           [-1, 64, 500, 4]               0
                  Dropout-20           [-1, 64, 500, 4]               0
                AvgPool2d-21           [-1, 64, 500, 1]               0

                    FACNN-22           [-1, 64, 500, 1]               0

                      GRU-23  [[-1, 500, 512], [-1, 2, 256]]          0
                    BiGRU-24             [-1, 500, 512]               0

                  Dropout-25             [-1, 500, 512]               0
                   Linear-26               [-1, 500, 1]             513
                  Sigmoid-27               [-1, 500, 1]               0

                   Linear-28               [-1, 500, 1]             513
                  Softmax-29               [-1, 500, 1]               0
        ================================================================
        Total params: 88,386
        Trainable params: 88,386
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.12
        Forward/backward pass size (MB): 867.66
        Params size (MB): 0.34
        Estimated Total Size (MB): 868.12
        ----------------------------------------------------------------
    """
    def __init__(self,
                 n_input_ch,
                 n_class,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 wav=False,
                 concat=False,
                 **convkwargs
                 ):
        super(FACRNN, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_class = n_class
        self.activation = activation
        self.conv_dropout = conv_dropout
        self.n_RNN_cell = n_RNN_cell
        self.n_RNN_layer = n_RNN_layer
        self.rec_dropout = rec_dropout
        self.attention = attention
        self.convkwargs = convkwargs
        self.wav = wav
        self.concat = concat

        if self.concat:
            self.cnn = FACNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
            self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)
            self.dropout = nn.Dropout(conv_dropout)
            self.sigmoid = nn.Sigmoid()
            self.dense = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention:
                self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
                self.softmax = nn.Softmax(dim=-1)
        else:  # omit recurrent layer
            self.cnn = FACNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
            self.dropout = nn.Dropout(conv_dropout)
            self.sigmoid = nn.Sigmoid()
            self.dense = nn.Linear(self.cnn.n_filt[-1], n_class)
            if self.attention:
                self.dense_softmax = nn.Linear(self.cnn.n_filt[-1], n_class)
                self.softmax = nn.Softmax(dim=-1)


    def wav2spec(self, x):
        # Log mel spectrogram
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=320,  # 320
            hop_length=160,
            f_min=50,  # 50
            f_max=8000,  # 8000
            n_mels=64,  # 64
        )
        x = transform(x.cpu())  # -> (B, NMELS, WDW), i.e. (32, 64, 500)
        x = x.permute(0, 2, 1)
        x = x[:, None, :]  # -> (B, 1, NMELS, WDW), i.e. (32, 1, 64, 500)
        transform = audtorch.transforms.RandomCrop(20, axis=-2)
        x = transform(x)
        # print(x.shape)  # torch.Size([32, 1, 20, 64])
        x = x.permute(0, 1, 3, 2)  # # torch.Size([32, 1, 64, 20])

        # check different inputs:
        stretch = T.TimeStretch(n_freq=64, hop_length=160)
        rate = 0.04
        x = stretch(x, rate)  # torch.Size([32, 1, 64, 500])
        x = x.permute(0, 1, 3, 2).cuda().float()  # torch.Size([32, 1, 500, 64])
        return x

    def forward(self, x):  # input size : [bs, freqs, frames]
        # cnn

        if self.wav:
            x = self.wav2spec(x)

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous.view(bs, frame, ch * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # x size : [bs, frames, chan]

        if self.concat:
            # rnn
            x = self.rnn(x)  # x size : [bs, frames, 2 * chan]
            x = self.dropout(x)

        # classifier
        strong = self.dense(x)  # strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # sof size : [bs, frames, n_class]
            sof = self.softmax(sof)  # sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, n_class]
        else:
            weak = strong.mean(1)

        # return strong.transpose(1, 2), weak
        return weak


""" FDYAugSED - https://github.com/frednam93/FDY-SED/blob/main/utils/model.py """


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(
            torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
            requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):  # x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)  # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)  # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1)  # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size,
                                            self.kernel_size)  # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  # x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  # x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  # x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)  # x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)  # x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)  # x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)  # x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)


class FDYCNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq'):
        super(FDYCNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            if DY_layers[i] == 1:
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i],
                                                                   n_basis_kernels=n_basis_kernels,
                                                                   temperature=temperature, pool_dim=pool_dim))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class FDYCRNN(nn.Module, audobject.Object):
    """
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1          [-1, 64, 500, 64]             640
               BatchNorm2d-2          [-1, 64, 500, 64]             128
                    Linear-3          [-1, 500, 64, 64]           4,160
                   Sigmoid-4          [-1, 64, 500, 64]               0
                       GLU-5          [-1, 64, 500, 64]               0
                   Dropout-6          [-1, 64, 500, 64]               0
                 AvgPool2d-7          [-1, 64, 500, 16]               0

                    Conv1d-8              [-1, 16, 500]           3,072
               BatchNorm1d-9              [-1, 16, 500]              32
                     ReLU-10              [-1, 16, 500]               0

                   Conv1d-11               [-1, 4, 500]              68
              attention2d-12               [-1, 4, 500]               0

           Dynamic_conv2d-13          [-1, 64, 500, 16]         147,456
              BatchNorm2d-14          [-1, 64, 500, 16]             128
                   Linear-15          [-1, 500, 16, 64]           4,160
                  Sigmoid-16          [-1, 64, 500, 16]               0
                      GLU-17          [-1, 64, 500, 16]               0
                  Dropout-18          [-1, 64, 500, 16]               0
                AvgPool2d-19           [-1, 64, 500, 4]               0

                   Conv1d-20              [-1, 16, 500]           3,072
              BatchNorm1d-21              [-1, 16, 500]              32
                     ReLU-22              [-1, 16, 500]               0

                   Conv1d-23               [-1, 4, 500]              68
              attention2d-24               [-1, 4, 500]               0

           Dynamic_conv2d-25           [-1, 64, 500, 4]         147,456
              BatchNorm2d-26           [-1, 64, 500, 4]             128
                   Linear-27           [-1, 500, 4, 64]           4,160
                  Sigmoid-28           [-1, 64, 500, 4]               0
                      GLU-29           [-1, 64, 500, 4]               0
                  Dropout-30           [-1, 64, 500, 4]               0
                AvgPool2d-31           [-1, 64, 500, 1]               0

                   FDYCNN-32           [-1, 64, 500, 1]               0

                      GRU-33  [[-1, 500, 512], [-1, 2, 256]]          0
                    BiGRU-34             [-1, 500, 512]               0

                  Dropout-35             [-1, 500, 512]               0
                   Linear-36               [-1, 500, 1]             513

                  Sigmoid-37               [-1, 500, 1]               0
                   Linear-38               [-1, 500, 1]             513
                  Softmax-39               [-1, 500, 1]               0
        ================================================================
        Total params: 315,786
        Trainable params: 315,786
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 0.12
        Forward/backward pass size (MB): 867.23
        Params size (MB): 1.20
        Estimated Total Size (MB): 868.56
        ----------------------------------------------------------------

    """
    def __init__(self,
                 n_input_ch,
                 n_class=1,
                 activation="glu",
                 conv_dropout=0.5,
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(FDYCRNN, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_class = n_class
        self.activation = activation
        self.conv_dropout = conv_dropout
        self.n_RNN_cell = n_RNN_cell
        self.n_RNN_layer = n_RNN_layer
        self.rec_dropout = rec_dropout
        self.attention = attention
        self.convkwargs = convkwargs

        self.cnn = FDYCNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            self.softmax = nn.Softmax(dim=1)
            # if self.attention == "time":
            #     self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            # elif self.attention == "class":
            #     self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def forward(self, x):  # input size : [bs, freqs, frames]
        # cnn

        # if self.n_input_ch > 1:
        #     x = x.transpose(2, 3)
        # else:
        #     x = x.transpose(1, 2).unsqueeze(1) # x size : [bs, chan, frames, freqs]

        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous.view(bs, frame, ch * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # x size : [bs, frames, chan]

        # rnn
        x = self.rnn(x)  # x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        # classifier
        strong = self.dense(x)  # strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # sof size : [bs, frames, n_class]
            sof = self.softmax(sof)  # sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, n_class]
        else:
            weak = strong.mean(1)

        # return strong.transpose(1, 2), weak
        return weak


""" Split Conv, BiGru, SED-project [SSED] """


# https://www.researchgate.net/publication/340349516_Joint_Optimization_of_Deep_Neural_Network-Based_Dereverberation_and_Beamforming_for_Sound_Event_Detection_in_Multi-Channel_Environments


class BiConvBlock(nn.Module, audobject.Object):
    def __init__(
            self
    ):
        super(BiConvBlock, self).__init__()
        self.cnn_left = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(32)
        )
        self.cnn_right = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cnn_tail = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=1)

    def forward(self, x):
        y_left = self.cnn_left(x)
        y_right = self.cnn_right(x)
        y = torch.cat((y_left, y_right), dim=-1)
        y = self.cnn_tail(y)
        y = y.view(y.size(0), -1)
        return y


class BiGRUBlock(nn.Module):
    def __init__(
            self,
            input_size=64 * 66 * 19,
            hidden_size=256,
            num_layers=1,
            device='cnn:0',
            dropout_p=0.2,
    ):
        super(BiGRUBlock, self).__init__()
        # members
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        # layers
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True,
                            dropout=dropout_p, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, H * W]
        x, _ = self.bigru(x)  # [B, H * W] -> [B, hidden_size * 2]
        x = torch.tanh(x)
        return x


class SSEDCRNN(nn.Module, audobject.Object):
    def __init__(
            self
    ):
        super(SSEDCRNN, self).__init__()

        self.cnn = BiConvBlock()
        self.rnn = BiGRUBlock(num_layers=5)  # [B, 512]
        self.fc1 = nn.Linear(512, 256)  # [B, 256]
        self.fc2 = nn.Linear(256, 128)  # [B, 128]
        self.fc3 = nn.Linear(128, 1)  # [B, 1]

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


"""
    BiLSTM-CRNN
    https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
"""


class BidirectionalLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            num_hidden,
            num_layers,
            num_out=1
    ):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, num_hidden, bidirectional=True, num_layers=num_layers)
        self.embedding = nn.Linear(num_hidden * 2, num_out)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class BiLSTMCRNN(nn.Module, audobject.Object):
    def __init__(
            self,
            n_mels=64,
            num_channels=1,
            num_class=1,
            num_hidden=256,
            num_layers=2,
            leakyRelu=False
    ):

        super(BiLSTMCRNN, self).__init__()
        self.n_mels = n_mels
        self.num_channels = num_channels
        self.num_class = num_class
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.leakyRelu = leakyRelu

        assert n_mels % 16 == 0, 'n_mels has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = num_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, num_hidden, num_hidden, num_layers),
            BidirectionalLSTM(num_hidden, num_hidden, num_class, num_layers)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


"""
    DCASE 2017 Winner
    https://github.com/yongxuUSTC/dcase2017_task4_cvssp/blob/main/main_crnn_sed.py
"""



""" 
    DCASE2020 task4: Sound event detection in domestic environments using source separation
    https://github.com/turpaultn/dcase20_task4
    
    comment: equals FilterAugSED almost identical
"""


class CNN(nn.Module, audobject.Object):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(CNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.activation = activation
        self.conv_dropout = conv_dropout
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.nb_filters = nb_filters
        self.pooling = pooling

        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

        self.cnn = cnn

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        return x


class CRNN_DC20(nn.Module, audobject.Object):

    @audobject.init_decorator(hide=['dropout'])  # RuntimeError: Arguments ['dropout'] of <class 'models.crnn.CRNN_DC20'> not assigned to attributes of same name.
    def __init__(
            self,
            n_in_channel=1,
            n_class=1,
            attention=False,
            activation="Relu",
            dropout=0,
            train_cnn=True,
            rnn_type='BGRU',
            n_RNN_cell=64,
            n_layers_RNN=1,
            dropout_recurrent=0,
            cnn_integration=False,
            **kwargs
    ):
        super(CRNN_DC20, self).__init__()

        self.n_in_channel = n_in_channel
        self.n_class = n_class
        self.attention = attention
        self.activation = activation
        self.dropout = dropout
        self.train_cnn = train_cnn
        self.rnn_type = rnn_type
        self.n_RNN_cell = n_RNN_cell
        self.n_layers_RNN = n_layers_RNN
        self.dropout_recurrent = dropout_recurrent
        self.cnn_integration = cnn_integration
        self.kwargs = kwargs

        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if rnn_type == 'BGRU':
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BiGRU(nb_in, n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq), e.g. [16, 1, 500, 64]
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        # return strong, weak
        return weak

""" SELFMADE """


class CNNLSTM1(nn.Module, audobject.Object):
    """
    # Example:

        model = CNNLSTMRegressor(input_size=64, hidden_size=128, output_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) \

        # Generate some dummy input data of shape (batch_size, 1, height, width)
        batch_size = 16
        height = 500
        width = 64
        x = torch.randn(batch_size, 1, height, width).to(device)
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            num_layers=1
    ):
        super(CNNLSTM1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # self.bn0 = torch.nn.BatchNorm2d(64) # didn't really improve the results...

        # 4 CONV BLOCKS
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, -1, self.hidden_size)

        lstm_output, (h_n, c_n) = self.lstm(cnn_output)
        output = self.linear(lstm_output[:, -1, :])

        return output

    def init_weights(self):
        init_layer(self.linear)


class CNNLSTM2(nn.Module, audobject.Object):
    """
    # Example:

        model = CNNLSTMRegressor(input_size=64, hidden_size=128, output_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) \

        # Generate some dummy input data of shape (batch_size, 1, height, width)
        batch_size = 16
        height = 500
        width = 64
        x = torch.randn(batch_size, 1, height, width).to(device)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(CNNLSTM2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 5 CONV BLOCKS
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(batch_size, -1, self.hidden_size)
        lstm_output, (h_n, c_n) = self.lstm(cnn_output)
        output = self.linear(lstm_output[:, -1, :])
        return output


class CNN10LSTM(nn.Module, audobject.Object):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            num_layers=1,
            sigmoid_output: bool = False,
            segmentwise: bool = False,
            bidirectional = False
    ):

        super(CNN10LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise
        self.bidirectional = bidirectional

        # CNN10
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = torch.nn.Linear(512, 512, bias=True)

        # LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers,
                            bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(2*hidden_size, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

        self.init_weight()

    def forward(self, x):
        batch_size = x.size(0)
        cnn_output = self.get_embedding(x)

        if self.sigmoid_output:
            cnn_output = torch.sigmoid(cnn_output)
        cnn_output = cnn_output.view(batch_size, -1, self.hidden_size)

        lstm_output, _ = self.lstm(cnn_output)
        output = self.linear(lstm_output[:, -1, :])
        return output

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.linear)
        init_layer(self.fc1)

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


class WavegramLogmelCnn14LSTM(nn.Module, audobject.Object):
    def __init__(
            self,
            sample_rate=16000,
            window_size=500,
            hop_size=250,
            mel_bins=64,
            fmin=50,
            fmax=1024,
            classes_num=1,
            num_rnn_layers=1,
            rnn_hidden_dim=256,
            bidirectional=False,
            sigmoid_output: bool = False,
            segmentwise: bool = False,
    ):
        super(WavegramLogmelCnn14LSTM, self).__init__()
        # audtorch args:
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.classes_num = classes_num
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.bidirectional = bidirectional
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        # wv-cnn
        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=5, padding=5,
                                   bias=False)  # kernel_size = 11 initially
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # melspec-cnn:
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 1024, bias=True)

        # lstm
        self.lstm = nn.LSTM(input_size=256, hidden_size=rnn_hidden_dim, num_layers=num_rnn_layers,
                            bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(2 * rnn_hidden_dim, classes_num)
        else:
            self.linear = nn.Linear(rnn_hidden_dim, classes_num)

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.linear)

    def forward(self, input, mixup_lambda=None):
        """ Input: (batch_size, data_length) ; data_length should be Sample_rate * wdw_size"""

        B = input.size(0)

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=2)  # pool_size = 4 initially
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))  # torch.Size([16, 64, 125, 32])

        # # Log mel spectrogram
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=320,  # 320
            f_min=0,  # 50
            f_max=256,  # 8000
            n_mels=64,  # 64
        )
        x = transform(input.cpu())
        x = x.permute(0, 2, 1)
        x = x[:, None, :]
        transform = audtorch.transforms.RandomCrop(500, axis=-2)
        x = transform(x).cuda()

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        self.training = False  # ?
        if self.training:
            x = self.spec_augmenter(x)
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
        cnn_output = torch.mean(x, dim=3)

        if self.sigmoid_output:
            cnn_output = torch.sigmoid(cnn_output)
        # cnn_output = self.fc1(cnn_output)
        cnn_output = cnn_output.view(B, -1, self.rnn_hidden_dim)
        lstm_output, _ = self.lstm(cnn_output)
        output = self.linear(lstm_output[:, -1, :])
        return output


if __name__ == '__main__':

    # CRNN_DC20(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], pooling=[(1, 4), (1, 4), (1, 4)])

    fdim = 64
    tdim = 500
    hidden_size = 128
    num_layers = 2
    num_classes = 1

    # mdl = FACRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2, n_class=1)
    mdl = FDYCRNN(n_input_ch=1, n_RNN_cell=256, n_RNN_layer=2)

    batch = torch.randn([16, 1, tdim, fdim])
    preds = mdl(batch)

    #print(preds.shape)
    #print(preds)

    from torchsummary import summary
    summary(mdl.cuda(), input_size=(1, tdim, fdim))
