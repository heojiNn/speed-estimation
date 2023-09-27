import audobject
import audtorch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio

from tools import augmentation
from models.util import init_layer
from models.cnn import ConvBlock
from scipy.signal import butter, filtfilt

class DNN(nn.Module, audobject.Object):
    """
                Layer (type)               Output Shape         Param #
    ================================================================
         FilterAugment-1           [-1, 1, 500, 64]               0
                Linear-2                  [-1, 256]       8,192,256
                  ReLU-3                  [-1, 256]               0
                Linear-4                  [-1, 256]          65,792
                  ReLU-5                  [-1, 256]               0
                Linear-6                  [-1, 256]          65,792
                  ReLU-7                  [-1, 256]               0
                Linear-8                  [-1, 256]          65,792
                  ReLU-9                  [-1, 256]               0
               Linear-10                  [-1, 256]          65,792
                 ReLU-11                  [-1, 256]               0
               Linear-12                    [-1, 1]             257
    ================================================================
    Total params: 8,455,681
    Trainable params: 8,455,681
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.12
    Forward/backward pass size (MB): 0.26
    Params size (MB): 32.26
    Estimated Total Size (MB): 32.64
    ----------------------------------------------------------------
    """
    def __init__(
            self,
            input_size=500 * 64,
            hidden_size=256,
            batch_norm=False,
            augment=True,
            out_dim=1,
            sigmoid_output: bool = False,
    ):
        super(DNN, self).__init__()

        # audobject vars:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.augment = augment
        self.out_dim = out_dim
        self.sigmoid_output = sigmoid_output

        # augmentations:
        self.spec_augmenter = augmentation.SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2
        )  # 2 2
        self.filter_augmenter = augmentation.FilterAugment()

        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x):
        if self.training and self.augment:
            # x = self.spec_augmenter(x)
            x = self.filter_augmenter(x)

        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, H * W]; e.g.
        x = self.dnn(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class CDNN(torch.nn.Module, audobject.Object):
    def __init__(
            self,
            output_dim: int = 1,
            sigmoid_output: bool = False,
            segmentwise: bool = False
    ):

        super().__init__()
        self.output_dim = output_dim
        self.sigmoid_output = sigmoid_output
        self.segmentwise = segmentwise

        self.bn0 = torch.nn.BatchNorm2d(64)

        # cnn layers:
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # dnn layers:
        self.dnn = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.out = torch.nn.Linear(512, output_dim, bias=True)

        self.init_weight()

    def init_weight(self):
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
        x = self.dnn(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def clipwise_path(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dnn(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.out(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


def main():
    from torchsummary import summary
    summary(DNN().cuda(), input_size=(1, 500, 64))

    from torchviz import make_dot
    inputsize = 16000 * 5
    model = DNN(inputsize, augment=False)
    x = torch.randn(1, inputsize)
    y = model(x)
    make_dot(y, show_saved=True, params=dict(model.named_parameters())).render("my_model")


if __name__ == "__main__":
    main()
