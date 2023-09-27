import audobject
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
# from torchsummary import summary
from torchinfo import summary


class config:
    r"""Get/set defaults for the :mod:`audfoo` module."""

    transforms = {
        16000: {
            'n_fft': 512,
            'win_length': 512,
            'hop_length': 160,
            'n_mels': 64,
            'f_min': 50,
            'f_max': 8000,
        }
    }


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    torch.nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_weights2(self):
    """Initialize the weights of the convolutional and fully connected layers using
    Xavier initialization."""
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] + \
           x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out


def do_mixup2(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0,-1) * mixup_lambda + torch.flip(x, dims = [0]).transpose(0,-1) * (1 - mixup_lambda)).transpose(0,-1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def summarize_model(model, inp=(1, 1, 500, 64)):
    summary(model, input_size=inp)


def main():

    pass


if __name__ == '__main__':
    main()






















































































