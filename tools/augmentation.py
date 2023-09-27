import os
import audtorch
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
from torch.nn.functional import conv1d, conv2d
import torch

# --------------------------------------------------------------------------------------------
# PassT feature augmentation:

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window)
        x = (x ** 2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                 vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )


# --------------------------------------------------------------------------------------------
# SpecAugment: https://arxiv.org/abs/1904.08779
# https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
from matplotlib import pyplot as plt
from tools.datasets import CachedDataset


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()
        assert dim in [2, 3]  # dim 2: time; dim 3: frequency
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, features):
        """input: (batch_size, channels, time_steps, freq_bins)"""
        assert features.ndimension() == 4
        if self.training is False:
            return features
        else:
            batch_size = features.shape[0]
            total_width = features.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(features[n], total_width)
            return features

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
            if self.dim == 2:
                e[:, bgn: bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn: bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """
        super(SpecAugmentation, self).__init__()
        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num)
        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


# --------------------------------------------------------------------------------------------
# FilterAugment: https://arxiv.org/abs/2110.03282v4
# https://github.com/frednam93/FDY-SED/blob/main/utils/data_aug.py

class FilterAugment(nn.Module):
    def __init__(self, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
        super(FilterAugment, self).__init__()

        self.db_range = db_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type

        if not isinstance(filter_type, str):
            if torch.rand(1).item() < filter_type:
                self.filter_type = "step"
                self.n_band = [2, 5]
                self.min_bw = 4
            else:
                self.filter_type = "linear"
                self.n_band = [3, 6]
                self.min_bw = 6

    def forward(self, features):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        # -> (batch_size, freq_bins, time_steps):
        assert features.size(1) == 1, 'FilterAugment: more than one channel dimension.'
        features = features.squeeze(1)  # remove channel dim

        batch_size, n_freq_bin, _ = features.shape
        n_freq_band = torch.randint(low=self.n_band[0], high=self.n_band[1], size=(1,)).item()  # [low, high)

        if n_freq_band > 1:
            while n_freq_bin - n_freq_band * self.min_bw + 1 < 0:
                self.min_bw -= 1
            band_bndry_freqs = torch.sort(
                torch.randint(0, n_freq_bin - n_freq_band * self.min_bw + 1, (n_freq_band - 1,))
            )[0] + torch.arange(1, n_freq_band) * self.min_bw

            band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))
            if self.filter_type == "step":
                band_factors = torch.rand((batch_size, n_freq_band)).to(features) * \
                               (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                band_factors = 10 ** (band_factors / 20)
                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(
                        -1).unsqueeze(-1)

            elif self.filter_type == "linear":
                band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) \
                               * (self.db_range[1] - self.db_range[0]) + self.db_range[0]
                freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
                for i in range(n_freq_band):
                    for j in range(batch_size):
                        freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = \
                            torch.linspace(band_factors[j, i], band_factors[j, i + 1],
                                           band_bndry_freqs[i + 1] - band_bndry_freqs[i]).unsqueeze(-1)
                freq_filt = 10 ** (freq_filt / 20)
            return (features * freq_filt).unsqueeze(1)
        else:
            return features.unsqueeze(1)  # restore channel dim (?) and return


# --------------------------------------------------------------------------------------------


class Mixup(nn.Module):
    """see concrete implementation in training.py/mixup_collate_fn"""
    def __init__(
            self,
            mixup_type='eo',
            alpha=0.2,

    ):
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.mixup_type = mixup_type

    def mixup_even_odd(self, x, mixup_lambda):
        """Based on: https://github.com/qiuqiangkong/audioset_tagging_cnn"""
        """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
        (1, 3, 5, ...).
        Args:
          x: (batch_size * 2, ...)
          mixup_lambda: (batch_size * 2,)
        Returns:
          out: (batch_size, ...)
        """
        out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] + x[1:: 2].transpose(0, -1) * mixup_lambda[
                                                                                             1:: 2]).transpose(0, -1)
        return out

    def mixup(self, x, size):
        # TODO
        """https://github.com/fschmid56/EfficientAT/blob/main/helpers/utils.py"""
        """Performs mixup on examples from different batches."""
        assert type(x) == list, 'multiple batches'

        rn_indices = torch.randperm(size)
        lambd = np.random.beta(self.alpha, self.alpha, size).astype(np.float32)
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lam = torch.FloatTensor(lambd)
        return rn_indices, lam

    def forward(self, x):
        """ x : (Batchsize, 1, H, W)"""
        b = x.size(0)
        device = x.device
        mixup_lambda = torch.from_numpy(np.random.beta(self.alpha, self.alpha, size=b * 2)).float().to(device)
        if self.mixup_type == 'eo':
            xs = torch.cat((x, x), dim=0)  # mix batch with itself according to even/odd indices
            x = self.mixup_even_odd(xs, mixup_lambda)
        else:
            print('undefined mixup type')
        return x


# --------------------------------------------------------------------------------------------

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


# --------------------------------------------------------------------------------------------


if __name__ == '__main__':

    data_root = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\wdw5\fold0'
    features_p = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\melspec_data\wdw5\fold0\features.csv'

    window_size = 500
    batch_size = 32

    # Recreating Training-Data environment:
    # Read data:
    df_train = pd.read_csv(os.path.join(data_root, 'train.csv'))
    df_train.set_index(['file', 'start', 'end'], inplace=True)
    df_dev = pd.read_csv(os.path.join(data_root, 'devel.csv'))
    df_dev.set_index(['file', 'start', 'end'], inplace=True)
    df_test = pd.read_csv(os.path.join(data_root, 'test.csv'))
    df_test.set_index(['file', 'start', 'end'], inplace=True)

    features = pd.read_csv(features_p).set_index(['file', 'start', 'end'])
    features['features'] = features['features'].apply(
        lambda x: os.path.join(os.path.dirname(features_p), x)
    )
    db_args = {
        'features': features,
        'target_column': 'steps',
        'transform': audtorch.transforms.RandomCrop(window_size, axis=-2)
        # https://audtorch.readthedocs.io/en/0.4.1/api-transforms.html
    }
    db_class = CachedDataset
    train_dataset = db_class(
        df_train.copy(),
        **db_args
    )
    x, y = train_dataset[0]

    dev_dataset = db_class(
        df_dev.copy(),
        **db_args
    )

    test_dataset = db_class(
        df_test.copy(),
        **db_args
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False
    )

    # get a sample:
    randind = np.random.randint(0, batch_size)
    sample = next(iter(train_loader))[0]
    print(sample.shape)  # torch.Size([32, 1, 500, 64])
    cmap = ['inferno', 'plasma', 'magma', 'cividis', 'Greys', 'coolwarm', 'viridis']
    cmap = ['viridis']
    more_maps = [
        'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
        'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
        'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
        'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
        'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
        'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2',
        'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
        'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
        'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
        'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
        'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
        'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
        'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r',
        'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma',
        'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma',
        'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
        'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
        'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
        'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    ]

    show_all_augments = True
    stretchb = False
    stretch = torchaudio.transforms.TimeStretch(hop_length=160, n_freq=64)
    stretch_factor = 0.5

    if show_all_augments:
        for b in range(batch_size):
            for map in cmap:
                fig, axs = plt.subplots(2, 2, figsize=(20, 8))


                sample_s = sample[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                if stretchb:
                    sample_s = stretch(sample_s.permute(1, 0, 2), stretch_factor).float()
                    sample_s = sample_s[:, :, 0].unsqueeze(2).permute(1, 0, 2)

                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[0, 0].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                # axs[0, 0].set_xlabel("time (s)")
                axs[0, 0].set_ylabel("frequency (hz)")
                axs[0, 0].set_title('Original')
                fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])  # add colorbar to the subplot
                # plt.show()

                # SPECAUGMENT

                spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                                  freq_drop_width=16, freq_stripes_num=2)

                # Training stage
                spec_augmenter.train()  # set to spec_augmenter.eval() for evaluation
                result = spec_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                # plt.imshow(sample_s, origin="lower", aspect="auto", cmap=cmap, interpolation="none")
                # plt.colorbar()
                # plt.xlabel("time (s)")
                # plt.ylabel("frequency (hz)")
                # plt.title('SpecAugment')
                axs[0, 1].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                # axs[0, 1].set_xlabel("time (s)")
                axs[0, 1].set_ylabel("frequency (hz)")
                axs[0, 1].set_title('SpecAugment')
                fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])  # add colorbar to the subplot
                # plt.show()

                # FILTERAUGMENT

                filter_augmenter = FilterAugment()
                filter_augmenter.train()
                result = filter_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[1, 0].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[1, 0].set_xlabel("tdim")
                axs[1, 0].set_ylabel("frequency (hz)")
                axs[1, 0].set_title('FilterAugment')
                fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])  # add colorbar to the subplot
                # plt.show()

                # MIXUP

                alpha = 0.2
                mixup_augmenter = Mixup()
                result = mixup_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[1, 1].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[1, 1].set_xlabel("tdim")
                axs[1, 1].set_ylabel("frequency (hz)")
                axs[1, 1].set_title('Mixup-even-odd')
                fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])  # add colorbar to the subplot

                plt.show()

    show_all_mixup = False

    if show_all_mixup:
        for b in range(batch_size):
            for map in cmap:
                fig, axs = plt.subplots(2, 2, figsize=(20, 8))

                # MIXUP

                alpha = 0.0
                mixup_augmenter = Mixup()
                result = mixup_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[0, 0].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[0, 0].set_xlabel("time (s)")
                axs[0, 0].set_ylabel("frequency (hz)")
                axs[0, 0].set_title(f'Mixup: alpha={alpha}')
                fig.colorbar(axs[0, 0].images[0], ax=axs[0, 0])  # add colorbar to the subplot

                alpha = 0.2
                mixup_augmenter = Mixup()
                result = mixup_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[0, 1].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[0, 1].set_xlabel("time (s)")
                axs[0, 1].set_ylabel("frequency (hz)")
                axs[0, 1].set_title(f'Mixup: alpha={alpha}')
                fig.colorbar(axs[0, 1].images[0], ax=axs[0, 1])  # add colorbar to the subplot

                alpha = 0.5
                mixup_augmenter = Mixup()
                result = mixup_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[1, 0].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[1, 0].set_xlabel("time (s)")
                axs[1, 0].set_ylabel("frequency (hz)")
                axs[1, 0].set_title(f'Mixup: alpha={alpha}')
                fig.colorbar(axs[1, 0].images[0], ax=axs[1, 0])  # add colorbar to the subplot

                alpha = 1.0
                mixup_augmenter = Mixup()
                result = mixup_augmenter(sample)
                print(result.shape)
                sample_s = result[b]
                print(sample_s.shape)
                sample_s = sample_s.permute(2, 1, 0)
                print(sample_s.shape)
                # plt.figure(figsize=(15, 7.5))
                axs[1, 1].imshow(sample_s, origin="lower", aspect="auto", cmap=map, interpolation="none")
                axs[1, 1].set_xlabel("time (s)")
                axs[1, 1].set_ylabel("frequency (hz)")
                axs[1, 1].set_title(f'Mixup: alpha={alpha}')
                fig.colorbar(axs[1, 1].images[0], ax=axs[1, 1])  # add colorbar to the subplot

                plt.show()