# @title Prepare data and utility functions. {display-mode: "form"}
# @markdown
# @markdown You do not need to look into this cell.
# @markdown Just execute once and you are good to go.
# @markdown
# @markdown In this tutorial, we will use a speech data from [VOiCES dataset](https://iqtlabs.github.io/voices/),
# @markdown which is licensed under Creative Commos BY 4.0.

# -------------------------------------------------------------------------------
# Preparation of data and helper functions.
# -------------------------------------------------------------------------------
import librosa
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import (
    Resample, Spectrogram, TimeStretch, TimeMasking, FrequencyMasking, MelScale
)


# SAMPLE_WAV_SPEECH_PATH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_SPEECH_PATH = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\kirun-data-by-runner\IfWBgNFvI9Yob8LdErhNeecrrgF2\b67ka\-MgufDWjXVMOCe1gVKsy\-MgufDWjXVMOCe1gVKsy___2021-08-12T14-03-35-469Z.wav'


def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.load(path)


def get_speech_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)


def get_spectrogram(
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)

def get_melspectrogram(
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = get_speech_sample()
    spectrogram = T.MelSpectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect, cmap='inferno')
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


if __name__ == '__main__':

    # TimeStretch
    spec = get_spectrogram()
    stretch = T.TimeStretch()
    #plot_spectrogram(torch.abs(spec[0]), title="Original", aspect="equal", xmax=304)
    rate = 1.2
    spec_ = stretch(spec, rate)
    #plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect="equal", xmax=304)

    # rate = 2.0
    # spec_ = stretch(spec, rate)
    # plot_spectrogram(torch.abs(spec_[0]), title=f"Stretched x{rate}", aspect="equal", xmax=304)

    spec = get_melspectrogram(win_len=320, hop_len=160, n_fft=320)
    # print(spec.shape)
    # print(spec[0].shape)
    # print(spec[0, :, :500].shape)
    #
    plot_spectrogram(torch.abs(spec[0, :, :500]), title=f"Stretched x{rate}", aspect="equal", xmax=304)
    #
    # import tools.augmentation as aug
    # spec = spec[:, :, :500]
    # plot_spectrogram(torch.abs(spec[0]), title=f"normal", aspect="equal", xmax=304)
    # alpha=0.1
    # mixup = aug.Mixup(alpha=alpha)
    # spec = mixup(spec)
    # plot_spectrogram(torch.abs(spec[0]), title=f"mixup, alpha={alpha}", aspect="equal", xmax=304)
    #
    # spec_augmenter = aug.SpecAugmentation(time_drop_width=64, time_stripes_num=2,
    #                                        freq_drop_width=8, freq_stripes_num=2)  # 2 2
    #
    # filter_augmenter = aug.FilterAugment()
    #
    # spec = spec_augmenter(spec)
    # plot_spectrogram(torch.abs(spec[0]), title=f"spec_augmenter", aspect="equal", xmax=304)
    #
    # spec = filter_augmenter(spec)
    # plot_spectrogram(torch.abs(spec[0]), title=f"filter_augmenter", aspect="equal", xmax=304)





    # TimeMasking
    torch.random.manual_seed(4)
    spec = get_spectrogram()
    #plot_spectrogram(torch.abs(spec[0]), title="Original", xmax=304)
    masking = T.TimeMasking(time_mask_param=80)
    spec = masking(spec)
    #plot_spectrogram(torch.abs(spec[0]), title="Masked along time axis", aspect="equal", xmax=304)

    # FrequencyMasking
    torch.random.manual_seed(4)
    masking = T.FrequencyMasking(freq_mask_param=160)
    spec = masking(spec)
    #plot_spectrogram(torch.abs(spec[0]), title="Masked along frequency axis", aspect="equal", xmax=304)

