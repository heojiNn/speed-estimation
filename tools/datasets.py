import math
import audiofile
import audtorch
import librosa
import numpy as np
import pandas as pd
import torch

import ast


class CachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
            self,
            df: pd.DataFrame,
            features: pd.DataFrame,
            target_column: str,
            transform=None,
            target_transform=None,
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, 'features'])
        signal = signal.transpose(0, 2, 1)  # (1, 64, 501) -> (1, 501, 64)
        target = self.df[self.target_column].loc[index]

        # for multitargets as str
        if isinstance(target, str):
            target = ast.literal_eval(target.replace(".", ","))
            target = torch.tensor(target)
            # print('getting targets:', target, type(target), target.shape)

        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target


class WavDataset(torch.utils.data.Dataset):
    r"""Dataset of raw audio data.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
            self,
            df: pd.DataFrame,
            features: pd.DataFrame,
            target_column: str,
            transform=None,
            target_transform=None,
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, 'features'])
        # signal = audiofile.read(index, always_2d=False)[0]
        target = self.df[self.target_column].loc[index]

        # for multitargets as str
        if isinstance(target, str):
            target = ast.literal_eval(target.replace(".", ","))
            target = torch.tensor(target)
            # print('getting targets:', target, type(target), target.shape)

        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target


class SpecDataset(torch.utils.data.Dataset):
    r"""Dataset of spectral audio feature data.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
            self,
            df: pd.DataFrame,
            features: pd.DataFrame,
            target_column: str,
            _type='mfcc',
            transform=None,
            target_transform=None,
            window_size=5,  # secs
            sample_rate=16000,
            n_fft=320,
            window_length=320,
            hop_length=160,
            n_mels=64,
            n_mfcc=13,
            n_chroma=12,
            **kwargs
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self._type = _type
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mels = n_mels

        self.kwargs = kwargs

        self.indices = list(self.df.index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, 'features'])  # loading windowed audio file to waveform again

        # --------------------------------------------------------------------
        # Window-Normalization before feature extraction:

        target = self.df[self.target_column].loc[index]

        # for multitargets as str
        if isinstance(target, str):
            target = ast.literal_eval(target.replace(".", ","))
            target = torch.tensor(target)
            # print('getting targets:', target, type(target), target.shape)

        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)
        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # --------------------------------------------------------------------
        # Spectral Feature Extraction:

        if self._type == 'melspec':
            signal = librosa.feature.melspectrogram(signal, sr=self.sample_rate, n_fft=self.n_fft,
                                                    win_length=self.window_length, hop_length=self.window_length // 2,
                                                    n_mels=self.n_mels, center=False, fmin=50,
                                                    fmax=self.sample_rate // 2)
        elif self._type == 'chroma_stft':  # todo
            # center = False: disables zero-padding
            signal = librosa.feature.chroma_stft(signal, sr=self.sample_rate, n_chroma=self.n_chroma, n_fft=self.n_fft,
                                                 hop_length=self.hop_length, win_length=self.window_length,
                                                 center=False)
        elif self._type == 'chroma_cqt':  # todo
            signal = librosa.feature.chroma_cqt(signal, sr=self.sample_rate, n_chroma=self.n_chroma, hop_length=self.hop_length)
        elif self._type == 'chroma_cens':  # todo
            signal = librosa.feature.chroma_cens(signal, sr=self.sample_rate, n_chroma=self.n_chroma, center=False)
        elif self._type == 'mfcc':
            signal = librosa.feature.mfcc(signal, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_mels=self.n_mels,
                                          hop_length=self.hop_length,
                                          n_fft=self.n_fft, win_length=self.window_length, center=False)
        elif self._type == 'spectral_contrast':  # todo
            signal = librosa.feature.spectral_contrast(signal, sr=self.sample_rate, center=False)
        elif self._type == 'poly_features':  # todo
            signal = librosa.feature.poly_features(signal, sr=self.sample_rate, center=False)
        elif self._type == 'tonnetz':  # todo
            signal = librosa.feature.tonnetz(signal, sr=self.sample_rate, n_chroma=self.n_chroma, center=False)
        else:
            print(f'Type {self._type} not defined.')
            return RuntimeError()

        # --------------------------------------------------------------------

        return signal, target

    def __repr__(self):
        print(self.df, self.features, self.target_column, self._type, self.transform, self.target_transform,
              self.window_size, self.sample_rate, self.n_fft, self.hop_length, self.n_mfcc, self.kwargs)
